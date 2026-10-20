"""OpenTelemetry tracing setup for GANDALF.

In queue mode one query crosses two processes, and its trace follows it: the
API's request span carries the job id and ``job.enqueued`` /
``job.result_received`` events, the job carries the W3C trace context
(``Job.trace_headers``), and the worker parents its spans on it:

* ``gandalf.queue_wait`` -- from when the API accepted the query to when a
  worker picked it up (its start is the API's clock, its end the worker's);
* ``gandalf.job`` -- the worker's whole handling of the job, picked up to
  answered, with ``gandalf.execute`` (running the query) and
  ``gandalf.deliver`` (storing the result, or POSTing the callback) inside.

The ``opentelemetry`` packages are imported only by :func:`init_otel`, so
everything here is a no-op, importing nothing, while tracing is off.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from gandalf.config import settings

logger = logging.getLogger(__name__)

# Internal handles populated by init_otel() when tracing is enabled.  Left as
# None at import so that under ``gunicorn --preload`` the master (which imports
# this module before forking) builds no SDK objects.
_inject: Optional[Callable[[dict], None]] = None
_record_baggage_impl: Optional[Callable[[], None]] = None
_provider = None
_tracer = None
_initialized = False

#: Shape of a well-formed ``pk`` baggage value (the caller's query id).
_pk_match = re.compile(r"\A[0-9A-Za-z-]{36}\Z").match


class _NoSpan:
    """What :func:`span` yields while tracing is off: takes calls, records nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass


_NO_SPAN = _NoSpan()


def inject_headers(carrier: dict) -> None:
    """Inject the active W3C trace context (traceparent/baggage) into *carrier*.

    No-op until :func:`init_otel` runs with tracing enabled.  Used so outgoing
    requests -- specifically the async callback POST -- stay linked to the
    originating trace; the callback's httpx client is not auto-instrumented and
    the background threadpool does not inherit the request's contextvars, so the
    context must be captured into a header dict explicitly.
    """
    if _inject is not None:
        _inject(carrier)


def record_baggage() -> None:
    """Record selected incoming W3C baggage onto the current span.

    No-op until :func:`init_otel` runs with tracing enabled.  Baggage is
    extracted into the request context by the propagator but never attached to
    spans automatically, so this copies the values we care about across.
    """
    if _record_baggage_impl is not None:
        _record_baggage_impl()


def add_event(name: str, **attributes: Any) -> None:
    """Add a timestamped event to the current span (no-op while tracing is off)."""
    if _tracer is None:
        return
    from opentelemetry import trace

    trace.get_current_span().add_event(name, attributes)


def set_attributes(**attributes: Any) -> None:
    """Set attributes on the current span (no-op while tracing is off)."""
    if _tracer is None:
        return
    from opentelemetry import trace

    trace.get_current_span().set_attributes(attributes)


def _ns(epoch_seconds: float) -> int:
    """Wall-clock seconds as the nanoseconds OpenTelemetry timestamps are in."""
    return int(epoch_seconds * 1e9)


@contextmanager
def span(
    name: str,
    *,
    carrier: Optional[dict[str, str]] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> Iterator[Any]:
    """Run the block in a span, made current so spans inside nest under it.

    Args:
        name: The span name.
        carrier: W3C headers (``traceparent``, ``baggage``) to parent the
            span on, as a job carries them from the API; the current span
            when not given.  A ``pk`` in its baggage is recorded on the span.
        attributes: Set on the span as it starts.

    Yields:
        The span, or a stand-in that records nothing while tracing is off.
    """
    if _tracer is None:
        yield _NO_SPAN
        return
    context = _extract(carrier) if carrier is not None else None
    with _tracer.start_as_current_span(
        name, context=context, attributes=attributes
    ) as current:
        if context is not None:
            _record_pk(current, context)
        yield current


def record_span(
    name: str,
    *,
    start: float,
    end: float,
    carrier: Optional[dict[str, str]] = None,
    attributes: Optional[dict[str, Any]] = None,
) -> None:
    """Record a span that already happened, from *start* to *end* (epoch seconds).

    For an interval nobody could time as it passed, such as the time a job
    spent on the stream: the API that started it and the worker that ended
    it are different processes.
    """
    if _tracer is None:
        return
    context = _extract(carrier) if carrier is not None else None
    recorded = _tracer.start_span(
        name, context=context, attributes=attributes, start_time=_ns(start)
    )
    if context is not None:
        _record_pk(recorded, context)
    recorded.end(end_time=_ns(end))


def _extract(carrier: dict[str, str]) -> Any:
    """The OpenTelemetry context the W3C headers in *carrier* describe."""
    from opentelemetry.propagate import extract

    return extract(carrier)


def _record_pk(target: Any, context: Any = None) -> None:
    """Record a well-formed ``pk`` baggage entry of *context* on *target*."""
    from opentelemetry import baggage

    pk = baggage.get_baggage("pk", context)
    if isinstance(pk, str) and _pk_match(pk):
        target.set_attribute("pk", pk)


def init_otel(app=None) -> None:
    """Build the OpenTelemetry SDK for the current process and instrument *app*.

    Must run **once per worker, after any fork** -- never in a preloaded
    gunicorn master.  With ``preload_app = True`` the application module is
    imported in the master before workers are forked; constructing the SDK there
    would tie the OTLP/gRPC exporter channel and the ``BatchSpanProcessor``
    export thread to the master and leave them inherited in a broken state by
    the children (gRPC channels are not fork-safe).  Deferring construction to
    ``post_fork`` / worker startup gives each worker its own exporter, channel,
    and export thread.

    Idempotent and a no-op when ``settings.otel_enabled`` is False, so it is
    safe to call from both the gunicorn ``post_fork`` hook (production) and the
    FastAPI lifespan startup (the uvicorn dev server).

    Args:
        app: The FastAPI application to instrument; None in the queue
            worker, which has no app and makes its spans with :func:`span`.
            Its spans are exported on exit by the provider's own hook.
    """
    global _inject, _record_baggage_impl, _provider, _tracer, _initialized

    if _initialized or not settings.otel_enabled:
        return

    from opentelemetry import trace
    from opentelemetry.propagate import inject as _real_inject
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    resource = Resource(attributes={SERVICE_NAME: settings.otel_service_name})

    exporter: SpanExporter
    if settings.otel_use_console_exporter:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter(
            endpoint=f"{settings.jaeger_host}:{settings.jaeger_port}",
        )

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=provider,
            excluded_urls="docs,openapi.json",
            # receive/send spans are asgi events that are parts of a larger communication
            # they are more so noisy than useful, exclude them from traces
            exclude_spans=["receive", "send"],
        )
    _provider = provider
    # From this provider rather than the global one: a process sets the
    # global provider once, and tests build more than one.
    _tracer = provider.get_tracer("gandalf")
    _inject = _real_inject

    def _record() -> None:
        """Attach the incoming ``pk`` baggage value to the current span."""
        current = trace.get_current_span()
        if current.is_recording():
            _record_pk(current)

    _record_baggage_impl = _record
    _initialized = True

    logger.info(
        "OpenTelemetry tracing initialized in PID=%d (service=%s).",
        os.getpid(),
        settings.otel_service_name,
    )
