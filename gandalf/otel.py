"""OpenTelemetry tracing setup for GANDALF."""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Optional

from gandalf.config import settings

logger = logging.getLogger(__name__)

# Internal handles populated by init_otel() when tracing is enabled.  Left as
# None at import so that under ``gunicorn --preload`` the master (which imports
# this module before forking) builds no SDK objects.
_inject: Optional[Callable[[dict], None]] = None
_record_baggage_impl: Optional[Callable[[], None]] = None
_provider = None
_initialized = False


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


def init_otel(app) -> None:
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
        app: The FastAPI application to instrument.
    """
    global _inject, _record_baggage_impl, _provider, _initialized

    if _initialized or not settings.otel_enabled:
        return

    from opentelemetry import baggage as _otel_baggage
    from opentelemetry import trace
    from opentelemetry.propagate import inject as _real_inject
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

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
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        excluded_urls="docs,openapi.json",
    )
    _provider = provider
    _inject = _real_inject

    _pk_regex_match = re.compile(r"\A[0-9A-Za-z-]{36}\Z").match

    def _record() -> None:
        """Attach the incoming ``pk`` baggage value to the current span."""
        span = trace.get_current_span()
        if not span.is_recording():
            return
        baggage_pk = _otel_baggage.get_baggage("pk")
        if isinstance(baggage_pk, str) and _pk_regex_match(baggage_pk):
            span.set_attribute("pk", baggage_pk)

    _record_baggage_impl = _record
    _initialized = True

    logger.info(
        "OpenTelemetry tracing initialized in PID=%d (service=%s).",
        os.getpid(),
        settings.otel_service_name,
    )
