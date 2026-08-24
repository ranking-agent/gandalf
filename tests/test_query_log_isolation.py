"""Tests that a query's TRAPI logs contain that query's records and no others.

The server runs ``lookup`` on FastAPI's thread pool, so several queries execute
concurrently inside one worker process.  A log handler attached to the shared
``gandalf`` logger receives records from every one of them, so these tests force
two queries to overlap and assert that neither sees the other's entries.

Overlap is forced with a ``threading.Barrier`` inside a registered
``NodeFilter`` -- a real extension point invoked during traversal, so the two
queries are genuinely in flight together rather than merely appearing so.
"""

import copy
import logging
import threading

import pytest
from fastapi.testclient import TestClient

import gandalf.search.node_filters as node_filters_module
from gandalf.logging_config import configure_logging, make_query_logger
from gandalf.search import lookup
from gandalf.search.node_filters import register_node_filter
from tests.search_fixtures import graph  # noqa: F401

# Pinned to a chemical with treats-edges in the fixture graph, so traversal
# reaches the node filter where the barrier lives.
_QUERY = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Disease"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:treats"],
                },
            },
        },
    },
}

_BARRIER_KEY = "log_isolation_barrier"

# Emitted once per lookup, so a response carrying more than one of each is
# carrying another query's records.
_START = "Starting lookup."
_RETURNING = "Returning "


@pytest.fixture
def gandalf_logger():
    """The shared ``gandalf`` logger at INFO, restored afterwards.

    A query logger inherits its effective level from this one, which decides
    what reaches the query's TRAPI logs.
    """
    log = logging.getLogger("gandalf")
    prev_level, prev_handlers = log.level, list(log.handlers)
    configure_logging(logging.INFO)
    yield log
    log.setLevel(prev_level)
    log.handlers[:] = prev_handlers


@pytest.fixture
def overlap_barrier():
    """Register a NodeFilter that parks each query until both have arrived.

    Restores the process-wide filter registry on teardown.
    """
    registry = node_filters_module._REGISTRY
    saved = list(registry)
    barrier = threading.Barrier(2, timeout=30)

    def factory(cfg):
        if cfg.get(_BARRIER_KEY) is None:
            return None
        arrived = []

        def node_filter(graph, node_idx):
            if not arrived:
                arrived.append(True)
                barrier.wait()
            return True

        return node_filter

    register_node_filter(_BARRIER_KEY, factory)
    yield barrier
    registry[:] = saved


def _run_concurrently(targets):
    """Run each ``(name, callable)`` in its own thread and collect results."""
    results, errors = {}, {}

    def wrap(name, fn):
        try:
            results[name] = fn()
        except BaseException as exc:  # surface thread failures in the test
            errors[name] = exc

    threads = [
        threading.Thread(target=wrap, args=(name, fn), name=f"query-{name}")
        for name, fn in targets
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    return results


def _count(log_entries, prefix):
    return sum(1 for e in log_entries if e["message"].startswith(prefix))


def _overlapping_lookups(graph, bmt, **kwargs):
    """Run two lookups that are guaranteed to be in flight at the same time."""

    def run(name):
        return lookup(
            graph,
            copy.deepcopy(_QUERY),
            bmt=bmt,
            filter_config={_BARRIER_KEY: name},
            **kwargs,
        )

    return _run_concurrently([(n, lambda n=n: run(n)) for n in ("A", "B")])


@pytest.mark.parametrize("prefix", [_START, _RETURNING])
def test_concurrent_lookups_do_not_share_logs(
    graph, bmt, gandalf_logger, overlap_barrier, prefix
):
    """Each overlapping lookup logs its own lifecycle entries exactly once.

    A handler on the shared ``gandalf`` logger would give the query that
    started first two of each -- its own and its neighbour's.
    """
    responses = _overlapping_lookups(graph, bmt)

    for name in ("A", "B"):
        assert _count(responses[name]["logs"], prefix) == 1, responses[name]["logs"]


def test_concurrent_lookups_have_independent_log_counts(
    graph, bmt, gandalf_logger, overlap_barrier
):
    """Overlapping lookups collect the same number of entries as each other.

    Bleed inflates the earlier query's list rather than replacing it, so a
    count mismatch catches leaks that per-message checks would miss.
    """
    responses = _overlapping_lookups(graph, bmt)
    counts = {name: len(responses[name]["logs"]) for name in ("A", "B")}
    assert counts["A"] == counts["B"], counts


def test_concurrent_http_queries_do_not_share_logs(
    graph, bmt, gandalf_logger, overlap_barrier
):
    """The same isolation holds end to end through the FastAPI app.

    ``/query`` is a sync ``def`` endpoint, so Starlette dispatches it to the
    anyio worker thread pool -- the concurrency that makes this bug reachable.
    """
    import gandalf.server as server

    prev_graph, prev_bmt = server.GRAPH, server.BMT
    server.GRAPH, server.BMT = graph, bmt
    client = TestClient(server.APP)

    def run(name):
        body = copy.deepcopy(_QUERY)
        body["parameters"] = {"filter_config": {_BARRIER_KEY: name}}
        response = client.post("/query", json=body)
        response.raise_for_status()
        return response.json()

    try:
        with client:
            responses = _run_concurrently([(n, lambda n=n: run(n)) for n in ("A", "B")])
    finally:
        server.GRAPH, server.BMT = prev_graph, prev_bmt

    for name in ("A", "B"):
        assert _count(responses[name]["logs"], _START) == 1, responses[name]["logs"]
        assert _count(responses[name]["logs"], _RETURNING) == 1


def test_query_logs_still_reach_the_shared_logger(
    graph, bmt, gandalf_logger, overlap_barrier
):
    """Isolating the collector must not stop records reaching stderr.

    Query loggers propagate to ``gandalf``, so its handlers -- the process's
    stderr output -- still see every query.
    """

    class Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    capture = Capture()
    gandalf_logger.addHandler(capture)
    try:
        _overlapping_lookups(graph, bmt)
    finally:
        gandalf_logger.removeHandler(capture)

    assert sum(1 for m in capture.messages if m == _START) == 2, capture.messages


def test_log_level_is_per_query_and_leaves_shared_logger_alone(
    graph, bmt, gandalf_logger, overlap_barrier
):
    """A ``log_level`` request applies to the requesting query only.

    Saving and restoring the level on the shared logger races: with two queries
    overlapping, the second restores the level the first set, pinning the whole
    worker at DEBUG for every query that follows.
    """

    def run(name, log_level):
        return lookup(
            graph,
            copy.deepcopy(_QUERY),
            bmt=bmt,
            filter_config={_BARRIER_KEY: name},
            log_level=log_level,
        )

    responses = _run_concurrently(
        [("A", lambda: run("A", "DEBUG")), ("B", lambda: run("B", None))]
    )

    def levels(name):
        return {e["level"] for e in responses[name]["logs"]}

    assert "DEBUG" in levels("A"), "query asking for DEBUG did not get DEBUG entries"
    assert "DEBUG" not in levels("B"), "query that did not ask for DEBUG got DEBUG"
    assert gandalf_logger.level == logging.INFO


def test_query_logger_is_isolated_and_not_interned():
    """A query logger propagates for stderr but is not held by the registry.

    ``logging.getLogger`` interns by name and never evicts, so a name-per-query
    would leak a registry entry per request for the life of the process.
    """
    log = make_query_logger("testquery")

    assert log.name == "gandalf.query.testquery"
    assert log.parent is logging.getLogger("gandalf")
    assert "gandalf.query.testquery" not in logging.root.manager.loggerDict
