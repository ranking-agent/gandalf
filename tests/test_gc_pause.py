"""The GC pause shared by lookups and the server's query handlers.

A lookup pauses the cyclic GC while it builds millions of objects, and the
server keeps it paused until the response has been serialized and freed, so
no collection ever scans the response.  The pause is counted: concurrent
queries share it, and GC comes back on only when the last one is done.
"""

import gc

import pytest

from tests.search_fixtures import graph  # noqa: F401

from gandalf.search import gc_utils, lookup
from gandalf.search.gc_utils import gc_disabled, pause_gc, resume_gc

QUERY = {
    "message": {
        "query_graph": {
            "nodes": {
                "n0": {"ids": ["CHEBI:6801"]},
                "n1": {"categories": ["biolink:Gene"]},
            },
            "edges": {
                "e0": {
                    "subject": "n0",
                    "object": "n1",
                    "predicates": ["biolink:affects"],
                }
            },
        }
    }
}


@pytest.fixture(autouse=True)
def gc_on():
    """Start each test with GC on and nobody holding the pause."""
    gc.enable()
    assert gc_utils._pause_holders == 0
    yield
    assert gc_utils._pause_holders == 0, "a pause was not released"
    gc.enable()


def test_pause_is_shared_by_overlapping_holders():
    pause_gc()  # query A
    pause_gc()  # query B, started while A runs
    resume_gc()  # A finishes first ...
    assert not gc.isenabled()  # ... and must not turn GC on under B
    resume_gc()
    assert gc.isenabled()


def test_pause_leaves_gc_off_if_it_was_off():
    gc.disable()
    with gc_disabled():
        assert not gc.isenabled()
    assert not gc.isenabled()


def test_resume_without_pause_is_an_error():
    with pytest.raises(RuntimeError):
        resume_gc()


def test_lookup_pauses_gc_and_releases_it(graph, bmt, monkeypatch):  # noqa: F811
    seen = []
    build = lookup.__globals__["_build_response"]

    def watched(*args, **kwargs):
        seen.append(gc.isenabled())
        return build(*args, **kwargs)

    monkeypatch.setitem(lookup.__globals__, "_build_response", watched)
    lookup(graph, QUERY, bmt=bmt)
    assert seen == [False]
    assert gc.isenabled()


@pytest.fixture
def server(graph, bmt, monkeypatch):  # noqa: F811
    monkeypatch.setenv("GANDALF_SKIP_PRELOAD", "true")
    monkeypatch.setenv("GANDALF_OTEL_ENABLED", "false")
    from gandalf import server as gandalf_server

    monkeypatch.setattr(gandalf_server, "GRAPH", graph)
    monkeypatch.setattr(gandalf_server, "BMT", bmt)
    return gandalf_server


def test_query_handler_holds_the_pause_until_the_response_is_serialized(
    server, monkeypatch
):
    seen = []
    trapi_response = server._trapi_response

    def watched(content):
        seen.append(gc.isenabled())
        return trapi_response(content)

    monkeypatch.setattr(server, "_trapi_response", watched)
    rendered = server.sync_lookup(request=QUERY, profile=None)
    assert seen == [False]
    assert gc.isenabled()
    assert b'"results"' in rendered.body


def test_async_handler_serializes_under_the_pause_and_posts_after(server, monkeypatch):
    posted = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, content, headers):
            posted.append((content, gc.isenabled()))

            class Response:
                status_code = 200

                def raise_for_status(self):
                    pass

            return Response()

    monkeypatch.setattr(server.httpx, "Client", Client)
    server._async_lookup("http://callback.invalid", QUERY)
    assert len(posted) == 1
    body, gc_enabled_while_posting = posted[0]
    assert b'"results"' in body
    assert gc_enabled_while_posting
