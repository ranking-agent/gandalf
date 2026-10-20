"""The Biolink Model Toolkit gandalf builds (``gandalf.biolink.make_toolkit``).

Marked ``integration``: building a real toolkit fetches the pinned Biolink
Model release from GitHub.

    python -m pytest -m integration tests/test_biolink.py
"""

import urllib.request

import pytest

from gandalf.biolink import make_toolkit

pytestmark = pytest.mark.integration


def test_the_toolkit_fetches_nothing_once_built(monkeypatch):
    """Using the toolkit fetches nothing: its schema's imports are resolved at build.

    BMT resolves them lazily, so a query's first inverse or descendant lookup
    used to download more of the model, inside the first query each process ran.
    """
    toolkit = make_toolkit()
    opened: list[str] = []
    open_url = urllib.request.OpenerDirector.open

    def record(self, url, *args, **kwargs):
        opened.append(url if isinstance(url, str) else url.full_url)
        return open_url(self, url, *args, **kwargs)

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", record)
    # The lookups a query's predicate expansion makes first.
    toolkit.has_inverse("biolink:affects")
    assert "biolink:Gene" in toolkit.get_descendants("biolink:Gene", formatted=True)
    assert opened == []
