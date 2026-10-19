"""``gandalf.__version__`` is the version the package is built with."""

import re
from pathlib import Path

import gandalf


def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    match = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert match
    assert gandalf.__version__ == match.group(1)
