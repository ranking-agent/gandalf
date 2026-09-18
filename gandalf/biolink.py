"""Shared Biolink Model Toolkit construction.

Gandalf pins the Biolink Model version so that qualifier/predicate
classification matches the rest of Translator -- in particular the
BioPack/retriever tier 1 driver, which builds its Toolkit from a tagged
biolink-model release (``bmt.Toolkit(schema=.../v{version}/biolink-model.yaml)``).

A bare ``bmt.Toolkit()`` resolves to whatever default the installed ``bmt``
ships (e.g. biolink 4.2.2), which can differ from the version tier 1 uses and
cause the same edge to be classified differently across tiers (e.g.
``disease_context_qualifier`` is an attribute in 4.2.2 but a qualifier in 4.3.2).
"""

from bmt.toolkit import Toolkit

from gandalf.config import settings

#: Root of the Biolink class hierarchy.  TRAPI 2.0 requires Node.categories
#: and gives it a minItems of 1, so a record that names no category is served
#: as a NamedThing rather than with an (invalid) empty list.
NAMED_THING = "biolink:NamedThing"

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/biolink/biolink-model/"
    "refs/tags/v{version}/biolink-model.yaml"
)
_PREDICATE_MAP_URL = (
    "https://raw.githubusercontent.com/biolink/biolink-model/"
    "refs/tags/v{version}/predicate_mapping.yaml"
)


def make_toolkit() -> Toolkit:
    """Construct a BMT Toolkit pinned to ``settings.biolink_version``.

    If the version is empty, fall back to BMT's built-in default schema.
    """
    version = settings.biolink_version
    if not version:
        return Toolkit()
    return Toolkit(
        schema=_SCHEMA_URL.format(version=version),
        predicate_map=_PREDICATE_MAP_URL.format(version=version),
    )
