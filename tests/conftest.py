"""Pytest fixtures shared across all test modules."""

import pytest

import gandalf.validation as _validation_module


class _MockElement:
    """Mock Biolink Model element returned by get_element()."""

    def __init__(self, canonical=False):
        self.annotations = {"canonical_predicate": True} if canonical else {}


class _MockView:
    """Mock for bmt.view providing enum access."""

    _ENUMS = {
        "GeneOrGeneProductOrChemicalEntityAspectEnum": None,
        "DirectionQualifierEnum": None,
    }

    def all_enums(self):
        return dict(self._ENUMS)


class MockBMT:
    """Mock Biolink Model Toolkit for testing without network access.

    Provides hardcoded predicate metadata matching the real Biolink Model
    for the predicates used in the test fixtures.  This avoids the external
    API call that ``bmt.Toolkit()`` makes on initialisation, which can hang
    in network-restricted environments.
    """

    _SYMMETRIC = frozenset(
        {
            "biolink:interacts_with",
            "biolink:correlated_with",
            "biolink:related_to",
        }
    )

    _INVERSES = {
        "biolink:treats": "biolink:treated_by",
        "biolink:treated_by": "biolink:treats",
        "biolink:affects": "biolink:affected_by",
        "biolink:affected_by": "biolink:affects",
        "biolink:gene_associated_with_condition": "biolink:condition_associated_with_gene",
        "biolink:condition_associated_with_gene": "biolink:gene_associated_with_condition",
        "biolink:has_phenotype": "biolink:phenotype_of",
        "biolink:phenotype_of": "biolink:has_phenotype",
        "biolink:participates_in": "biolink:has_participant",
        "biolink:has_participant": "biolink:participates_in",
        "biolink:subclass_of": "biolink:superclass_of",
        "biolink:superclass_of": "biolink:subclass_of",
        "biolink:ameliorates_condition": "biolink:ameliorated_by",
        "biolink:ameliorated_by": "biolink:ameliorates_condition",
        "biolink:causes": "biolink:caused_by",
        "biolink:caused_by": "biolink:causes",
        "biolink:preventative_for_condition": "biolink:prevented_by",
        "biolink:prevented_by": "biolink:preventative_for_condition",
    }

    _CANONICAL = frozenset(
        {
            "biolink:treats",
            "biolink:affects",
            "biolink:gene_associated_with_condition",
            "biolink:has_phenotype",
            "biolink:interacts_with",
            "biolink:participates_in",
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
            "biolink:causes",
            "biolink:subclass_of",
            "biolink:related_to",
        }
    )

    # Hierarchy: parent -> list of children.
    # Covers both predicates and categories used in tests.
    _DESCENDANTS = {
        "biolink:treats": [
            "biolink:ameliorates_condition",
            "biolink:preventative_for_condition",
        ],
        "biolink:Drug": ["biolink:SmallMolecule"],
        "biolink:GeneOrGeneProduct": ["biolink:Gene"],
        "biolink:ChemicalEntity": ["biolink:SmallMolecule", "biolink:Drug"],
    }

    # Qualifier enum values used for qualifier expansion.
    _ENUM_VALUES = {
        "GeneOrGeneProductOrChemicalEntityAspectEnum": {
            "activity",
            "abundance",
            "expression",
            "synthesis",
            "degradation",
            "cleavage",
            "hydrolysis",
            "metabolic_processing",
            "mutation_rate",
            "stability",
            "transport",
            "secretion",
            "uptake",
            "splicing",
            "localization",
            "folding",
        },
        "DirectionQualifierEnum": {
            "increased",
            "decreased",
            "upregulated",
            "downregulated",
        },
    }

    def __init__(self):
        self.view = _MockView()

    def is_symmetric(self, predicate: str) -> bool:
        return predicate in self._SYMMETRIC

    def has_inverse(self, predicate: str) -> bool:
        return predicate in self._INVERSES

    def get_inverse_predicate(
        self, predicate: str, formatted: bool = False
    ) -> str | None:
        return self._INVERSES.get(predicate)

    def get_element(self, predicate: str):
        if predicate in self._CANONICAL:
            return _MockElement(canonical=True)
        if predicate in self._INVERSES or predicate in self._SYMMETRIC:
            return _MockElement(canonical=False)
        return None

    def get_descendants(self, predicate: str, formatted: bool = False) -> list[str]:
        return list(self._DESCENDANTS.get(predicate, []))

    def is_permissible_value_of_enum(self, enum_name: str, value: str) -> bool:
        return value in self._ENUM_VALUES.get(enum_name, set())

    def get_permissible_value_descendants(
        self, permissible_value: str, enum_name: str
    ) -> list[str]:
        # None of the qualifier values used in tests have children.
        return []


@pytest.fixture(scope="session")
def bmt():
    """Create a mock BMT instance shared across all tests.

    Uses MockBMT to avoid the external API call that bmt.Toolkit()
    makes on initialisation, preventing test hangs in network-restricted
    environments.
    """
    return MockBMT()


@pytest.fixture(scope="session", autouse=True)
def _patch_validation_bmt(bmt):
    """Patch the validation module's global BMT singleton with our mock.

    gandalf.validation maintains its own module-level _bmt that is lazily
    initialised via _get_bmt().  Setting it here ensures the validation
    helpers never attempt to create a real Toolkit().
    """
    _validation_module._bmt = bmt
