"""Attribute columns answer filters exactly as reading the records would.

The columns (``gandalf.attribute_columns``) are an index: a filter judged
from them must return what ``matches_attribute_constraints`` (or the
``min_information_content`` plugin) returns on the record itself.  These
tests check that on randomized awkward values, then rerun the lookup-level
constraint and filter suites on a saved graph, where the columns are used.
"""

import math
import pickle
import random

import numpy as np
import pytest

from gandalf.attribute_columns import (
    METADATA_KEY,
    AttributeColumns,
    NumericColumn,
    PublicationColumn,
    add_to_graph,
    prefilter,
)
from gandalf.graph import CSRGraph
from gandalf.loader import build_graph_from_jsonl
from gandalf.plugins.information_content import _MinInformationContent
from gandalf.search.attribute_constraints import matches_attribute_constraints
from tests import test_attribute_constraints as constraint_tests
from tests import test_node_filters as filter_tests
from tests.search_fixtures import EDGES_FILE, NODES_FILE

PUBS = "biolink:publications"
IC = "biolink:information_content"

PUBLICATION_VALUES = [
    "PMID:12",
    "pmid: 12",
    "pubmed:7",
    "PubMed:7 ",
    "https://pubmed.ncbi.nlm.nih.gov/12/",
    "http://www.ncbi.nlm.nih.gov/pubmed/7",
    "12",
    " 7 ",
    12,
    7,
    True,
    12.0,
    "PMID:012",
    "PMID:99999999999",
    "PMID:١٢",  # Arabic-Indic digits: \d matches, int() parses
    "doi:10.1/x",
    "",
    None,
    ["PMID:12"],
    {"a": 1},
]

NUMERIC_VALUES = [
    70,
    30.5,
    50,
    0,
    -1,
    True,
    "80",
    "abc",
    None,
    float("nan"),
    float("inf"),
    2**60,
    [70, 20],
    {"v": 1},
]

NUMERIC_BOUNDS = [50, "50", 49.9, -1, 0, True, "x", None, float("nan"), 2**60]


def _random_attributes(rng, attribute_id, values, list_valued):
    attributes = []
    for _ in range(rng.randrange(0, 3)):
        attr = {}
        which = rng.random()
        if which < 0.6:
            attr["attribute_type_id"] = attribute_id
        elif which < 0.8:
            attr["attribute_type_id"] = "biolink:other"
            attr["original_attribute_name"] = attribute_id
        else:
            attr["attribute_type_id"] = "biolink:other"
        if rng.random() < 0.95:
            if list_valued and rng.random() < 0.8:
                attr["value"] = rng.sample(values, rng.randrange(0, 4))
            else:
                attr["value"] = rng.choice(values)
        attributes.append(attr)
    return attributes


def _random_constraint(rng, attribute_id, operators, values):
    if rng.random() < 0.5:
        value = rng.choice(values)
    else:
        value = rng.sample(values, rng.randrange(0, 4))
    constraint = {
        "id": attribute_id if rng.random() < 0.9 else "biolink:other",
        "operator": rng.choice(operators),
        "value": value,
    }
    if rng.random() < 0.3:
        constraint["not"] = True
    return constraint


def _judge_with_columns(evaluate, constraints, records):
    """What a traversal decides: the columns, then the records they defer."""
    idxes = np.arange(len(records))
    maybe, need_record = prefilter(constraints, evaluate, idxes)
    verdict = maybe.copy()
    for i in np.flatnonzero(need_record).tolist():
        verdict[i] = bool(matches_attribute_constraints(records[i], constraints))
    return verdict.tolist()


def _expected(constraints, records):
    return [bool(matches_attribute_constraints(r, constraints)) for r in records]


def test_publication_column_agrees_with_the_records():
    rng = random.Random(1)
    records = [
        _random_attributes(rng, PUBS, PUBLICATION_VALUES, list_valued=True)
        for _ in range(600)
    ]
    column = PublicationColumn.build(len(records), enumerate(records), PUBS)
    columns = AttributeColumns(edge_publications={PUBS: column})
    served = 0
    for _ in range(400):
        constraints = [
            _random_constraint(
                rng, PUBS, ["==", "==", "===", "matches", ">"], PUBLICATION_VALUES
            )
            for _ in range(rng.randrange(1, 3))
        ]
        served += columns.edge_constraint(constraints[0], np.arange(1)) is not None
        assert _judge_with_columns(
            columns.edge_constraint, constraints, records
        ) == _expected(constraints, records), constraints
    assert served > 20, "the random constraints should exercise the column"


def test_numeric_column_agrees_with_the_records():
    rng = random.Random(2)
    records = [
        _random_attributes(rng, IC, NUMERIC_VALUES, list_valued=False)
        for _ in range(600)
    ]
    column = NumericColumn.build(len(records), enumerate(records), IC)
    columns = AttributeColumns(node_numeric={IC: column})
    served = 0
    for _ in range(400):
        constraints = [
            _random_constraint(
                rng, IC, [">", "<", ">", "<", "==", "matches"], NUMERIC_BOUNDS
            )
            for _ in range(rng.randrange(1, 3))
        ]
        served += columns.node_constraint(constraints[0], np.arange(1)) is not None
        assert _judge_with_columns(
            columns.node_constraint, constraints, records
        ) == _expected(constraints, records), constraints
    assert served > 100


class _RecordsGraph:
    """Just enough of a graph for the information content filter."""

    def __init__(self, records, columns=None):
        self.records = records
        self.attribute_columns = columns

    def get_node_property(self, node_idx, key, default=None):
        return {"attributes": self.records[node_idx]}.get(key, default)

    def get_all_node_properties_batch(self, nodes):
        return {n: {"attributes": self.records[n]} for n in nodes}


def test_information_content_filter_agrees_with_the_records():
    rng = random.Random(3)
    values = [v for v in NUMERIC_VALUES if not isinstance(v, (str, list, dict))]
    records = [
        _random_attributes(rng, IC, values, list_valued=False) for _ in range(600)
    ]
    column = NumericColumn.build(len(records), enumerate(records), IC)
    with_columns = _RecordsGraph(records, AttributeColumns(node_numeric={IC: column}))
    without = _RecordsGraph(records)
    nodes = np.arange(len(records))
    for threshold in (50, 70, 0, -5, 30.5, math.inf):
        node_filter = _MinInformationContent(threshold)
        expected = [node_filter(without, n) for n in nodes.tolist()]
        assert node_filter.mask(with_columns, nodes).tolist() == expected
        assert node_filter.mask(without, nodes).tolist() == expected


def test_irregular_records_are_deferred_to_the_record():
    records = [
        [{"attribute_type_id": PUBS, "value": ["PMID:012"]}],
        [{"attribute_type_id": PUBS, "value": [["PMID:1"]]}],
        [{"attribute_type_id": PUBS, "value": ["PMID:1"]}],
    ]
    column = PublicationColumn.build(3, enumerate(records), PUBS)
    assert column.irregular.tolist() == [0, 1]
    passes, fallback = column.evaluate(
        {"id": PUBS, "operator": "==", "value": "PMID:1"}, np.arange(3)
    )
    assert fallback.tolist() == [True, True, False]
    assert passes[2]


def test_publication_build_requires_ascending_order():
    with pytest.raises(ValueError):
        PublicationColumn.build(3, [(2, []), (1, [])], PUBS)


# ---------------------------------------------------------------------------
# A saved graph carries the columns and node IDs, and lookups use them
# ---------------------------------------------------------------------------


@pytest.fixture
def saved_dir(tmp_path):
    graph = build_graph_from_jsonl(EDGES_FILE, NODES_FILE)
    graph.save_mmap(tmp_path / "graph")
    return tmp_path / "graph"


@pytest.fixture
def mapped_graph(saved_dir):
    loaded = CSRGraph.load_mmap(saved_dir, serve_metadata=False)
    assert loaded.attribute_columns is not None
    assert loaded.node_id_store is not None
    return loaded


def test_saved_graph_has_the_columns(mapped_graph):
    columns = mapped_graph.attribute_columns
    ic = columns.node_numeric[IC].values
    assert (~np.isnan(ic)).sum() > 0
    assert len(columns.edge_publications[PUBS].accessions) > 0


def test_node_ids_come_from_the_mapped_store(mapped_graph):
    nodes = list(range(mapped_graph.num_nodes))
    from_lmdb = mapped_graph.node_store.get_node_ids_batch(nodes)
    assert mapped_graph.get_node_ids_batch(np.array(nodes[::-1])) == from_lmdb
    assert mapped_graph.get_node_ids_batch(set(nodes)) == from_lmdb
    assert [mapped_graph.get_node_id(n) for n in nodes] == [from_lmdb[n] for n in nodes]
    assert mapped_graph.get_node_id(mapped_graph.num_nodes) is None


def _close(graph):
    for store in (graph.node_store, graph.lmdb_store, graph.source_urls_store):
        if store is not None:
            store.close()


def test_columns_can_be_added_to_an_older_graph(saved_dir):
    loaded = CSRGraph.load_mmap(saved_dir, serve_metadata=False)
    fresh = loaded.attribute_columns
    _close(loaded)
    with open(saved_dir / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    del metadata[METADATA_KEY], metadata["node_ids_unique"]
    with open(saved_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    older = CSRGraph.load_mmap(saved_dir, serve_metadata=False)
    assert older.attribute_columns is None and older.node_id_store is None
    _close(older)

    add_to_graph(saved_dir)
    added = CSRGraph.load_mmap(saved_dir, serve_metadata=False)
    np.testing.assert_array_equal(
        added.attribute_columns.node_numeric[IC].values,
        fresh.node_numeric[IC].values,
    )
    for name in ("offsets", "accessions", "irregular"):
        np.testing.assert_array_equal(
            getattr(added.attribute_columns.edge_publications[PUBS], name),
            getattr(fresh.edge_publications[PUBS], name),
        )
    assert added.node_id_store is not None


# The lookup-level suites, rerun on a saved graph so the columns answer.


class TestNodeConstraintsOnColumns(constraint_tests.TestNodeConstraintsIntegration):
    @pytest.fixture
    def graph(self, mapped_graph):
        return mapped_graph


class TestEdgeConstraintsOnColumns(
    constraint_tests.TestEdgeAttributeConstraintsIntegration
):
    @pytest.fixture
    def graph(self, mapped_graph):
        return mapped_graph


class TestPubMedEdgeFilteringOnColumns(constraint_tests.TestPubMedEdgeFiltering):
    @pytest.fixture
    def graph(self, mapped_graph):
        return mapped_graph


class TestMinInformationContentOnColumns(filter_tests.TestMinInformationContent):
    @pytest.fixture
    def graph(self, mapped_graph):
        return mapped_graph


class TestCombinedFiltersOnColumns(filter_tests.TestCombinedFilters):
    @pytest.fixture
    def graph(self, mapped_graph):
        return mapped_graph
