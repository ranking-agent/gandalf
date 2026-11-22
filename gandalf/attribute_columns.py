"""Columns of the attribute values that traversal filters read.

A node or edge attribute constraint, or the ``min_information_content`` node
filter, would otherwise read each candidate's whole record from LMDB, decode
it, and walk its attributes list in Python.  For the attributes filters use
most, the build also writes the values out as memory-mapped arrays indexed by
node index or forward-CSR edge position, so a whole traversal step is judged
with a few vectorized array reads:

* **Numeric node attributes** (``NODE_NUMERIC_ATTRIBUTES``): one ``float64``
  per node, ``NaN`` where the node has none.
* **Edge publications** (``EDGE_PUBLICATION_ATTRIBUTES``): each edge's PubMed
  accessions as integers, laid end to end (``accessions``) with an offsets
  array per edge, like a CSR.  Every spelling of a PubMed reference that
  :mod:`gandalf.search.attribute_constraints` treats as equal (``PMID:1``,
  ``pubmed:1``, a PubMed URL, the bare ``1``) is stored as the same integer.

A column answers only what it can answer exactly as
:func:`~gandalf.search.attribute_constraints.matches_attribute_constraints`
would.  Records whose values it cannot represent faithfully (two
information_content attributes, a value that is not a plain number, a
publication accession with a leading zero, a nested list) are listed in the
column's ``irregular`` array, and constraints it does not cover (other
operators, other attributes, a non-PubMed value) are not served at all.
Either way the caller falls back to the record in LMDB, so a column only ever
changes how fast a filter runs, never what it returns.

``gandalf-build`` writes the columns, along with the node IDs and names as
memory-mapped files (:class:`gandalf.edge_id_store.NodeIdStore`,
:class:`~gandalf.edge_id_store.NodeNameStore`) for building responses.
A graph built before they existed works without them, and can be given
them without a rebuild::

    python -m gandalf.attribute_columns /data/graph_mmap
"""

import argparse
from array import array
import pickle
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import orjson

from gandalf.search.attribute_constraints import (
    _bare_accession,
    _explicit_pubmed_accession,
)

#: Node attributes stored as a numeric column.
NODE_NUMERIC_ATTRIBUTES = ("biolink:information_content",)

#: Edge attributes stored as a PubMed-accession column.
EDGE_PUBLICATION_ATTRIBUTES = ("biolink:publications",)

#: The ``metadata.pkl`` entry naming the columns a graph has.
METADATA_KEY = "attribute_columns"

_UINT32_MAX = int(np.iinfo(np.uint32).max)
# Integers beyond this lose precision as a float64.
_EXACT_FLOAT_INT = 2**53

Items = Iterable[tuple[int, list]]
#: Opens one saved ``.npy`` column file.
LoadArray = Callable[[Path], np.ndarray]
#: ``(passes, fallback)``: per record, whether the column says it passes, and
#: whether its record must be read instead (the column cannot tell).
Verdict = tuple[np.ndarray, np.ndarray]


def _map_npy(path: Path) -> np.ndarray:
    """Memory-map a ``.npy`` file as a plain ndarray view."""
    return np.asarray(np.load(path, mmap_mode="r"))


def _slug(attribute_id: str) -> str:
    """A file-name-safe form of an attribute id.

    >>> _slug("biolink:information_content")
    'biolink_information_content'
    """
    return "".join(c if c.isalnum() else "_" for c in attribute_id)


def _contains(sorted_idx: np.ndarray, idxes: np.ndarray) -> np.ndarray:
    """Whether each of *idxes* is in the sorted array *sorted_idx*."""
    if len(sorted_idx) == 0:
        return np.zeros(len(idxes), dtype=np.bool_)
    pos = np.searchsorted(sorted_idx, idxes)
    pos[pos == len(sorted_idx)] = 0
    found: np.ndarray = sorted_idx[pos] == idxes
    return found


def _constraint_values(constraint: dict) -> Optional[list]:
    """The constraint's value as a flat list of scalars, or None if nested.

    A list value means OR over its members; a list inside it would mean OR
    again, which the columns do not bother to flatten.
    """
    value = constraint.get("value")
    values = value if isinstance(value, list) else [value]
    if any(isinstance(v, (list, tuple, dict)) for v in values):
        return None
    return values


def _matching_attributes(attributes: list, attribute_id: str) -> tuple[list, bool]:
    """The attributes a constraint on *attribute_id* reads, and whether any of
    them matched only by ``original_attribute_name``."""
    matched, by_name_only = [], False
    for attr in attributes or ():
        if attr.get("attribute_type_id") == attribute_id:
            matched.append(attr)
        elif attr.get("original_attribute_name") == attribute_id:
            matched.append(attr)
            by_name_only = True
    return matched, by_name_only


class NumericColumn:
    """One number per record (see the module docstring).

    Examples:
        >>> col = NumericColumn.build(4, [
        ...     (0, [{"attribute_type_id": "x", "value": 70}]),
        ...     (1, [{"attribute_type_id": "x", "value": 30.5}]),
        ...     (2, [{"attribute_type_id": "x", "value": "high"}]),
        ... ], "x")
        >>> col.values.tolist()[:2], col.irregular.tolist()
        ([70.0, 30.5], [2])
        >>> passes, fallback = col.evaluate(
        ...     {"id": "x", "operator": ">", "value": 50}, np.array([0, 1, 2, 3]))
        >>> passes.tolist(), fallback.tolist()
        ([True, False, False, False], [False, False, True, False])
    """

    __slots__ = ("values", "irregular")

    def __init__(self, values: np.ndarray, irregular: np.ndarray):
        self.values = values
        self.irregular = irregular

    @classmethod
    def build(cls, size: int, items: Items, attribute_id: str) -> "NumericColumn":
        """Index *attribute_id* from ``(index, attributes)`` pairs.

        A record is regular when it has no such attribute, or exactly one,
        found by ``attribute_type_id``, whose value is an ``int`` or
        ``float`` (not a ``bool``) that a float64 holds exactly.
        """
        values = np.full(size, np.nan, dtype=np.float64)
        irregular = []
        for idx, attributes in items:
            matched, by_name_only = _matching_attributes(attributes, attribute_id)
            if not matched:
                continue
            value = matched[0].get("value")
            if (
                len(matched) != 1
                or by_name_only
                or type(value) not in (int, float)
                or (type(value) is int and abs(value) > _EXACT_FLOAT_INT)
            ):
                irregular.append(idx)
                continue
            values[idx] = value
        return cls(values, np.array(sorted(irregular), dtype=np.int64))

    def evaluate(self, constraint: dict, idxes: np.ndarray) -> Optional[Verdict]:
        """Judge an attribute constraint on *idxes*, or None if not served.

        Serves ``>`` and ``<`` (a list value is OR over its members), with
        ``not``, matching ``_compare_numeric``: a member that does not convert
        with ``float()`` never matches, and neither does a record without the
        attribute.
        """
        operator = constraint.get("operator")
        values = _constraint_values(constraint)
        if operator not in (">", "<") or values is None:
            return None
        bounds = []
        for v in values:
            try:
                bounds.append(float(v))
            except (TypeError, ValueError):
                continue
        column = self.values[idxes]
        passes = np.zeros(len(idxes), dtype=np.bool_)
        with np.errstate(invalid="ignore"):
            for bound in bounds:
                passes |= column > bound if operator == ">" else column < bound
        if constraint.get("not", False):
            passes = ~passes
        return passes, _contains(self.irregular, idxes)

    def at_least(self, idxes: np.ndarray, threshold) -> Verdict:
        """Whether each record's value is ``>= threshold`` (absent: False)."""
        with np.errstate(invalid="ignore"):
            passes = self.values[idxes] >= threshold
        return passes, _contains(self.irregular, idxes)

    def save(self, directory: Path, stem: str) -> None:
        np.save(directory / f"{stem}.npy", self.values)
        np.save(directory / f"{stem}_irregular.npy", self.irregular)

    @classmethod
    def load(cls, directory: Path, stem: str, load: "LoadArray") -> "NumericColumn":
        return cls(
            load(directory / f"{stem}.npy"),
            np.load(directory / f"{stem}_irregular.npy"),
        )

    def nbytes(self) -> int:
        return int(self.values.nbytes + self.irregular.nbytes)


def _accession(value) -> tuple[Optional[int], bool]:
    """A publication value's PubMed accession as stored, and whether the
    column cannot represent it.

    Mirrors ``_publications_equal`` opposite an explicit PubMed identifier:
    an explicit identifier or a bare accession on the record's side counts.
    An accession is stored as an integer, which is only faithful when its
    digits are that integer's canonical spelling.

    >>> _accession("PMID:123"), _accession("https://pubmed.ncbi.nlm.nih.gov/9/")
    ((123, False), (9, False))
    >>> _accession(" 42 "), _accession(42), _accession(True), _accession("doi:1")
    ((42, False), (42, False), (None, False), (None, False))
    >>> _accession("PMID:0123"), _accession(["PMID:1"])
    ((None, True), (None, True))
    """
    if isinstance(value, (list, tuple)):
        return None, True
    digits = _explicit_pubmed_accession(value)
    if digits is None:
        digits = _bare_accession(value)
    if digits is None:
        return None, False
    number = int(digits)
    if str(number) != digits or number > _UINT32_MAX:
        return None, True
    return number, False


def _query_accession(value) -> Optional[int]:
    """A constraint value's accession, if it is an explicit PubMed identifier
    the column can look up."""
    digits = _explicit_pubmed_accession(value)
    if digits is None:
        return None
    number = int(digits)
    if str(number) != digits or number > _UINT32_MAX:
        return None
    return number


class PublicationColumn:
    """Each record's PubMed accessions (see the module docstring).

    Examples:
        >>> col = PublicationColumn.build(3, [
        ...     (0, [{"attribute_type_id": "p", "value": ["PMID:1", "doi:x"]}]),
        ...     (1, [{"attribute_type_id": "p", "value": ["pubmed:2", "3"]}]),
        ... ], "p")
        >>> col.offsets.tolist(), col.accessions.tolist()
        ([0, 1, 3, 3], [1, 2, 3])
        >>> passes, _ = col.evaluate(
        ...     {"id": "p", "operator": "==", "value": ["PMID:3", "PMID:9"]},
        ...     np.array([0, 1, 2]))
        >>> passes.tolist()
        [False, True, False]
        >>> col.evaluate({"id": "p", "operator": "==", "value": "doi:x"},
        ...              np.array([0])) is None
        True
    """

    __slots__ = ("offsets", "accessions", "irregular")

    def __init__(self, offsets: np.ndarray, accessions: np.ndarray, irregular):
        self.offsets = offsets
        self.accessions = accessions
        self.irregular = irregular

    @classmethod
    def build(cls, size: int, items: Items, attribute_id: str) -> "PublicationColumn":
        """Index the PubMed accessions of *attribute_id* from
        ``(index, attributes)`` pairs in ascending index order (as an LMDB
        cursor yields them), streaming: only the column itself is held."""
        counts = np.zeros(size + 1, dtype=np.int64)
        found = array("I")
        irregular = []
        last = -1
        for idx, attributes in items:
            if idx <= last:
                raise ValueError("items must come in ascending index order")
            last = idx
            before, odd = len(found), False
            matched, _ = _matching_attributes(attributes, attribute_id)
            for attr in matched:
                value = attr.get("value")
                for member in value if isinstance(value, list) else [value]:
                    accession, bad = _accession(member)
                    odd |= bad
                    if accession is not None:
                        found.append(accession)
            if odd:
                irregular.append(idx)
            counts[idx + 1] = len(found) - before
        offsets = np.cumsum(counts)
        accessions = np.frombuffer(found, dtype=f"u{found.itemsize}").astype(np.uint32)
        dtype = np.uint32 if offsets[-1] <= _UINT32_MAX else np.int64
        return cls(
            offsets.astype(dtype),
            accessions,
            np.array(irregular, dtype=np.int64),
        )

    def evaluate(self, constraint: dict, idxes: np.ndarray) -> Optional[Verdict]:
        """Judge an attribute constraint on *idxes*, or None if not served.

        Serves ``==`` (with ``not``) when every value is an explicit PubMed
        identifier: a record passes when it carries any of them.
        """
        values = _constraint_values(constraint)
        if constraint.get("operator") != "==" or not values:
            return None
        wanted = [_query_accession(v) for v in values]
        if any(w is None for w in wanted):
            return None
        idxes = np.asarray(idxes, dtype=np.int64)
        starts = self.offsets[idxes].astype(np.int64)
        counts = self.offsets[idxes + 1].astype(np.int64) - starts
        # Every candidate's accessions, laid end to end
        positions = np.arange(int(counts.sum())) + np.repeat(
            starts - (np.cumsum(counts) - counts), counts
        )
        hit = np.isin(self.accessions[positions], np.array(wanted, dtype=np.uint32))
        owner = np.repeat(np.arange(len(idxes)), counts)
        passes = np.zeros(len(idxes), dtype=np.bool_)
        passes[owner[hit]] = True
        if constraint.get("not", False):
            passes = ~passes
        return passes, _contains(self.irregular, idxes)

    def save(self, directory: Path, stem: str) -> None:
        np.save(directory / f"{stem}_offsets.npy", self.offsets)
        np.save(directory / f"{stem}_accessions.npy", self.accessions)
        np.save(directory / f"{stem}_irregular.npy", self.irregular)

    @classmethod
    def load(cls, directory: Path, stem: str, load: "LoadArray") -> "PublicationColumn":
        return cls(
            load(directory / f"{stem}_offsets.npy"),
            load(directory / f"{stem}_accessions.npy"),
            np.load(directory / f"{stem}_irregular.npy"),
        )

    def nbytes(self) -> int:
        return int(self.offsets.nbytes + self.accessions.nbytes + self.irregular.nbytes)


class AttributeColumns:
    """A graph's attribute columns, by attribute id."""

    def __init__(
        self,
        node_numeric: Optional[dict] = None,
        edge_publications: Optional[dict] = None,
    ):
        self.node_numeric: dict[str, NumericColumn] = node_numeric or {}
        self.edge_publications: dict[str, PublicationColumn] = edge_publications or {}

    @classmethod
    def build(
        cls,
        num_nodes: int,
        node_items: Callable[[], Items],
        num_edges: int,
        edge_items: Optional[Callable[[], Items]],
    ) -> "AttributeColumns":
        """Build every column.

        Args:
            node_items / edge_items: Each returns a fresh iterable of
                ``(index, attributes list)``, read once per column; edges
                must come in ascending index order.  *edge_items* is None
                for a graph without stored edge attributes: it gets no edge
                columns, since every edge attribute constraint then fails
                whatever its ``not``.
        """
        return cls(
            {
                a: NumericColumn.build(num_nodes, node_items(), a)
                for a in NODE_NUMERIC_ATTRIBUTES
            },
            (
                {
                    a: PublicationColumn.build(num_edges, edge_items(), a)
                    for a in EDGE_PUBLICATION_ATTRIBUTES
                }
                if edge_items is not None
                else {}
            ),
        )

    def node_constraint(self, constraint: dict, idxes: np.ndarray) -> Optional[Verdict]:
        column = self.node_numeric.get(str(constraint.get("id")))
        return None if column is None else column.evaluate(constraint, idxes)

    def edge_constraint(self, constraint: dict, idxes: np.ndarray) -> Optional[Verdict]:
        column = self.edge_publications.get(str(constraint.get("id")))
        return None if column is None else column.evaluate(constraint, idxes)

    def save(self, directory: Path) -> dict:
        """Write every column into *directory*; returns its metadata entry."""
        entry: dict = {"node_numeric": {}, "edge_publications": {}}
        for attribute_id, numeric in self.node_numeric.items():
            stem = f"node_column_{_slug(attribute_id)}"
            numeric.save(directory, stem)
            entry["node_numeric"][attribute_id] = stem
        for attribute_id, pubs in self.edge_publications.items():
            stem = f"edge_column_{_slug(attribute_id)}"
            pubs.save(directory, stem)
            entry["edge_publications"][attribute_id] = stem
        return {METADATA_KEY: entry}

    @classmethod
    def load(
        cls, directory: Path, metadata: dict, load: Optional["LoadArray"] = None
    ) -> Optional["AttributeColumns"]:
        """Open the columns *metadata* names, or None if it names none.

        *load* opens one ``.npy`` file; by default it is memory-mapped.
        """
        if load is None:
            load = _map_npy
        entry = metadata.get(METADATA_KEY)
        if not entry:
            return None
        return cls(
            {
                a: NumericColumn.load(directory, stem, load)
                for a, stem in entry.get("node_numeric", {}).items()
            },
            {
                a: PublicationColumn.load(directory, stem, load)
                for a, stem in entry.get("edge_publications", {}).items()
            },
        )

    def nbytes(self) -> int:
        return sum(c.nbytes() for c in self.node_numeric.values()) + sum(
            c.nbytes() for c in self.edge_publications.values()
        )


def prefilter(
    constraints: Sequence[dict],
    evaluate: Optional[Callable[[dict, np.ndarray], Optional[Verdict]]],
    idxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Judge attribute *constraints* (AND) from columns where possible.

    Args:
        constraints: TRAPI AttributeConstraint dicts.
        evaluate: ``AttributeColumns.node_constraint`` or ``edge_constraint``,
            or None when the graph has no columns.
        idxes: The records to judge.

    Returns:
        ``(maybe, need_record)``: a record with *maybe* False fails for sure;
        one with *maybe* True passes unless *need_record*, in which case its
        record must be read and checked against every constraint.
    """
    n = len(idxes)
    maybe = np.ones(n, dtype=np.bool_)
    need_record = np.zeros(n, dtype=np.bool_)
    for constraint in constraints:
        verdict = None if evaluate is None else evaluate(constraint, idxes)
        if verdict is None:
            need_record[:] = True
            continue
        passes, fallback = verdict
        maybe &= passes | fallback
        need_record |= fallback
    return maybe, need_record & maybe


def _node_items(store) -> Callable[[], Items]:
    def items() -> Items:
        for idx, props in store.iter_properties():
            yield idx, props.get("attributes", [])

    return items


def edge_items_from_lmdb(lmdb_store) -> Callable[[], Items]:
    """``(edge position, attributes)`` for every edge in an edge-attribute
    LMDB store, read in key order."""
    import struct

    def items() -> Items:
        with lmdb_store._env.begin(buffers=True) as txn:
            for key, value in txn.cursor():
                yield struct.unpack(">I", bytes(key))[0], orjson.loads(value)

    return items


def node_items_from_dict(node_properties: dict) -> Callable[[], Items]:
    def items() -> Items:
        for idx, props in node_properties.items():
            yield int(idx), props.get("attributes", [])

    return items


def add_to_graph(directory: Path) -> AttributeColumns:
    """Give an already-built graph its attribute columns, without a rebuild.

    Reads every node record and every edge's attributes, then writes the
    column files, the memory-mapped node IDs and names (``NodeIdStore``,
    ``NodeNameStore``) and their ``metadata.pkl`` entries, as ``save_mmap``
    does for a new graph.  Running it again rewrites them.  Processes
    serving the graph pick them up when they next load it.
    """
    from gandalf.edge_id_store import NodeIdStore, NodeNameStore
    from gandalf.lmdb_store import LMDBPropertyStore
    from gandalf.node_store import NodeStore

    directory = Path(directory)
    with open(directory / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    num_edges = len(np.load(directory / "fwd_targets.npy", mmap_mode="r"))
    edge_path = directory / "edge_properties.lmdb"
    with NodeStore(directory / "node_store.lmdb", readonly=True) as node_store:
        edge_store = LMDBPropertyStore(edge_path) if edge_path.exists() else None
        try:
            columns = AttributeColumns.build(
                metadata["num_nodes"],
                _node_items(node_store),
                num_edges,
                edge_items_from_lmdb(edge_store) if edge_store else None,
            )
        finally:
            if edge_store is not None:
                edge_store.close()
    metadata.update(columns.save(directory))
    # The node IDs, memory-mapped for building responses
    with NodeStore(directory / "node_store.lmdb", readonly=True) as node_store:
        node_ids = NodeIdStore.from_id_map(
            node_store.iter_id_to_idx(), metadata["num_nodes"]
        )
    node_ids.save(directory)
    metadata.update(node_ids.metadata())
    # The node names, for dehydrated responses
    names: list = [None] * metadata["num_nodes"]
    with NodeStore(directory / "node_store.lmdb", readonly=True) as node_store:
        for idx, props in node_store.iter_properties():
            names[idx] = props.get("name")
    node_names = NodeNameStore.from_names(names)
    del names
    if node_names is not None:
        node_names.save(directory)
        metadata.update(node_names.metadata())
    with open(directory / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    return columns


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line for :func:`add_to_graph`."""
    parser = argparse.ArgumentParser(
        description="Add the attribute columns to an already-built graph."
    )
    parser.add_argument("graph", type=Path, help="graph directory (gandalf-build)")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    columns = add_to_graph(args.graph)
    for attribute_id, numeric in columns.node_numeric.items():
        print(
            f"{attribute_id} (nodes): {int((~np.isnan(numeric.values)).sum()):,} "
            f"values, {len(numeric.irregular):,} irregular"
        )
    for attribute_id, pubs in columns.edge_publications.items():
        print(
            f"{attribute_id} (edges): {len(pubs.accessions):,} PubMed "
            f"accessions, {len(pubs.irregular):,} irregular edges"
        )
    print(
        f"Wrote {columns.nbytes() / 2**20:.1f} MB of columns to {args.graph} "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
