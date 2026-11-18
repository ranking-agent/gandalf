"""One query edge's matches, held as parallel NumPy arrays.

An edge query on a hub can match millions of edges.  As one Python tuple per
match that is several hundred bytes each, plus the time to build and later
unpack them; as arrays it is 17 bytes each, and every step that reads the
matches (binding the next edge's nodes, the joins) is a NumPy operation.
"""

from typing import Mapping, Sequence, Union

import numpy as np

#: Predicate codes -> predicate strings, as either a dict or a list.
PredicateNames = Union[Mapping[int, str], Sequence[str]]


class EdgeMatches:
    """The edges matching one QEdge, in the order they were found.

    Attributes:
        subj, obj: Each edge's subject and object node index **as stored in
            the graph** (``int32``).  Where ``via_inverse`` is set the edge
            matched through its inverse or a symmetric predicate, so the
            query's subject is the stored object and the other way round;
            :attr:`query_subj` / :attr:`query_obj` give the query's view.
        pred: Each edge's predicate, as a code into ``predicates`` (``int32``).
        via_inverse: Whether the edge was found through the inverse lookup.
        fwd_edge_idx: Each edge's forward-CSR position, unique per physical
            edge (``int32``); -1 for a match that is no real edge (a node
            standing as its own subclass).
        predicates: The predicate string of each code in ``pred``.

    Examples:
        >>> matches = EdgeMatches.from_tuples(
        ...     [(1, "biolink:treats", 2, False, 7),
        ...      (3, "biolink:treated_by", 4, True, 8)]
        ... )
        >>> len(matches)
        2
        >>> matches.query_subj.tolist(), matches.query_obj.tolist()
        ([1, 4], [2, 3])
        >>> matches.to_tuples()[1]
        (3, 'biolink:treated_by', 4, True, 8)
    """

    __slots__ = ("subj", "obj", "pred", "via_inverse", "fwd_edge_idx", "predicates")

    def __init__(
        self,
        subj: np.ndarray,
        obj: np.ndarray,
        pred: np.ndarray,
        via_inverse: np.ndarray,
        fwd_edge_idx: np.ndarray,
        predicates: PredicateNames,
    ):
        self.subj = subj
        self.obj = obj
        self.pred = pred
        self.via_inverse = via_inverse
        self.fwd_edge_idx = fwd_edge_idx
        self.predicates = predicates

    def __len__(self) -> int:
        return len(self.subj)

    @property
    def query_subj(self) -> np.ndarray:
        """Each match's node on the query edge's subject side."""
        subjects: np.ndarray = np.where(self.via_inverse, self.obj, self.subj)
        return subjects

    @property
    def query_obj(self) -> np.ndarray:
        """Each match's node on the query edge's object side."""
        objects: np.ndarray = np.where(self.via_inverse, self.subj, self.obj)
        return objects

    @classmethod
    def empty(cls, predicates: PredicateNames = ()) -> "EdgeMatches":
        """No matches."""
        return cls.concatenate([], predicates)

    @classmethod
    def concatenate(
        cls, parts: Sequence[tuple], predicates: PredicateNames
    ) -> "EdgeMatches":
        """Join ``(subj, obj, pred, via_inverse, fwd_edge_idx)`` array tuples.

        ``via_inverse`` may be a single bool standing for the whole part.
        """
        columns: list[list] = [[], [], [], [], []]
        for subj, obj, pred, via_inverse, fwd_edge_idx in parts:
            if isinstance(via_inverse, (bool, np.bool_)):
                via_inverse = np.full(len(subj), via_inverse, dtype=np.bool_)
            for column, values in zip(
                columns, (subj, obj, pred, via_inverse, fwd_edge_idx)
            ):
                column.append(values)
        dtypes = (np.int32, np.int32, np.int32, np.bool_, np.int32)
        subj, obj, pred, via_inverse, fwd_edge_idx = (
            (
                np.concatenate(column).astype(dtype, copy=False)
                if column
                else np.empty(0, dtype=dtype)
            )
            for column, dtype in zip(columns, dtypes)
        )
        return cls(subj, obj, pred, via_inverse, fwd_edge_idx, predicates)

    @classmethod
    def from_tuples(cls, tuples: Sequence[tuple]) -> "EdgeMatches":
        """Build from ``(subj, predicate, obj, via_inverse, fwd_edge_idx)`` tuples."""
        names: dict[str, int] = {}
        if not tuples:
            return cls.empty()
        subj, preds, obj, via_inverse, fwd_edge_idx = zip(*tuples)
        codes = [names.setdefault(p, len(names)) for p in preds]
        return cls(
            np.array(subj, dtype=np.int32),
            np.array(obj, dtype=np.int32),
            np.array(codes, dtype=np.int32),
            np.array(via_inverse, dtype=np.bool_),
            np.array(fwd_edge_idx, dtype=np.int32),
            predicates=list(names),
        )

    def to_tuples(self) -> list[tuple]:
        """The matches as ``(subj, predicate, obj, via_inverse, fwd_edge_idx)``."""
        predicates = self.predicates
        return [
            (subj, predicates[pred], obj, via_inverse, fwd_edge_idx)
            for subj, pred, obj, via_inverse, fwd_edge_idx in zip(
                self.subj.tolist(),
                self.pred.tolist(),
                self.obj.tolist(),
                self.via_inverse.tolist(),
                self.fwd_edge_idx.tolist(),
            )
        ]
