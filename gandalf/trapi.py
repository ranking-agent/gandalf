"""TRAPI 2.0 protocol constants, query parameters, and Response assembly.

TRAPI 2.0 moved the query-time knobs that do not change a query's meaning
(``log_level``, ``bypass_cache`` and the new ``timeout``) out of the request
body's top level and into a ``parameters`` object, which the server MUST
repeat back in its Response.  ``QueryParameters`` below is that object's
``additionalProperties: true`` nature put to use: it carries the standard TRAPI
knobs alongside gandalf's own (``subclass``, ``dehydrated``, ...).

This module owns the pieces of the protocol that are not about graph search:
reading those parameters, enforcing a client's time budget, stamping the
Response envelope, and the serialization rules below.

Nulls and empty containers
--------------------------
TRAPI 2.0 is an ``openapi: 3.1.2`` document and dropped every ``nullable:
true`` that 1.x carried.  Under OpenAPI 3.1 / JSON Schema 2020-12 a property
declared ``type: string`` does not admit null (that would need
``type: ["string", "null"]``, which the spec never uses), so **an absent value
must be an absent property, never null**.  The only exceptions are
``Attribute.value`` and ``AttributeConstraint.value``, which declare no
``type`` at all ("May be any data type"), and anything under an
``additionalProperties: true``.

Empty containers are *not* forbidden across the board -- the rule is
per-property, and there are three cases:

1. ``minItems: 1`` / ``minProperties: 1`` on an **optional** property: an
   empty value is invalid, so the property is omitted instead.  These are the
   ones worth naming in code, because they are the ones a server can get wrong
   quietly: ``Message.auxiliary_graphs``, ``Edge.qualifiers``,
   ``RetrievalSource.upstream_resource_ids``, ``Response.logs``,
   ``Response.data_release_versions``, ``Result.analyses``,
   ``Analysis.edge_bindings``, ``Analysis.support_graphs``, ``MetaEdge``'s
   list properties, and the QNode / QEdge / QPath constraint lists that come
   back in the echoed ``query_graph``.
2. ``minItems: 1`` on a **required** property (``Node.categories``,
   ``Edge.sources``, ``NodeBinding.ids``, ``EdgeBinding.ids``,
   ``AuxiliaryGraph.edges``, ``Result.node_bindings``): omitting it does not
   help, so an empty value is a data problem to fix at its source.
3. No minimum at all, where an empty container is perfectly valid:
   ``KnowledgeGraph.nodes`` / ``.edges`` (and ``nodes`` is required, so it must
   stay), ``Node.attributes``, ``Edge.attributes``, ``Analysis.attributes``.
   ``Message.results`` belongs here and goes further -- 2.0 says that when
   results are expected and none were found the property "SHOULD be an array
   with 0 Results in it", so emptying it out would be the violation.

Because of case 3 there is no blanket "strip falsy values" pass here, and
because a response can carry millions of results there is no recursive walk of
one either: each value is kept valid where it is produced.
"""

import logging
import time
from typing import Any, Optional, Union, cast

import orjson
from translator_tom import CURIE
from translator_tom.model_dicts import (
    AttributeDict,
    EdgeDict,
    KnowledgeGraphDict,
    LogEntryDict,
    MessageDict,
    NodeDict,
    QualifierDict,
    QueryDict,
    QueryParametersDict,
    ResponseDict,
    RetrievalSourceDict,
)
from typing_extensions import NotRequired, TypedDict

from gandalf.biolink import NAMED_THING
from gandalf.config import settings

logger = logging.getLogger(__name__)

#: Version of the TRAPI schema this server implements.
SCHEMA_VERSION = "2.0.0"


class QueryTimeout(Exception):
    """Raised when a query exceeds the client's ``parameters.timeout`` budget."""


class TimeoutNotSatisfiable(ValueError):
    """Raised when the client's requested timeout is below what this server offers.

    TRAPI 2.0: "If the service knows it cannot respond in the given time, it
    MAY respond with an HTTP 409 and a response explaining its time
    capabilities."
    """


def resolve_timeout(parameters: Optional[QueryParametersDict]) -> Optional[float]:
    """Return the wall-clock budget in seconds for a query, or None for unlimited.

    Reads TRAPI 2.0's ``parameters.timeout``:

    - absent: the server's own default (``settings.query_timeout``), if any
    - negative: "disable any default timeout the server implements"
    - positive: the client's budget

    Args:
        parameters: The request's ``parameters`` object, if any.

    Returns:
        The budget in seconds, or None when no timeout applies.

    Raises:
        TimeoutNotSatisfiable: if the client asks for a budget this server
            knows it cannot meet.

    Examples:
        >>> resolve_timeout({"timeout": 30})
        30.0
        >>> resolve_timeout({"timeout": -1}) is None
        True
        >>> resolve_timeout({"timeout": 0}) is None
        True
    """
    if not parameters or "timeout" not in parameters:
        default = settings.query_timeout
        return float(default) if default > 0 else None

    requested = parameters["timeout"]
    if not isinstance(requested, (int, float)) or isinstance(requested, bool):
        raise TimeoutNotSatisfiable(
            f"parameters.timeout must be a number of seconds, got {requested!r}"
        )

    requested = float(requested)
    if requested <= 0:
        # Negative disables the server default; zero is treated the same way
        # rather than as a budget no query could ever meet.
        return None

    if requested < settings.min_query_timeout:
        raise TimeoutNotSatisfiable(
            f"requested timeout of {requested}s is below the {settings.min_query_timeout}s "
            f"this server can answer within; omit parameters.timeout to accept "
            f"the server default, or send a negative value to disable it"
        )

    return requested


class Deadline:
    """A wall-clock budget for one query, checked at coarse stage boundaries.

    An instance with no budget is falsy and :meth:`check` is a no-op, so the
    unlimited path costs nothing.

    Examples:
        >>> bool(Deadline(None))
        False
        >>> Deadline(None).check("qedge")
        >>> expired = Deadline(0.0000001)
        >>> import time; time.sleep(0.001)
        >>> expired.check("qedge")
        Traceback (most recent call last):
        gandalf.trapi.QueryTimeout: query exceeded the 1e-07s timeout during qedge
    """

    __slots__ = ("budget", "_start")

    def __init__(self, budget: Optional[float]):
        self.budget = budget
        self._start = time.monotonic()

    def __bool__(self) -> bool:
        return self.budget is not None

    @property
    def elapsed(self) -> float:
        """Seconds since this deadline started."""
        return time.monotonic() - self._start

    @property
    def expired(self) -> bool:
        """Whether the budget has been spent."""
        return self.budget is not None and self.elapsed > self.budget

    def check(self, stage: str) -> None:
        """Raise :class:`QueryTimeout` if the budget has been spent.

        Args:
            stage: What the query was doing, named in the error and the log.
        """
        if self.expired:
            raise QueryTimeout(
                f"query exceeded the {self.budget}s timeout during {stage}"
            )


def data_release_versions() -> dict:
    """Return the configured source-data versions for ``Response.data_release_versions``.

    Read from the ``GANDALF_DATA_RELEASE_VERSIONS`` environment variable as a
    JSON object mapping a source name to its release version, e.g.
    ``{"translator_kg": "2026_06_21"}``.  Returns an empty dict when unset or
    unparseable, in which case the Response omits the property (TRAPI 2.0
    requires at least one entry when it is present).
    """
    raw = settings.data_release_versions
    if not raw:
        return {}
    try:
        parsed = orjson.loads(raw)
    except orjson.JSONDecodeError:
        logger.warning(
            "GANDALF_DATA_RELEASE_VERSIONS is not valid JSON; "
            "omitting data_release_versions from responses"
        )
        return {}
    if not isinstance(parsed, dict) or not parsed:
        logger.warning(
            "GANDALF_DATA_RELEASE_VERSIONS must be a non-empty JSON object "
            "mapping source name to version; omitting data_release_versions"
        )
        return {}
    return {str(name): str(version) for name, version in parsed.items()}


class GandalfParametersDict(QueryParametersDict):
    """TRAPI ``parameters`` plus gandalf's own, as a TypedDict.

    The TypedDict counterpart of :class:`gandalf.models.QueryParameters`, for
    the fast request path where the body stays a plain dict.  TRAPI 2.0 gives
    the ``parameters`` object ``additionalProperties: true`` so a server can
    carry its own settings there; typing them means a misspelled key or a
    wrong value type is a type error rather than a silently ignored setting.
    """

    subclass: NotRequired[bool]
    subclass_depth: NotRequired[int]
    dehydrated: NotRequired[bool]
    rehydrate: NotRequired[bool]
    filter_config: NotRequired[dict[str, Any]]
    annotator_config: NotRequired[dict[str, Any]]


def query_parameters(query: QueryDict) -> GandalfParametersDict:
    """Return a request's ``parameters``, or an empty object when absent.

    Args:
        query: A ``/query`` or ``/asyncquery`` body.

    Returns:
        The parameters, typed so gandalf's own settings are checked too.

    Examples:
        >>> query_parameters({"message": {}})
        {}
        >>> query_parameters({"message": {}, "parameters": {"subclass": False}})
        {'subclass': False}
    """
    return cast("GandalfParametersDict", query.get("parameters") or {})


def edge_attributes(edge: Any) -> list:
    """An edge's attributes as a list, decoding them in place if they are
    still the JSON stored in the graph, so the caller may change the list.

    A full response built for the server (``lookup(attributes_as_json=True)``)
    carries each real edge's attributes as the ``bytes`` of the JSON array
    stored in the graph rather than as a list: the server hands them to orjson
    as :class:`orjson.Fragment` just before serializing
    (:func:`attributes_to_fragments`), so they go from the graph into the
    response without ever being decoded.  ``bytes`` is never a valid TRAPI
    attribute list, so it cannot be mistaken for one.  Code that reads or
    changes an edge's attributes before that -- a response annotator plugin
    -- calls this, which decodes them in place.

    >>> edge = {"attributes": b'[{"attribute_type_id": "biolink:x"}]'}
    >>> edge_attributes(edge)
    [{'attribute_type_id': 'biolink:x'}]
    >>> edge["attributes"]
    [{'attribute_type_id': 'biolink:x'}]
    >>> edge_attributes({})  # no attributes: an empty list, not attached
    []
    """
    attributes = edge.get("attributes")
    if type(attributes) is bytes:
        attributes = edge["attributes"] = orjson.loads(attributes)
    return attributes if attributes is not None else []


def attributes_to_fragments(response: Any) -> None:
    """Turn every knowledge-graph edge's stored attributes JSON (``bytes``;
    see :func:`edge_attributes`) into an :class:`orjson.Fragment`, in place.

    The last step before serializing: orjson copies a Fragment's bytes into
    its output as they are, where going through a ``default`` hook per edge
    would cost as much as encoding the decoded lists.  A Fragment cannot be
    read back, so nothing may need an edge's attributes afterwards.
    """
    message = response.get("message") or {}
    edges = (message.get("knowledge_graph") or {}).get("edges") or {}
    fragment = orjson.Fragment
    for edge in edges.values():
        attributes = edge.get("attributes")
        if type(attributes) is bytes:
            edge["attributes"] = fragment(attributes)


class InFlightEdge(TypedDict):
    """An Edge as gandalf assembles it, which is not always a full TRAPI Edge.

    Deliberately looser than :class:`~translator_tom.model_dicts.EdgeDict` in
    one way, and wider in another:

    * ``sources`` is optional here.  TRAPI 2.0 requires it on every Edge, but
      a **dehydrated** response omits it -- along with the cold-path
      ``attributes`` -- because that mode exists to make the payload as small
      as possible, and ``sources`` is the largest thing left on an edge once
      attributes are gone (~200 bytes each).  A dehydrated response is
      therefore knowingly not schema-valid; see ``dehydrated`` in
      :func:`gandalf.search.lookup.lookup`, and the note in
      ``tests/test_trapi_conformance.py`` on why those responses are excluded
      from conformance checks.
    * ``attributes`` may be ``bytes`` -- the stored JSON, not yet decoded --
      in a response built for the server; read it through
      :func:`edge_attributes`.
    * Three bookkeeping properties hang off an edge while a response is built
      -- the knowledge-graph id it should be filed under, and the
      subject/object in *query* direction rather than the stored direction --
      and are popped again before serialization.

    Keeping the difference in a named type means the dehydrated contract is
    written down in one place, and a typo in either branch that builds an edge
    is still a type error.
    """

    # TOM annotates this ``Biolink.Predicate``, which is an alias of CURIE
    # assigned as a class attribute -- mypy will not take that as a type.
    predicate: CURIE
    subject: CURIE
    object: CURIE
    knowledge_level: str
    agent_type: str
    sources: NotRequired[list[RetrievalSourceDict]]
    attributes: NotRequired[Union[list[AttributeDict], bytes]]
    qualifiers: NotRequired[list[QualifierDict]]
    _edge_id: NotRequired[str]
    _query_subject: NotRequired[str]
    _query_object: NotRequired[str]


#: Served-``Edge`` properties that TRAPI 2.0 gives a ``minItems`` of 1 while
#: leaving optional, so an empty value has to be omitted.  ``sources`` is
#: excluded deliberately: it is required, so an empty one is a data problem
#: (see the module docstring, case 2).  ``attributes`` is excluded because 2.0
#: sets no minimum on it.
EMPTY_FORBIDDEN_EDGE_PROPERTIES = ("qualifiers",)


def prune_edge(edge: EdgeDict | InFlightEdge) -> EdgeDict | InFlightEdge:
    """Drop the properties of a served Edge that TRAPI 2.0 forbids empty.

    Called once per distinct knowledge-graph Edge rather than per result, and
    only for the handful of properties in
    :data:`EMPTY_FORBIDDEN_EDGE_PROPERTIES`, so it costs nothing measurable on
    the hot path.

    Args:
        edge: A TRAPI Edge dict (mutated in place).

    Returns:
        The same dict.

    Examples:
        >>> prune_edge({"predicate": "biolink:treats", "qualifiers": []})
        {'predicate': 'biolink:treats'}
        >>> prune_edge({"qualifiers": [{"qualifier_type_id": "biolink:x",
        ...                             "qualifier_value": "y"}]})
        {'qualifiers': [{'qualifier_type_id': 'biolink:x', 'qualifier_value': 'y'}]}
        >>> prune_edge({"attributes": []})
        {'attributes': []}
    """
    # A TypedDict cannot be subscripted with a non-literal key, so the loop
    # works through a plain-dict view of the same object.
    properties = cast("dict[str, Any]", edge)
    for prop in EMPTY_FORBIDDEN_EDGE_PROPERTIES:
        if prop in properties and not properties[prop]:
            del properties[prop]
    return edge


def drop_null_properties(obj: NodeDict | EdgeDict) -> NodeDict | EdgeDict:
    """Drop every null-valued property from a node or edge, in place.

    For objects assembled from client input -- the ``rehydrate`` path hands
    back a knowledge graph the client supplied -- where a property may arrive
    present-but-null.  TRAPI 2.0 admits no nulls, and a null also has to read
    as "absent" so that enrichment fills it from the graph instead of passing
    it through.

    Not used on the search path, which builds its own nodes and edges and
    never puts a null in one.

    Args:
        obj: A TRAPI Node or Edge dict (mutated in place).

    Returns:
        The same dict.

    Examples:
        >>> drop_null_properties({"name": None, "categories": ["biolink:Gene"]})
        {'categories': ['biolink:Gene']}
        >>> drop_null_properties({"attributes": []})
        {'attributes': []}
    """
    properties = cast("dict[str, Any]", obj)
    for key in [k for k, v in properties.items() if v is None]:
        del properties[key]
    return obj


def ensure_node_category(node: NodeDict) -> NodeDict:
    """Give a Node the Biolink root class when nothing more specific is known.

    ``Node.categories`` is required with a ``minItems`` of 1, so neither an
    empty list nor an absent property is valid and the gap has to be filled
    rather than dropped.

    Args:
        node: A TRAPI Node dict (mutated in place).

    Returns:
        The same dict.

    Examples:
        >>> ensure_node_category({"categories": []})
        {'categories': ['biolink:NamedThing']}
        >>> ensure_node_category({"categories": ["biolink:Gene"]})
        {'categories': ['biolink:Gene']}
    """
    if not node.get("categories"):
        node["categories"] = [NAMED_THING]
    return node


def prune_retrieval_sources(
    sources: list[RetrievalSourceDict],
) -> list[RetrievalSourceDict]:
    """Drop the properties of an Edge's RetrievalSources that 2.0 forbids empty.

    ``upstream_resource_ids`` has a ``minItems`` of 1, so a primary knowledge
    source -- which by definition has nothing upstream of it -- carries no
    such property rather than an empty list.  Applied at build time, before
    the source lists are interned, so serving them costs nothing.

    Args:
        sources: RetrievalSource dicts (mutated in place).

    Returns:
        The same list.

    Examples:
        >>> prune_retrieval_sources([
        ...     {"resource_id": "infores:a", "resource_role": "primary_knowledge_source",
        ...      "upstream_resource_ids": []},
        ...     {"resource_id": "infores:b", "resource_role": "aggregator_knowledge_source",
        ...      "upstream_resource_ids": ["infores:a"]},
        ... ]) == [
        ...     {"resource_id": "infores:a", "resource_role": "primary_knowledge_source"},
        ...     {"resource_id": "infores:b", "resource_role": "aggregator_knowledge_source",
        ...      "upstream_resource_ids": ["infores:a"]},
        ... ]
        True
    """
    for source in sources:
        properties = cast("dict[str, Any]", source)
        for prop in ("upstream_resource_ids", "source_record_urls"):
            if prop in properties and not properties[prop]:
                del properties[prop]
    return sources


def finalize_response(
    response: ResponseDict,
    request: Optional[QueryDict] = None,
    status: str = "Success",
    description: Optional[str] = None,
) -> ResponseDict:
    """Stamp the TRAPI 2.0 Response envelope onto a response dict, in place.

    Adds the version metadata every TRAPI Response should carry and echoes the
    request's ``parameters`` back, which TRAPI 2.0 requires of the server.
    Empty ``logs`` are dropped rather than sent as ``[]``, since 2.0 gives the
    property a ``minItems`` of 1.

    Args:
        response: The response dict, already carrying its ``message``.
        request: The originating request, read for its ``parameters``.
        status: A short status code for the outcome.
        description: A brief human-readable description of the outcome.

    Returns:
        The same dict, now complete.

    Examples:
        >>> finalize_response({"message": {}})["schema_version"]
        '2.0.0'
        >>> finalize_response({"message": {}, "logs": []}).get("logs") is None
        True
        >>> finalize_response({"message": {}}, {"parameters": {"timeout": 30}})["parameters"]
        {'timeout': 30}
    """
    response["status"] = status
    if description is not None:
        response["description"] = description

    response["schema_version"] = SCHEMA_VERSION
    response["biolink_version"] = settings.biolink_version

    releases = data_release_versions()
    if releases:
        response["data_release_versions"] = releases

    # TRAPI 2.0: "The server MUST repeat the parameters it is given in its
    # Response."
    parameters = request.get("parameters") if request else None
    if parameters:
        response["parameters"] = parameters

    if not response.get("logs"):
        response.pop("logs", None)

    # Message.auxiliary_graphs has a minProperties of 1, so a response that
    # inferred nothing carries no auxiliary_graphs rather than an empty map.
    message = response.get("message")
    if isinstance(message, dict) and not message.get("auxiliary_graphs", True):
        del message["auxiliary_graphs"]

    return response


def timeout_response(
    query: QueryDict, deadline: Deadline, logs: list[LogEntryDict]
) -> ResponseDict:
    """Build the Response for a query that ran out of time.

    TRAPI 2.0 lets a service that overruns ``parameters.timeout`` "consider the
    query failed and respond with logs indicating as such".  The response keeps
    the query graph and reports no results, so a client can tell a timeout from
    a genuine empty answer by its status and logs.

    Args:
        query: The original request dict.
        deadline: The exhausted deadline, read for its budget and elapsed time.
        logs: TRAPI LogEntry dicts collected before the timeout.

    Returns:
        A complete TRAPI Response with ``status`` of ``Timeout``.
    """
    description = (
        f"Query exceeded the requested timeout of {deadline.budget}s "
        f"after {deadline.elapsed:.1f}s."
    )
    message: MessageDict = {
        "knowledge_graph": KnowledgeGraphDict(nodes={}, edges={}),
        "results": [],
    }
    # Echo the query graph back when there is one.  Message.query_graph is a
    # typed property, so a request without one must leave it absent rather
    # than set it to null.
    incoming = query.get("message")
    query_graph = incoming.get("query_graph") if incoming else None
    if query_graph is not None:
        message["query_graph"] = query_graph

    response: ResponseDict = {"message": message, "logs": logs}
    return finalize_response(response, query, status="Timeout", description=description)
