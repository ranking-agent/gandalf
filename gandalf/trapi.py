"""TRAPI 2.0 protocol constants, query parameters, and Response assembly.

TRAPI 2.0 moved the query-time knobs that do not change a query's meaning
(``log_level``, ``bypass_cache`` and the new ``timeout``) out of the request
body's top level and into a ``parameters`` object, which the server MUST
repeat back in its Response.  ``QueryParameters`` below is that object's
``additionalProperties: true`` nature put to use: it carries the standard TRAPI
knobs alongside gandalf's own (``subclass``, ``dehydrated``, ...).

This module owns the pieces of the protocol that are not about graph search:
reading those parameters, enforcing a client's time budget, and stamping the
Response envelope.
"""

import logging
import time
from typing import Any, Optional

import orjson

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


def resolve_timeout(parameters: Optional[dict]) -> Optional[float]:
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


def finalize_response(
    response: dict,
    request: Optional[dict] = None,
    status: str = "Success",
    description: Optional[str] = None,
) -> dict:
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
    parameters = (request or {}).get("parameters")
    if parameters:
        response["parameters"] = parameters

    if not response.get("logs"):
        response.pop("logs", None)

    return response


def timeout_response(query: dict, deadline: Deadline, logs: list) -> dict:
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
    response: dict[str, Any] = {
        "message": {
            "query_graph": query.get("message", {}).get("query_graph"),
            "knowledge_graph": {"nodes": {}, "edges": {}},
            "results": [],
        },
        "logs": logs,
    }
    return finalize_response(response, query, status="Timeout", description=description)
