"""Read TRAPI 1.6 response shapes in tests (the TRAPI 1.6 branch only).

The 2.0 line reads ``binding["ids"]`` and ``edge["knowledge_level"]``; under
1.6 a binding is a list of ``{"id", "attributes"}`` objects, and
knowledge_level / agent_type are edge attributes.
"""

from typing import Any

from gandalf.trapi import edge_attributes


def binding_ids(binding: list) -> list:
    """The IDs a 1.6 NodeBinding / EdgeBinding list binds, in order."""
    return [b["id"] for b in binding]


def edge_attribute(edge: dict, attribute_type_id: str) -> Any:
    """The value of the edge attribute with this type, or KeyError."""
    for attribute in edge_attributes(edge):
        if attribute["attribute_type_id"] == attribute_type_id:
            return attribute["value"]
    raise KeyError(attribute_type_id)


def knowledge_level(edge: dict) -> Any:
    return edge_attribute(edge, "biolink:knowledge_level")


def agent_type(edge: dict) -> Any:
    return edge_attribute(edge, "biolink:agent_type")
