from fastapi.openapi.utils import get_openapi
from pathlib import Path
import os
import yaml
from pydantic.json_schema import models_json_schema

from gandalf.config import settings
from gandalf.models import AsyncTRAPIQuery, TRAPIQuery


def _inject_request_schemas(open_api_schema):
    """Document the request bodies for /query and /asyncquery.

    Those handlers take the body as a raw dict so it is not run through
    Pydantic on the hot path (see ``server._request_dict``), which means
    FastAPI no longer emits their request schemas.  Re-attach the schemas and
    examples here so Swagger still shows the TRAPI request shape, without
    re-enabling runtime validation.
    """
    # Generate self-consistent JSON schemas for the request models, with refs
    # pointing at the OpenAPI components section, and merge their definitions
    # into components/schemas (without clobbering any FastAPI already emitted).
    _, top = models_json_schema(
        [(TRAPIQuery, "validation"), (AsyncTRAPIQuery, "validation")],
        ref_template="#/components/schemas/{model}",
    )
    defs = top.get("$defs", {})
    schemas = open_api_schema.setdefault("components", {}).setdefault("schemas", {})
    for name, sub in defs.items():
        schemas.setdefault(name, sub)

    def _request_body(model):
        examples = (model.model_config.get("json_schema_extra") or {}).get(
            "examples", []
        )
        content = {"schema": {"$ref": f"#/components/schemas/{model.__name__}"}}
        if examples:
            content["examples"] = {
                f"example_{i + 1}": {"value": ex} for i, ex in enumerate(examples)
            }
        return {"required": True, "content": {"application/json": content}}

    for path, model in (("/query", TRAPIQuery), ("/asyncquery", AsyncTRAPIQuery)):
        op = open_api_schema.get("paths", {}).get(path, {}).get("post")
        if op is not None:
            op["requestBody"] = _request_body(model)


def construct_open_api_schema(app, description=None, subpath=""):
    """
    This creates the Open api schema object

    :return:
    """
    open_api_schema = get_openapi(
        title=app.title, version=app.version, routes=app.routes
    )

    _inject_request_schemas(open_api_schema)

    open_api_extended_file_path = os.path.join(
        Path(os.path.dirname(__file__)), "openapi-config.yaml"
    )

    with open(open_api_extended_file_path) as open_api_file:
        open_api_extended_spec = yaml.load(open_api_file, Loader=yaml.SafeLoader)

    x_translator_extension = open_api_extended_spec.get("x-translator")
    x_trapi_extension = open_api_extended_spec.get("x-trapi")
    contact_config = open_api_extended_spec.get("contact")
    terms_of_service = open_api_extended_spec.get("termsOfService")
    servers_conf = open_api_extended_spec.get("servers")
    tags = open_api_extended_spec.get("tags")
    app_version = open_api_extended_spec.get("version")

    if tags:
        open_api_schema["tags"] = tags

    if x_translator_extension:
        # if x_translator_team is defined amends schema with x_translator extension
        open_api_schema["info"]["x-translator"] = x_translator_extension
        open_api_schema["info"]["x-translator"]["infores"] = settings.infores

    if x_trapi_extension:
        # if x_translator_team is defined amends schema with x_translator extension
        open_api_schema["info"]["x-trapi"] = x_trapi_extension

    if contact_config:
        open_api_schema["info"]["contact"] = contact_config

    if terms_of_service:
        open_api_schema["info"]["termsOfService"] = terms_of_service

    if description:
        open_api_schema["info"]["description"] = description
    else:
        open_api_schema["info"]["description"] = open_api_extended_spec.get(
            "description", ""
        )

    open_api_schema["info"]["title"] = app.title

    if app_version:
        open_api_schema["info"]["version"] = app_version

    # adds support to override server root path
    server_root = str(settings.server_url)
    if subpath:
        server_root += subpath

    if servers_conf:
        for s in servers_conf:
            if s["description"].startswith("Default"):
                s["url"] = server_root
                s["x-maturity"] = settings.server_maturity
                s["x-location"] = settings.server_location

        open_api_schema["servers"] = servers_conf

    return open_api_schema
