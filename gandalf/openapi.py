from fastapi.openapi.utils import get_openapi
from pathlib import Path
import os
import yaml

from gandalf.config import settings


def construct_open_api_schema(app, description=None, subpath=""):
    """
    This creates the Open api schema object

    :return:
    """
    open_api_schema = get_openapi(
        title=app.title, version=app.version, routes=app.routes
    )

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
