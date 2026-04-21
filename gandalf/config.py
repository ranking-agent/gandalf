from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------------------------
    # Configuration via environment variables
    # ---------------------------------------------------------------------------

    graph_path: str = "/data/graph"
    graph_format: Literal["csr", "qlever"] = "csr"
    load_mmaps_into_memory: bool = False
    qlever_host: str = "localhost"
    qlever_port: int = 8888
    qlever_access_token: str = ""
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    cors_origins: str = "*"
    max_request_size_mb: int = 10
    rate_limit: int = 0
    server_url: str = "http://localhost:6429"
    server_maturity: str = "development"
    server_location: str = "RENCI"
    # Infores identifiers
    infores: str = "infores:gandalf"

    # Heartbeat (Automat cluster registration)
    automat_host: str = ""  # e.g. "http://automat:8080"; empty = disabled
    heart_rate: int = 30  # seconds between heartbeats
    service_address: str = ""  # reachable address of this Gandalf instance
    web_port: int = 8080  # port Gandalf is serving on
    plater_title: str = ""

    otel_enabled: bool = True
    otel_service_name: str = "gandalf"
    otel_use_console_exporter: bool = False
    jaeger_host: str = "http://jaeger"
    jaeger_port: int = 4317

    # Module-level graph preloading (server.py)
    skip_preload: bool = False

    # Gunicorn worker count
    workers: int = 2

    # Path reconstruction tunables (search/reconstruct.py)
    debug_paths_tsv: str = ""
    large_result_threshold: int = 50000
    max_path_limit: int = 0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="gandalf_",
        extra="allow",
    )


settings = Settings()
