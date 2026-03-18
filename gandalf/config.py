from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------------------------------------------------------------------------
    # Configuration via environment variables
    # ---------------------------------------------------------------------------

    graph_path: str = "/data/graph"
    graph_format: str = "auto"  # "auto" or "mmap"
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    cors_origins: str = "*"
    max_request_size_mb: int = 10
    rate_limit: int = 0
    server_url: str = "http://localhost:6429"
    server_maturity: str = "development"
    server_location: str = "RENCI"

    otel_enabled: bool = True
    otel_service_name: str = "gandalf"
    otel_use_console_exporter: bool = False
    jaeger_host: str = "http://jaeger"
    jaeger_port: int = 4317

    class Config:
        env_file = ".env"


settings = Settings()
