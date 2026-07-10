from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ---------------------------------------------------------------------------
    # Configuration via environment variables
    # ---------------------------------------------------------------------------

    graph_path: str = "/data/graph"
    graph_format: str = "auto"  # "auto" or "mmap"
    load_mmaps_into_memory: bool = False
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    cors_origins: str = "*"
    max_request_size_mb: int = 10
    rate_limit: int = 0

    # HTTP body compression (responses) + decompression (requests), zstandard.
    # The decompressed request size is capped at max_request_size_mb.
    compression_enabled: bool = True
    compress_response_enabled: bool = True
    decompress_request_enabled: bool = True
    compress_minimum_size: int = 500  # bytes; responses smaller than this are sent raw
    compress_zstd_level: int = 4
    server_url: str = "http://localhost:6429"
    server_maturity: str = "development"
    server_location: str = "RENCI"
    # Infores identifiers
    # infores: str = "infores:gandalf"
    infores: str = "infores:dogpark-tier0"

    # Infores credited as the primary knowledge source for edges that Gandalf
    # infers through subclass (ontology-based) reasoning -- the composite
    # edges emitted with knowledge_level=logical_entailment and a support
    # graph. These entailments come from the ontology-based inference engine
    # (OBIE), not from Gandalf's own graph, so they are attributed to
    # infores:obie with Gandalf recorded as the aggregator that returned them.
    subclass_inference_infores: str = "infores:obie"

    # Biolink Model version to pin the BMT Toolkit to. Must match the version
    # used by the tier 1 driver (BioPack/retriever) so qualifier/predicate
    # classification is identical across tiers. Empty uses BMT's built-in
    # default schema.
    biolink_version: str = "4.3.2"

    # Heartbeat (Automat cluster registration)
    automat_host: str = ""  # e.g. "http://automat:8080"; empty = disabled
    heart_rate: int = 30  # seconds between heartbeats
    service_address: str = ""  # reachable address of this Gandalf instance
    web_port: int = 8080  # port Gandalf is serving on
    plater_title: str = ""

    otel_enabled: bool = True
    otel_service_name: str = "dogpark-tier0"
    otel_use_console_exporter: bool = False
    jaeger_host: str = "http://jaeger"
    jaeger_port: int = 4317

    # Module-level graph preloading (server.py)
    skip_preload: bool = False

    # When True, enable Pydantic response_model validation on TRAPI routes
    validate_responses: bool = False

    # Gunicorn worker count
    workers: int = 2

    # Path reconstruction tunables (search/reconstruct.py)
    debug_paths_tsv: str = ""
    large_result_threshold: int = 10000000
    max_path_limit: int = 0

    # Default service URL for the literature_cooccurrence annotator plugin.
    # Empty disables the plugin unless a request supplies its own service_url.
    cooccurrence_service_url: str = ""

    # ---------------------------------------------------------------------------
    # Redis-backed async queue (durable /asyncquery + KEDA autoscaling)
    # ---------------------------------------------------------------------------
    # When enabled, /asyncquery pushes jobs onto a Redis Stream instead of
    # running them as in-process FastAPI BackgroundTasks. A separate pool of
    # ``python -m gandalf.worker`` consumers drains the stream, and KEDA's
    # redis-streams scaler scales that pool on the consumer-group lag. This
    # makes queued queries durable (they survive an API-pod restart) and gives
    # a real backlog signal to autoscale on -- no Prometheus required.
    queue_enabled: bool = False
    redis_url: str = ""  # e.g. "redis://redis:6379/0"; required when enabled
    queue_stream: str = "gandalf:asyncquery"
    queue_group: str = "gandalf-workers"
    # Consumer name within the group. Empty -> derived from hostname+pid so
    # each worker process is a distinct consumer (needed for correct pending
    # tracking and reclaim). Set explicitly only for debugging.
    queue_consumer: str = ""
    # Approximate cap on retained stream entries (XADD MAXLEN ~). Acked jobs
    # are XDEL'd immediately; this only bounds growth if jobs are never acked.
    queue_max_len: int = 100_000
    # Blocking read timeout (ms) for a single XREADGROUP call.
    queue_block_ms: int = 5_000
    # How many jobs a worker claims per read.
    queue_batch: int = 1
    # A pending (delivered-but-unacked) entry idle this long is assumed to
    # belong to a crashed worker and is reclaimed by a live one. Set safely
    # above your longest expected lookup time.
    queue_reclaim_idle_ms: int = 600_000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="gandalf_",
        extra="allow",
    )


settings = Settings()
