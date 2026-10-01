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
    biolink_version: str = "4.4.2"

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

    # TRAPI query-time budget (parameters.timeout, gandalf/trapi.py).
    # query_timeout is the server's own default in seconds; 0 disables it, and
    # a client can disable it explicitly with a negative parameters.timeout.
    # A client asking for less than min_query_timeout gets an HTTP 409, since
    # the server knows up front it cannot answer that fast.
    query_timeout: float = 0.0
    min_query_timeout: float = 1.0

    # Source-data versions reported as Response.data_release_versions, as a
    # JSON object mapping source name to release version, e.g.
    # '{"translator_kg": "2026_06_21"}'.  Empty omits the property.
    data_release_versions: str = ""

    # Path reconstruction tunables (search/reconstruct.py)
    debug_paths_tsv: str = ""
    large_result_threshold: int = 10_000_000
    max_path_limit: int = 0

    # Default service URL for the literature_cooccurrence annotator plugin.
    # Empty disables the plugin unless a request supplies its own service_url.
    cooccurrence_service_url: str = ""

    # Queue mode (gandalf/jobs.py, gandalf/worker.py).  With queue_url set,
    # the API enqueues each /query and /asyncquery as a job on a Redis Stream
    # and separate worker processes execute them; empty runs queries in the
    # API process as before.
    queue_url: str = ""  # e.g. "redis://redis:6379/0"
    queue_stream: str = "gandalf:jobs"
    queue_group: str = "gandalf-workers"
    queue_dead_stream: str = "gandalf:jobs:dead"
    # A job delivered more than this many times (a worker died running it)
    # is dead-lettered instead of run again.
    queue_max_deliveries: int = 2
    # A running job idle in the consumer group for longer than this is taken
    # over by another worker; workers refresh their job every
    # queue_keepalive_seconds while it runs, so only a dead worker's job
    # goes idle.
    queue_claim_idle_seconds: float = 120.0
    queue_keepalive_seconds: float = 30.0
    # How long a worker blocks waiting for a job before checking for
    # shutdown and refreshing its liveness heartbeat.
    queue_block_seconds: float = 5.0
    # How long /query waits for a worker's answer: the client's
    # parameters.timeout plus queue_sync_grace_seconds (for the worker's
    # own Timeout response to arrive), or queue_sync_max_wait_seconds when
    # the query has no timeout.
    queue_sync_max_wait_seconds: float = 1800.0
    queue_sync_grace_seconds: float = 30.0
    # Results of /query jobs are stored zstd-compressed in Redis, in chunks
    # (Redis caps a value at 512 MB), until the API collects them.
    result_ttl_seconds: int = 900
    result_chunk_bytes: int = 64 * 1024 * 1024

    # How many finished jobs the status page's history keeps (a capped
    # Redis Stream, gandalf:jobs:history).
    history_maxlen: int = 2000

    # Worker process (python -m gandalf.worker)
    worker_name: str = ""  # consumer name; default "<hostname>-<pid>"
    # Exit after this many jobs so Kubernetes restarts the process and
    # returns the memory glibc holds after large queries (0 = never).
    worker_max_jobs: int = 500
    # Prometheus /metrics port for the worker (0 = disabled).
    worker_metrics_port: int = 9100
    # Touched on every loop iteration; a liveness probe checks its age.
    worker_heartbeat_file: str = "/tmp/gandalf-worker-heartbeat"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="gandalf_",
        extra="allow",
    )


settings = Settings()
