# GANDALF Stress Tests

[Locust](https://locust.io/)-based load/stress tests for the GANDALF TRAPI server.

## Setup

```bash
pip install -r stress_tests/requirements-stress.txt
```

## Usage

### Web UI (interactive)

```bash
locust -f stress_tests/locustfile.py --host http://localhost:6429
```

Then open http://localhost:8089 to configure user count, spawn rate, and monitor results in real time.

### Headless (CLI)

```bash
# 50 concurrent users, spawning 5/sec, running for 60 seconds
locust -f stress_tests/locustfile.py --host http://localhost:6429 \
    --headless -u 50 -r 5 -t 60s
```

### Custom target

Point `--host` at any running GANDALF instance:

```bash
locust -f stress_tests/locustfile.py --host https://gandalf.example.org
```

## What's tested

| Endpoint | Weight | Description |
|---|---|---|
| `GET /health` | 2 | Health check |
| `GET /metadata` | 2 | Graph metadata |
| `GET /meta_knowledge_graph` | 1 | Meta knowledge graph |
| `GET /simple_spec` | 1 | Connection schema |
| `GET /node/{curie}` | 3 | Node lookup by CURIE |
| `GET /edges/{curie}` | 3 | Edge lookup with random filters |
| `GET /edge_summary/{curie}` | 2 | Edge type summary |
| `POST /query` (1-hop) | 5 | One-hop TRAPI query |
| `POST /query` (2-hop) | 3 | Two-hop TRAPI query |
| `POST /query` (qualifiers) | 2 | One-hop with qualifier constraints |
