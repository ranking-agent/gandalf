FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy application code
COPY gandalf/ gandalf/
COPY scripts/ scripts/
COPY static/ static/
COPY gunicorn.conf.py .

EXPOSE 6429

# Graph data should be mounted as a volume at runtime
# e.g. docker run -v /path/to/graph:/data/graph -e GANDALF_GRAPH_PATH=/data/graph
ENV GANDALF_GRAPH_PATH=/data/graph
ENV GANDALF_LOG_FORMAT=json

CMD ["gunicorn", "gandalf.server:APP", "-c", "gunicorn.conf.py"]
