# Use RENCI python base image
FROM ghcr.io/translatorsri/renci-python-image:3.11.5

ENV PYSTOW_HOME=/tmp/pystow

WORKDIR /app

# make sure all is writeable for the nru USER later on
RUN chmod -R 777 .

# Install dependencies first for better layer caching
COPY requirements.txt pyproject.toml setup.py README.md ./
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .[server]

# switch to the non-root user (nru). defined in the base image
USER nru

# Copy application code
COPY gandalf/ gandalf/
# COPY scripts/ scripts/
COPY gunicorn.conf.py .

EXPOSE 6429

# Graph data should be mounted as a volume at runtime
# e.g. docker run -v /path/to/graph:/data/graph -e GANDALF_GRAPH_PATH=/data/graph
ENV GANDALF_GRAPH_PATH=/data/graph
ENV GANDALF_LOG_FORMAT=json

# CMD ls /data/graph
CMD ["gunicorn", "gandalf.server:APP", "-c", "gunicorn.conf.py"]
