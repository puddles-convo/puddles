FROM python:3.14-slim AS core
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PUDDLES_CACHE=/cache
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY schemas.py /app/schemas.py
COPY puddles /app/puddles
COPY safety /app/safety
RUN mkdir /cache /runs && chown 10001:10001 /cache /runs
USER 10001:10001
ENTRYPOINT ["python", "-m"]
CMD ["puddles.eval.external", "--help"]

FROM ghcr.io/astral-sh/uv:0.10.4 AS uv
FROM core AS full
USER root
ARG VERA_REVISION=2ea4823cade663da3143e15dd7f63641c175502d
COPY --from=uv /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir /opt/vera \
    && curl -fsSL "https://codeload.github.com/SpringCare/VERA-MH/tar.gz/${VERA_REVISION}" -o /tmp/vera.tgz \
    && tar -xzf /tmp/vera.tgz --strip-components=1 -C /opt/vera \
    && rm /tmp/vera.tgz \
    && cd /opt/vera && uv sync --frozen --no-dev \
    && rm -rf /root/.cache/uv
ENV PUDDLES_VERA_ROOT=/opt/vera MPLCONFIGDIR=/tmp/matplotlib
USER 10001:10001
