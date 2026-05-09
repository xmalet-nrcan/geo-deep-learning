# Builder stage with CUDA 12.8.1.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libspatialindex-dev \
    libexpat1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock.cu128,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra cu128

#COPY . /app        XM : Copy too much in /app
COPY geo_deep_learning /app/geo_deep_learning
COPY pyproject.toml /app
COPY configs /app/configs
COPY uv.lock.cpu /app
COPY uv.lock.cu128 /app
#COPY data /app/data

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock.cu128,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --extra cu128


# ============================================================
# Stage 2 — Runtime (lean CUDA runtime)
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal34t64 \
    libspatialindex-c6 \
    libexpat1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# NRCan CA certificate
COPY NRCAN-RootCA.crt /usr/local/share/ca-certificates/NRCAN-RootCA.crt
RUN update-ca-certificates --fresh

ENV UID=9005 \
    USERNAME=gdl_user \
    PYTHONPATH=/app/ \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

RUN useradd --system --uid ${UID} --create-home ${USERNAME} && \
    mkdir -p ${PYTHONPATH} && \
    chown -R ${USERNAME}:${USERNAME} ${PYTHONPATH} /home/${USERNAME}

COPY --from=builder --chown=${USERNAME}:${USERNAME} /opt/python /opt/python
COPY --from=builder --chown=${USERNAME}:${USERNAME} ${PYTHONPATH} ${PYTHONPATH}

ENV PATH="${PYTHONPATH}.venv/bin:/opt/python/bin:$PATH"

USER ${USERNAME}
WORKDIR ${PYTHONPATH}



