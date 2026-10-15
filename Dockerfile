# Dockerfile
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime as builder

WORKDIR /app

COPY NRCAN-RootCA.crt /usr/local/share/ca-certificates/NRCAN-RootCA.crt
RUN update-ca-certificates --fresh

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

ENV UID=9005 \
    USERNAME=geo_deep_learning \
    PYTHONPATH=/app

RUN useradd --uid ${UID} --create-home ${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /app /home/${USERNAME}


COPY geo_deep_learning /app/geo_deep_learning
COPY configs /app/configs
#COPY data /app/data
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

USER ${USERNAME}

