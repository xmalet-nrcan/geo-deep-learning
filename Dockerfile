# Dockerfile
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY NRCAN-RootCA.crt /usr/local/share/ca-certificates/NRCAN-RootCA.crt
RUN update-ca-certificates

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV UID=9005 \
    USERNAME=geo_deep_learning \
    PYTHONPATH=/app

RUN useradd --uid ${UID} --create-home ${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /app /home/${USERNAME}


COPY geo_deep_learning /app/geo_deep_learning
COPY configs /app/configs
#COPY data /app/data

USER ${USERNAME}
