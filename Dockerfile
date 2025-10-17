FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

RUN apt update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev libexpat1  \
    curl \
    ca-certificates \
    python3-pip python3-venv \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

COPY NRCAN-RootCA.crt /usr/local/share/ca-certificates/NRCAN-RootCA.crt
RUN update-ca-certificates

# Set the working directory
WORKDIR /app

COPY ./requirements.txt .
RUN /opt/venv/bin/pip install poetry
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

ENTRYPOINT ["python", "/app/geo_deep_learning/utils/calculate_min_max_from_csv.py"]
