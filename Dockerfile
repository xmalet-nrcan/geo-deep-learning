FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

# Set the working directory
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app

RUN apt update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev libexpat1  \
    curl \
    ca-certificates \
    python3-pip python3-venv \
    && apt clean && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv
# Install any needed packages specified in requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/app/geo_deep_learning/utils/calculate_min_max_from_csv.py"]
