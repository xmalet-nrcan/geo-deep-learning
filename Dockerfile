FROM python:3

# Set the working directory
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "/app/geo_deep_learning/utils/calculate_min_max_from_csv.py"]