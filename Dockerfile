# Use the Python 3.9 slim image as the base image
FROM python:3.9-slim

# Set the working directory to the root of your project structure
WORKDIR /DanceToMusicApp

# Copy the requirements.txt file to the container
COPY requirements_docker.txt .

# Install necessary system libraries for OpenCV and Python dependencies in one RUN command
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && pip install --no-cache-dir -r requirements_docker.txt \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*


# Copy only the webapp directory
COPY webapp /DanceToMusicApp/webapp

# Set the working directory to the webapp directory where app.py is located
WORKDIR /DanceToMusicApp/webapp

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables (if needed)
ENV NAME World

# Use gunicorn to run the Flask app (assuming gunicorn is listed in your requirements.txt)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]