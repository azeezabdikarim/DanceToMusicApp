# Build stage
FROM python:3.9-slim as builder
WORKDIR /app

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install packages and dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim
WORKDIR /webapp

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy only the necessary directories and files into the working directory of the webapp
COPY ./webapp /webapp

# Your app will listen on port 5000
EXPOSE 5000

# Environment variables can be set here
ENV NAME World

# Run the command to start the Flask server
CMD ["python", "app.py"]
