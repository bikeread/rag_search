# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libreoffice \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports for API and Prometheus
EXPOSE 8000 9090

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"] 