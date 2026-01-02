# Face Recognition System - Docker Image
# Supports GUI and camera access

FROM python:3.12.7-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake=4.2.1* \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY *.md ./
COPY *.txt ./
COPY *.bat ./
COPY *.ps1 ./

# Create necessary directories
RUN mkdir -p training output validation models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Expose port for VNC (if using VNC for GUI)
EXPOSE 5900

# Default command
CMD ["python", "app.py"]

