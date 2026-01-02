#!/bin/bash
# Quick Docker run script for Face Recognition System

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Face Recognition System - Docker Runner${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if image exists
if ! docker images | grep -q facialrecognition; then
    echo "Building Docker image..."
    docker build -t facialrecognition:latest .
fi

# Create directories if they don't exist
mkdir -p training output validation models

# Allow X11 connections (Linux)
if [ "$(uname)" != "Darwin" ] && [ "$(uname)" != "MINGW"* ]; then
    xhost +local:docker 2>/dev/null
    export DISPLAY=${DISPLAY:-:0}
fi

# Run container
echo "Starting container..."
docker run -it --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$(pwd)/training:/app/training" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/validation:/app/validation" \
    -v "$(pwd)/models:/app/models" \
    --device=/dev/video0:/dev/video0 2>/dev/null \
    facialrecognition:latest

echo -e "${GREEN}Container stopped.${NC}"

