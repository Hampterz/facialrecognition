# Docker Setup Guide

Run the Face Recognition System easily on any PC with Docker!

## Prerequisites

### Install Docker

**Windows:**
- Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
- Install and start Docker Desktop
- Enable WSL 2 backend (recommended)

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER  # Log out and back in after this
```

**macOS:**
- Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
- Install and start Docker Desktop

### Verify Docker Installation

```bash
docker --version
docker-compose --version
```

## Quick Start

### Option 1: Using Docker Compose (Easiest)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hampterz/facialrecognition.git
   cd facialrecognition
   ```

2. **Build and run:**
   ```bash
   docker-compose up --build
   ```

3. **Run in background:**
   ```bash
   docker-compose up -d
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker Commands

1. **Build the image:**
   ```bash
   docker build -t facialrecognition:latest .
   ```

2. **Run the container:**
   ```bash
   # Linux (with X11 forwarding)
   xhost +local:docker
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     -v $(pwd)/training:/app/training \
     -v $(pwd)/output:/app/output \
     -v $(pwd)/models:/app/models \
     --device=/dev/video0:/dev/video0 \
     facialrecognition:latest
   
   # Windows (Docker Desktop)
   docker run -it --rm \
     -v %cd%\training:/app/training \
     -v %cd%\output:/app/output \
     -v %cd%\models:/app/models \
     facialrecognition:latest
   ```

## GUI Access (For Linux/Windows WSL)

### Linux with X11

The Dockerfile is configured for X11 forwarding. Make sure:

1. **Allow X11 connections:**
   ```bash
   xhost +local:docker
   ```

2. **Run with display:**
   ```bash
   docker run -it --rm \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     facialrecognition:latest
   ```

### Windows Docker Desktop

For Windows, you can:

1. **Use VNC (recommended for remote access):**
   - Install VNC server in container
   - Connect via VNC client

2. **Use X11 forwarding with WSL2:**
   - Install X server (VcXsrv or X410)
   - Configure DISPLAY variable

3. **Run headless and use web interface (if we add one):**
   - Access via browser

## Camera Access

### Linux

Camera access requires device mounting:

```bash
docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  facialrecognition:latest
```

### Windows

Camera access in Docker Desktop on Windows is limited. Options:

1. **Use USB passthrough** (if supported)
2. **Use network camera** (IP camera)
3. **Mount camera as device** (experimental)

## Volume Mounts

Data persists in mounted volumes:

- `./training` → Training images
- `./output` → Model encodings
- `./models` → Downloaded models
- `./validation` → Validation images

## Environment Variables

You can customize with environment variables:

```bash
docker run -it --rm \
  -e DISPLAY=:0 \
  -e CAMERA_INDEX=0 \
  facialrecognition:latest
```

## Troubleshooting

### GUI Not Showing

**Linux:**
```bash
# Allow X11 connections
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY

# Run with proper DISPLAY
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw facialrecognition:latest
```

**Windows:**
- Use VNC or X server
- Check Docker Desktop settings for GUI support

### Camera Not Working

**Linux:**
```bash
# List video devices
ls -l /dev/video*

# Mount specific device
docker run -it --rm --device=/dev/video0:/dev/video0 facialrecognition:latest
```

**Windows:**
- Camera access in Docker on Windows is limited
- Consider using network camera or USB passthrough

### Permission Errors

**Linux:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Build Errors

```bash
# Clean build
docker-compose build --no-cache

# Check logs
docker-compose logs
```

## Advanced Usage

### Run with Custom Python Version

Edit `Dockerfile` and change:
```dockerfile
FROM python:3.12.7-slim
```

### Add Additional Packages

Edit `Dockerfile` and add to RUN command:
```dockerfile
RUN apt-get install -y your-package
```

### Multi-stage Build (Smaller Image)

See `Dockerfile.optimized` for a smaller production image.

## Production Deployment

For production, consider:

1. **Use docker-compose.yml** for orchestration
2. **Set up volumes** for data persistence
3. **Configure networking** for remote access
4. **Add health checks** to Dockerfile
5. **Use .env file** for configuration

## Quick Reference

```bash
# Build image
docker build -t facialrecognition:latest .

# Run container
docker run -it --rm facialrecognition:latest

# Run with volumes
docker run -it --rm \
  -v $(pwd)/training:/app/training \
  -v $(pwd)/output:/app/output \
  facialrecognition:latest

# Run in background
docker run -d --name face-recognition facialrecognition:latest

# View logs
docker logs -f face-recognition

# Stop container
docker stop face-recognition

# Remove container
docker rm face-recognition

# Remove image
docker rmi facialrecognition:latest
```

## Notes

- Training data is stored in mounted volumes (not in image)
- Models are downloaded on first run (cached in volume)
- GUI requires X11 forwarding on Linux
- Camera access varies by platform

