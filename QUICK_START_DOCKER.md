# Quick Start: Run on Another PC with Docker

## Step 1: Install Docker

### Windows:
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. Install Docker Desktop
3. Start Docker Desktop (wait for it to fully start - whale icon in system tray)
4. Verify: Open PowerShell and run `docker --version`

### Linux:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
# Log out and back in after this command
```

### macOS:
1. Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/)
2. Install Docker Desktop
3. Start Docker Desktop
4. Verify: Open Terminal and run `docker --version`

## Step 2: Get the Code

### Option A: Clone from GitHub (Recommended)
```bash
git clone https://github.com/Hampterz/facialrecognition.git
cd facialrecognition
```

### Option B: Download ZIP
1. Go to https://github.com/Hampterz/facialrecognition
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal/PowerShell in the extracted folder

## Step 3: Run with Docker

### Easiest Method: Docker Compose

**Windows (PowerShell):**
```powershell
docker-compose up --build
```

**Linux/Mac (Terminal):**
```bash
docker-compose up --build
```

This will:
- Build the Docker image (first time only, takes a few minutes)
- Download all dependencies
- Start the application
- Show logs in the terminal

### Alternative: Quick Run Script

**Windows:**
```powershell
.\docker-run.bat
```

**Linux/Mac:**
```bash
bash docker-run.sh
```

### Alternative: Manual Docker Commands

**Windows (PowerShell):**
```powershell
# Build the image (first time only)
docker build -t facialrecognition:latest .

# Run the container
docker run -it --rm `
  -v ${PWD}\training:/app/training `
  -v ${PWD}\output:/app/output `
  -v ${PWD}\models:/app/models `
  facialrecognition:latest
```

**Linux/Mac:**
```bash
# Build the image (first time only)
docker build -t facialrecognition:latest .

# Run the container
docker run -it --rm \
  -v $(pwd)/training:/app/training \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  facialrecognition:latest
```

## Step 4: Use the Application

Once the container starts:
- The GUI should open automatically (if X11/VNC is configured)
- Or access via terminal if running headless
- All your training data will be saved in the `training/` folder
- Models will be cached in the `models/` folder

## Running in Background

To run in background (detached mode):

```bash
docker-compose up -d
```

View logs:
```bash
docker-compose logs -f
```

Stop:
```bash
docker-compose down
```

## Troubleshooting

### Docker not starting?
- **Windows:** Make sure WSL 2 is enabled in Docker Desktop settings
- **Linux:** Make sure Docker service is running: `sudo systemctl status docker`
- **All:** Restart Docker Desktop/Service

### Build fails?
```bash
# Clean build (removes cache)
docker-compose build --no-cache
```

### GUI not showing?
- **Windows:** GUI apps in Docker on Windows need X server or VNC
- **Linux:** Make sure X11 forwarding is enabled: `xhost +local:docker`
- **All:** You can still use the app via terminal/CLI

### Camera not working?
- **Windows:** Camera access in Docker is limited, may need USB passthrough
- **Linux:** Make sure camera device is accessible: `ls -l /dev/video0`
- **All:** You can still test with images/videos

### Permission errors?
```bash
# Linux: Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

## What Gets Saved?

All your data is saved in local folders:
- `training/` - Your training images
- `output/` - Trained model encodings
- `models/` - Downloaded AI models
- `validation/` - Validation images

These folders persist even when you stop/remove the container!

## Quick Commands Reference

```bash
# Start application
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop application
docker-compose down

# Rebuild (if you change code)
docker-compose up --build

# Remove everything (keeps your data)
docker-compose down -v
```

## That's It!

Once Docker is installed and you run `docker-compose up`, everything else is automatic:
- ✅ Python 3.12.7 installed
- ✅ All packages with exact versions
- ✅ CMake and dependencies
- ✅ Everything configured

No manual installation needed! 🎉

