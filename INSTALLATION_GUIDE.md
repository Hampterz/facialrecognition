# Complete Installation Guide with Exact Versions

## System Requirements

### Python
- **Version:** Python 3.12.7 (Recommended)
- **Download:** https://www.python.org/downloads/
- **Verify:** `python --version` should show `Python 3.12.7`

### CMake
- **Version:** CMake 4.2.1 or later
- **Windows:** Download from https://cmake.org/download/ or `choco install cmake`
- **Linux:** `sudo apt-get install cmake` (Ubuntu/Debian)
- **macOS:** `brew install cmake`
- **Verify:** `cmake --version`

### C Compiler
- **Windows:** MinGW or Visual Studio Build Tools
- **Linux:** gcc (usually pre-installed)
- **macOS:** Xcode Command Line Tools (`xcode-select --install`)

## Step-by-Step Installation

### 1. Install Python 3.12.7

**Windows:**
```powershell
# Download from python.org or use Anaconda
conda create -n face_recognition python=3.12.7
conda activate face_recognition
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-pip
```

**macOS:**
```bash
brew install python@3.12
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install CMake (if not installed)

**Windows:**
```powershell
choco install cmake
# Or download installer from cmake.org
```

**Linux:**
```bash
sudo apt-get install cmake
```

**macOS:**
```bash
brew install cmake
```

### 5. Install Python Packages

```bash
pip install -r requirements.txt
```

This installs:
- ultralytics==8.3.245
- huggingface-hub==0.36.0
- supervision==0.27.0
- face-recognition==1.3.0
- retina-face==0.0.17
- numpy==1.26.4
- Pillow==10.3.0
- opencv-python==4.11.0.86
- torch==2.9.1
- torchvision==0.24.1

### 6. Verify Installation

```bash
python verify_setup.py
```

Or manually check:
```bash
python -c "import ultralytics; import face_recognition; import cv2; print('✓ All imports successful!')"
```

## Troubleshooting

### Issue: "CMake not found"
**Solution:** Install CMake 4.2.1+ and add to PATH

### Issue: "Failed to build wheel"
**Solution:** 
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue: "torch installation fails"
**Solution:**
```bash
# For CPU only
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

# For CUDA (if you have GPU)
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "face-recognition installation fails"
**Solution:**
```bash
# Install dlib first (if needed)
pip install cmake
pip install dlib
pip install face-recognition==1.3.0
```

### Issue: Version conflicts
**Solution:** Use virtual environment and install exact versions:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Test

After installation, test the app:
```bash
python app.py
```

If the GUI opens without errors, installation is successful!

