# Face Recognition System

A Python-based face recognition system with a beautiful GUI that can recognize faces from static images and live camera feed.

## Features

- 🎨 **Beautiful GUI Interface** - Easy-to-use graphical interface
- 🎓 **Train Model** - Add people, upload photos, and train your model
- 📹 **Live Camera Recognition** - Real-time face recognition from USB camera
- 🖼️ **Test Images** - Test recognition on single images
- 👥 **Manage People** - View and delete registered people
- ⚙️ **Settings** - Configure camera and model settings

## Prerequisites

### Required System Components

**Python Version:**
- **Python 3.12.7** (Recommended - Tested and working)
- Python 3.9, 3.10, or 3.11 also supported

**CMake:**
- **CMake 4.2.1** or later
- Required for building some dependencies

**C Compiler:**
- **Windows:** MinGW or Visual Studio Build Tools
- **Linux:** gcc (usually pre-installed)
- **macOS:** Xcode Command Line Tools

### Installing System Prerequisites

**Windows:**
1. **Python 3.12.7:**
   - Download from [python.org](https://www.python.org/downloads/)
   - Or use Anaconda: `conda install python=3.12.7`

2. **CMake 4.2.1:**
   - Download from [CMake downloads](https://cmake.org/download/)
   - Or via Chocolatey: `choco install cmake --version=4.2.1`

3. **C Compiler:**
   - Install MinGW: `choco install mingw`
   - Or install Visual Studio Build Tools (includes C++ compiler)

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3-pip cmake gcc g++
```

**macOS:**
```bash
brew install python@3.12 cmake gcc
```

## Installation

### Option 1: Docker (Easiest - Recommended for Other PCs)

**Quick Start with Docker:**
```bash
# Clone repository
git clone https://github.com/Hampterz/facialrecognition.git
cd facialrecognition

# Build and run with Docker Compose
docker-compose up --build

# Or use the quick run script
# Linux/Mac:
bash docker-run.sh

# Windows:
docker-run.bat
```

**See [DOCKER_SETUP.md](DOCKER_SETUP.md) for detailed Docker instructions.**

### Option 2: Manual Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3.12 -m venv venv
source venv/bin/activate
```

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following **exact versions** (tested and working):
- **ultralytics==8.3.245** - YOLO models for face detection
- **huggingface-hub==0.36.0** - Model downloading
- **supervision==0.27.0** - Detection utilities
- **face-recognition==1.3.0** - Face encoding and matching
- **retina-face==0.0.17** - RetinaFace detector (optional)
- **numpy==1.26.4** - Numerical operations
- **Pillow==10.3.0** - Image processing
- **opencv-python==4.11.0.86** - Camera and video support
- **torch==2.9.1** - PyTorch framework
- **torchvision==0.24.1** - Vision utilities

### What's New: Multi-Model Face Detection

We support **three face detection models**:
- ✅ **YOLOv11** (Default) - Latest YOLO, best accuracy
- ✅ **YOLOv8** - Stable YOLO version
- ✅ **RetinaFace** - Deep learning with landmarks (optional)

**No more dlib compatibility issues!** Works with Python 3.12.

Models will be automatically downloaded on first use (~6-120MB each).

### Verify Installation

Run the verification script:
```bash
python verify_setup.py
```

This will check if all required packages are installed correctly.

## Project Structure

```
face_recognizer/
│
├── training/          # Training images (organized by person)
│   └── person_name/
│       ├── img1.jpg
│       └── img2.jpg
│
├── validation/        # Validation images
│   ├── person1.jpg
│   └── person2.jpg
│
├── output/            # Generated encodings
│   └── encodings.pkl
│
├── app.py             # Main GUI application (START HERE!)
├── detector.py        # Training and image recognition (CLI)
├── live_camera.py     # Live camera recognition (CLI)
├── requirements.txt
└── README.md
```

## Usage

### Quick Start with GUI (Recommended)

1. **Launch the Application:**
   
   **Option 1:** Double-click `run.bat` (Windows)
   
   **Option 2:** Run from command line:
   ```bash
   python app.py
   ```

2. **Train Your Model:**
   - Click "🎓 Train Model" on the homepage
   - Enter a person's name in the "Person Name" field
   - Click "📷 Add Photos" and select multiple photos of that person (at least 3-5 photos recommended)
   - Repeat for each person you want to recognize
   - Select model type (HOG for CPU/faster, CNN for GPU/more accurate)
   - Click "🚀 Train Model" to train
   - Wait for the training to complete

3. **Start Live Recognition:**
   - Click "📹 Live Camera Recognition" on the homepage
   - The camera window will open
   - Walk in front of the camera - your name will appear when recognized
   - Click "Stop" to close the camera

4. **Other Features:**
   - **Test Image**: Test recognition on a single image file
   - **View Registered People**: See all people in your trained model
   - **Settings**: Configure camera index and model type

### Command Line Usage (Alternative)

If you prefer command line, you can still use the original scripts:

**Train the model:**
```bash
python detector.py --train
```

**Run live camera:**
```bash
python live_camera.py
```

**Test an image:**
```bash
python detector.py --test -f path/to/image.jpg
```

## Tips for Best Results

1. **Training Images:**
   - Use multiple images per person (at least 3-5)
   - Use clear, front-facing photos
   - Ensure good lighting in training images
   - Include variety (different angles, expressions, lighting)

2. **Camera Setup:**
   - Ensure good lighting when using the camera
   - Face the camera directly for best recognition
   - Keep a reasonable distance (2-5 feet)

3. **Performance:**
   - HOG model works well on CPU and is faster
   - CNN model is more accurate but requires GPU and is slower
   - For live recognition, HOG is recommended unless you have a powerful GPU

## Troubleshooting

**Camera not opening:**
- Check if another application is using the camera
- Try different camera indices: `python live_camera.py -c 1` or `-c 2`
- On Windows, check camera permissions in Settings

**Poor recognition:**
- Add more training images
- Ensure training images are clear and well-lit
- Retrain the model after adding more images

**Import errors:**
- Make sure all dependencies are installed with exact versions: `pip install -r requirements.txt`
- Ensure CMake 4.2.1+ and gcc are properly installed
- Verify Python version: `python --version` (should be 3.12.7 or compatible)

**Version conflicts:**
- Use a virtual environment to avoid conflicts
- Install exact versions from requirements.txt
- If issues persist, try: `pip install --upgrade pip` then reinstall

## License

This project is for educational purposes.

