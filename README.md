# Face Recognition System with Live Voice Calls

A comprehensive Python-based face recognition system with a modern GUI that can recognize faces from static images and live camera feed. Features include advanced facial analysis (emotion, age, gender, race detection), support for multiple face detection models, and real-time voice conversations with Google Gemini Live API.

## Features

- 🎨 **Beautiful Modern GUI** - Dark theme interface with sleek design
- 🎓 **Train Model** - Add people, upload photos/videos, and train your model
- 📹 **Live Camera Recognition** - Real-time face recognition from USB camera
- 🖼️ **Test Images/Videos** - Test recognition on single images or video files
- 👥 **Manage People** - View and delete registered people
- ⚙️ **Settings** - Configure camera, model, and API keys
- 🎭 **Emotion/Age/Race Analysis** - DeepFace model provides detailed facial analysis
- 🎙️ **Live Voice Calls** - Real-time voice conversations with Google Gemini Live API (client-to-server)

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

### Step 1: Clone Repository (or Download)

```bash
git clone https://github.com/Hampterz/facialrecognition.git
cd facialrecognition
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3.12 -m venv venv
source venv/bin/activate
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install the following **exact versions** (tested and working):

**Core Face Detection & Recognition:**
- **ultralytics==8.3.245** - YOLO models for face detection
- **huggingface-hub==0.36.0** - Model downloading
- **supervision==0.27.0** - Detection utilities
- **face-recognition==1.3.0** - Face encoding and matching
- **retina-face==0.0.17** - RetinaFace detector (optional)
- **deepface==0.0.96** - DeepFace for face recognition + emotion/age/race/gender analysis (optional)

**Image & Video Processing:**
- **numpy==1.26.4** - Numerical operations
- **Pillow==10.3.0** - Image processing
- **opencv-python==4.11.0.86** - Camera and video support

**Deep Learning Frameworks:**
- **torch==2.9.1** - PyTorch framework
- **torchvision==0.24.1** - Vision utilities

**Gemini Live API (Client-to-Server) - Optional:**
- **websockets>=12.0** - WebSocket client for Live API
- **pyaudio>=0.2.14** - Audio input/output for voice calls
- **google-generativeai>=0.8.6** - Google Gemini API SDK


### What's New: Multi-Model Face Detection

We support **four face detection models**:
- ✅ **YOLOv11** (Default) - Latest YOLO, best accuracy
- ✅ **YOLOv8** - Stable YOLO version
- ✅ **RetinaFace** - Deep learning with landmarks (optional)
- ✅ **DeepFace** - Face recognition + Emotion/Age/Race/Gender analysis (optional)

**No more dlib compatibility issues!** Works with Python 3.12.

**DeepFace Features:**
- Face recognition with multiple backends (VGG-Face, Facenet, OpenFace, etc.)
- Real-time emotion detection (happy, sad, angry, etc.)
- Age estimation
- Gender classification
- Race/ethnicity analysis
- All analysis shown in live camera and test image results

Models will be automatically downloaded on first use (~6-120MB each).

### Verify Installation

Run the verification script:
```bash
python verify_setup.py
```

This will check if all required packages are installed correctly.

## Project Structure

```
facial-recognition/
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
│   ├── encodings_yolov11.pkl
│   ├── encodings_yolov8.pkl
│   ├── encodings_retinaface.pkl
│   └── encodings_deepface.pkl
│
├── models/            # Downloaded YOLO models (auto-created)
│   └── yolov11n_face_detection.pt
│
├── app.py             # Main GUI application (START HERE!)
├── detector.py        # Training and image recognition (CLI)
├── live_camera.py     # Live camera recognition (CLI)
├── yolo_face_detector.py      # YOLOv11 detector
├── yolov8_detector.py         # YOLOv8 detector
├── retinaface_detector.py     # RetinaFace detector
├── deepface_detector.py       # DeepFace detector with analysis
├── video_utils.py             # Video processing utilities
├── requirements.txt           # All dependencies with exact versions
└── README.md                  # This file
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
   - Select your preferred detection model (YOLOv11, YOLOv8, RetinaFace, or DeepFace)
   - Enter a person's name in the "Person Name" field
   - Click "📷 Add Photos" and select multiple photos of that person (at least 3-5 photos recommended)
   - Or use "📁 Import Folder" to import a folder with subfolders (each subfolder = one person)
   - Or use "🎬 Add Video" to extract frames from video files
   - Repeat for each person you want to recognize
   - Select encoding model type (HOG for CPU/faster, CNN for GPU/more accurate)
   - Click "🚀 Train Model" to train
   - Wait for the training to complete
   - **Note:** Each model trains independently - switch models to use different training data

3. **Start Live Recognition:**
   - Select your preferred model from the homepage dropdown (or keep default YOLOv11)
   - Click "📹 Live Camera Recognition" on the homepage
   - Walk in front of the camera - your name will appear when recognized
   - **With DeepFace model:** You'll see emotion, age, gender, and race analysis in the top-left overlay
   - Use camera controls to flip or rotate the camera feed
   - **Live Voice Calls:** Click "🎙️ Live Call: OFF" to start a real-time voice conversation with Gemini
   - Click "Stop" to close the camera

4. **Live Voice Calls with Gemini:**
   - **Setup:** Enter your Gemini API key in Settings (get free key from https://makersuite.google.com/app/apikey)
   - **Start Call:** Click "🎙️ Live Call: OFF" button in the camera window to enable
   - **Speak:** Talk naturally - Gemini will respond in real-time
   - **Features:** Low-latency bidirectional audio, voice activity detection, natural conversations
   - **Requirements:** PyAudio and websockets must be installed

5. **Other Features:**
   - **Test Image/Video**: Test recognition on a single image or video file (with DeepFace analysis if using DeepFace model)
   - **View Registered People**: See all people in your trained model
   - **Settings**: Configure camera index and encoding model
   - **Model Selection**: Switch between YOLOv11, YOLOv8, RetinaFace, and DeepFace models
   - **Incremental Training**: Only new photos are processed on subsequent training runs
   - **Folder Import**: Import entire folders with subfolders (each subfolder = one person)
   - **Video Training**: Extract frames from video files for training

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

**Live API not working:**
- Make sure PyAudio is installed: `pip install pyaudio` (or `pipwin install pyaudio` on Windows)
- Install websockets: `pip install websockets`
- Verify your Gemini API key is correct in Settings
- Check microphone permissions in your system settings
- Use headphones to prevent echo/feedback

## New Features Summary

### 🎭 DeepFace Integration
- Emotion detection (happy, sad, angry, neutral, etc.)
- Age estimation
- Gender classification  
- Race/ethnicity analysis
- Real-time overlay in camera window

### 🎙️ Live Voice Calls (Gemini Live API)
- Real-time bidirectional voice conversations
- Client-to-server WebSocket connection
- Low-latency audio streaming (16kHz input, 24kHz output)
- Voice Activity Detection (VAD) for natural interruptions
- Native audio output with natural speech
- Works seamlessly with face recognition

### 📁 Enhanced Training
- Folder import (subfolders = people)
- Video file support (frame extraction)
- Incremental training (only new files processed)
- Multi-model support (each model trains independently)

## License

This project is for educational purposes.

