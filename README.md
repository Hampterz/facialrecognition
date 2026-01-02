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

Before installing, make sure you have:
- Python 3.9 or 3.10 (3.11 may have compatibility issues)
- CMake installed
- C compiler (gcc) - on Windows, install MinGW

### Installing CMake and gcc

**Windows:**
- Download CMake from [CMake downloads page](https://cmake.org/download/)
- Install MinGW using Chocolatey: `choco install mingw`

**Linux:**
- `sudo apt-get install cmake gcc` (Ubuntu/Debian)
- `sudo yum install cmake gcc` (CentOS/RHEL)

**macOS:**
- `brew install cmake gcc`

## Installation

### Quick Install

```bash
pip install -r requirements.txt
```

This will install:
- **YOLOv8** for face detection (no dlib needed!)
- **face-recognition** for face encoding/matching
- **OpenCV** for camera support
- All other dependencies

### What's New: YOLOv8 Face Detection

We've replaced dlib with **YOLOv8 Face Detection** from Hugging Face:
- ✅ **No more dlib compatibility issues!**
- ✅ Works with Python 3.12
- ✅ More accurate face detection
- ✅ Faster and more modern

The YOLOv8 model (~6MB) will be automatically downloaded on first run.

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
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Ensure CMake and gcc are properly installed

## License

This project is for educational purposes.

