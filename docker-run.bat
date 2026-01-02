@echo off
REM Quick Docker run script for Face Recognition System (Windows)

echo Face Recognition System - Docker Runner
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if image exists
docker images | findstr /C:"facialrecognition" >nul 2>&1
if errorlevel 1 (
    echo Building Docker image...
    docker build -t facialrecognition:latest .
)

REM Create directories if they don't exist
if not exist "training" mkdir training
if not exist "output" mkdir output
if not exist "validation" mkdir validation
if not exist "models" mkdir models

REM Run container
echo Starting container...
docker run -it --rm ^
    -v "%cd%\training:/app/training" ^
    -v "%cd%\output:/app/output" ^
    -v "%cd%\validation:/app/validation" ^
    -v "%cd%\models:/app/models" ^
    facialrecognition:latest

echo Container stopped.
pause

