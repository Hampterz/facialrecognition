# app.py - Main GUI Application for Face Recognition System (Modern Dark Theme)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import cv2
from PIL import Image, ImageTk
import face_recognition
import pickle
from pathlib import Path
from collections import Counter
import shutil
import os
import numpy as np
from yolo_face_detector import YOLOFaceDetector
from yolov8_detector import YOLOv8FaceDetector
from retinaface_detector import RetinaFaceDetector
from deepface_detector import DeepFaceDetector
from video_utils import extract_frames_from_video, process_video_for_training, get_video_frames
from speech_recognition_module import initialize_whisper, transcribe_audio
from gemini_api import initialize_gemini, get_gemini_response

# Lazy imports for optional dependencies
pyaudio = None
try:
    import pyaudio
except ImportError:
    pass

# Model-specific encoding paths
ENCODINGS_PATHS = {
    "yolov8": Path("output/encodings_yolov8.pkl"),
    "yolov11": Path("output/encodings_yolov11.pkl"),
    "retinaface": Path("output/encodings_retinaface.pkl"),
    "deepface": Path("output/encodings_deepface.pkl"),
}

# Processed files paths for each model
PROCESSED_FILES_PATHS = {
    "yolov8": Path("output/processed_files_yolov8.pkl"),
    "yolov11": Path("output/processed_files_yolov11.pkl"),
    "retinaface": Path("output/processed_files_retinaface.pkl"),
    "deepface": Path("output/processed_files_deepface.pkl"),
}

# Default model
DEFAULT_MODEL = "yolov11"
TRAINING_DIR = Path("training")
OUTPUT_DIR = Path("output")
VALIDATION_DIR = Path("validation")

# Ensure directories exist
TRAINING_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VALIDATION_DIR.mkdir(exist_ok=True)

# Modern Dark Theme Colors
COLORS = {
    "bg_primary": "#0a0e27",      # Deep dark blue-black
    "bg_secondary": "#1a1f3a",   # Dark blue-gray
    "bg_tertiary": "#252b45",     # Medium dark
    "accent_blue": "#4a9eff",     # Bright blue
    "accent_green": "#00d4aa",    # Teal green
    "accent_purple": "#9d7ce8",   # Purple
    "accent_orange": "#ff6b6b",  # Coral
    "text_primary": "#ffffff",    # White
    "text_secondary": "#b8c5d1", # Light gray
    "border": "#2d3447",          # Subtle border
    "success": "#00d4aa",         # Success green
    "warning": "#ffa726",         # Warning orange
    "error": "#ff5252",           # Error red
}


class ModernButton(tk.Button):
    """Modern button with hover effects."""
    def __init__(self, parent, **kwargs):
        self.default_bg = kwargs.get('bg', COLORS['accent_blue'])
        self.hover_bg = self._lighten_color(self.default_bg)
        kwargs['bg'] = self.default_bg
        kwargs['activebackground'] = self.hover_bg
        kwargs['relief'] = tk.FLAT
        kwargs['bd'] = 0
        kwargs['cursor'] = 'hand2'
        kwargs['font'] = kwargs.get('font', ("Segoe UI", 11, "bold"))
        kwargs['padx'] = kwargs.get('padx', 20)
        kwargs['pady'] = kwargs.get('pady', 12)
        
        super().__init__(parent, **kwargs)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
    
    def on_enter(self, e):
        self['bg'] = self.hover_bg
    
    def on_leave(self, e):
        self['bg'] = self.default_bg
    
    def _lighten_color(self, color):
        """Lighten a hex color."""
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, c + 30) for c in rgb)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLORS["bg_primary"])
        
        # Center window on screen
        self.center_window()
        
        # Variables
        self.current_person_name = tk.StringVar()
        self.model_type = tk.StringVar(value="hog")  # For face_recognition encoding model
        self.detection_model = tk.StringVar(value=DEFAULT_MODEL)  # Face detection model
        self.camera_index = tk.IntVar(value=0)
        self.camera_running = False
        self.video_capture = None
        self.video_processing = False
        self.camera_flip_horizontal = tk.BooleanVar(value=False)
        self.camera_flip_vertical = tk.BooleanVar(value=False)
        self.camera_rotate = tk.IntVar(value=0)  # 0, 90, 180, 270 degrees
        
        # Audio and Gemini settings
        self.gemini_api_key = tk.StringVar(value="")
        self.audio_enabled = False
        self.audio_recording = False
        self.audio_stream = None
        self.pyaudio_instance = None
        self.audio_frames = []
        self.audio_sample_rate = 16000
        self.audio_chunk_size = 1024
        # Set audio format only if pyaudio is available
        if pyaudio is not None:
            self.audio_format = pyaudio.paInt16
        else:
            self.audio_format = None
        self.audio_channels = 1
        
        # Load saved Gemini API key if exists
        self.load_gemini_api_key()
        
        # Model-specific data
        self.loaded_encodings = {}  # Dict: {model_name: encodings}
        self.processed_files = {}  # Dict: {model_name: set of files}
        self.detectors = {}  # Cache detectors
        
        # Load encodings for all models
        self.load_all_encodings()
        self.load_all_processed_files()
        
        # Create UI
        self.create_homepage()
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def load_all_encodings(self):
        """Load face encodings for all models."""
        for model_name, encodings_path in ENCODINGS_PATHS.items():
            try:
                if encodings_path.exists():
                    with encodings_path.open(mode="rb") as f:
                        self.loaded_encodings[model_name] = pickle.load(f)
                else:
                    self.loaded_encodings[model_name] = None
            except Exception as e:
                print(f"Error loading encodings for {model_name}: {e}")
                self.loaded_encodings[model_name] = None
    
    def load_all_processed_files(self):
        """Load processed files for all models."""
        for model_name, processed_path in PROCESSED_FILES_PATHS.items():
            try:
                if processed_path.exists():
                    with processed_path.open(mode="rb") as f:
                        self.processed_files[model_name] = pickle.load(f)
                else:
                    self.processed_files[model_name] = set()
            except Exception as e:
                print(f"Error loading processed files for {model_name}: {e}")
                self.processed_files[model_name] = set()
    
    def save_processed_files(self, model_name=None):
        """Save list of processed files for a specific model."""
        if model_name is None:
            model_name = self.detection_model.get()
        
        try:
            processed_path = PROCESSED_FILES_PATHS[model_name]
            with processed_path.open(mode="wb") as f:
                pickle.dump(self.processed_files.get(model_name, set()), f)
        except Exception as e:
            print(f"Error saving processed files for {model_name}: {e}")
    
    def get_current_encodings(self):
        """Get encodings for current detection model."""
        model_name = self.detection_model.get()
        return self.loaded_encodings.get(model_name)
    
    def get_current_processed_files(self):
        """Get processed files for current detection model."""
        model_name = self.detection_model.get()
        return self.processed_files.get(model_name, set())
    
    def get_detector(self):
        """Get detector for current model. Only loads the selected model."""
        model_name = self.detection_model.get()
        
        # Only load the selected model (unloading is handled in on_model_change)
        if model_name not in self.detectors:
            if model_name == "yolov8":
                self.detectors[model_name] = YOLOv8FaceDetector()
            elif model_name == "yolov11":
                self.detectors[model_name] = YOLOFaceDetector()
            elif model_name == "retinaface":
                try:
                    self.detectors[model_name] = RetinaFaceDetector()
                except ImportError:
                    raise ImportError(
                        "RetinaFace is not installed. Please install it with: pip install retina-face"
                    )
            elif model_name == "deepface":
                try:
                    self.detectors[model_name] = DeepFaceDetector()
                except ImportError:
                    raise ImportError(
                        "DeepFace is not installed. Please install it with: pip install deepface"
                    )
        
        return self.detectors[model_name]
    
    def clear_frame(self):
        """Clear all widgets from the main frame."""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def create_homepage(self):
        """Create the main homepage with modern dark theme."""
        self.clear_frame()
        
        # Header with gradient effect
        header_frame = tk.Frame(self.root, bg=COLORS["bg_secondary"], height=120)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title with icon
        title_container = tk.Frame(header_frame, bg=COLORS["bg_secondary"])
        title_container.pack(expand=True, fill=tk.BOTH, pady=20)
        
        title_label = tk.Label(
            title_container,
            text="🎯 Face Recognition System",
            font=("Segoe UI", 32, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_container,
            text="AI-Powered Face Detection & Recognition",
            font=("Segoe UI", 12),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Model Selection Card
        model_card = tk.Frame(
            content_frame,
            bg=COLORS["bg_secondary"],
            relief=tk.FLAT,
            bd=0
        )
        model_card.pack(fill=tk.X, pady=(0, 20))
        
        model_header = tk.Frame(model_card, bg=COLORS["bg_secondary"])
        model_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(
            model_header,
            text="🔧 Active Detection Model:",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(side=tk.LEFT, padx=(0, 15))
        
        # Check if RetinaFace is available
        retinaface_available = False
        try:
            import retinaface
            retinaface_available = True
        except:
            pass
        
        model_options_home = ["yolov11", "yolov8"]
        if retinaface_available:
            model_options_home.append("retinaface")
        
        model_combo_home = ttk.Combobox(
            model_header,
            textvariable=self.detection_model,
            values=model_options_home,
            state="readonly",
            font=("Segoe UI", 11),
            width=15
        )
        model_combo_home.pack(side=tk.LEFT, padx=5)
        
        # Model description
        model_desc_home = tk.Label(
            model_header,
            text="",
            font=("Segoe UI", 9),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        )
        model_desc_home.pack(side=tk.LEFT, padx=15)
        
        def update_home_model_desc(*args):
            model_name = self.detection_model.get()
            desc_map = {
                "yolov11": "Latest YOLO, best accuracy",
                "yolov8": "Stable YOLO version",
                "retinaface": "Deep learning with landmarks",
                "deepface": "Face recognition + Emotion/Age/Race/Gender"
            }
            model_desc_home.config(text=desc_map.get(model_name, ""))
            # Unload other models
            self.root.after_idle(lambda: self._unload_other_models(model_name))
            # Update status
            self.root.after(100, lambda: self._update_homepage_status())
        
        self.detection_model.trace('w', update_home_model_desc)
        update_home_model_desc()
        
        # Status card with modern design
        status_card = tk.Frame(
            content_frame,
            bg=COLORS["bg_secondary"],
            relief=tk.FLAT,
            bd=0
        )
        status_card.pack(fill=tk.X, pady=(0, 30))
        
        # Status header
        status_header = tk.Frame(status_card, bg=COLORS["bg_secondary"])
        status_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(
            status_header,
            text="📊 System Status",
            font=("Segoe UI", 14, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(side=tk.LEFT)
        
        # Status indicator (will be updated dynamically)
        status_body = tk.Frame(status_card, bg=COLORS["bg_secondary"])
        status_body.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.status_label = tk.Label(
            status_body,
            text="",
            font=("Segoe UI", 11),
            bg=COLORS["bg_secondary"],
            fg=COLORS["success"],
            anchor="w"
        )
        self.status_label.pack(fill=tk.X)
        
        # Initial status update
        self._update_homepage_status()
        
        # Buttons grid with modern cards
        buttons_container = tk.Frame(content_frame, bg=COLORS["bg_primary"])
        buttons_container.pack(fill=tk.BOTH, expand=True)
        
        # Create button cards
        buttons = [
            {
                "text": "🎓 Train Model",
                "command": self.show_training_page,
                "color": COLORS["accent_blue"],
                "desc": "Add people and train recognition"
            },
            {
                "text": "📹 Live Recognition",
                "command": self.start_live_recognition,
                "color": COLORS["accent_green"],
                "desc": "Real-time camera recognition"
            },
            {
                "text": "🖼️ Test Image",
                "command": self.test_image,
                "color": COLORS["accent_purple"],
                "desc": "Test on uploaded images"
            },
            {
                "text": "👥 View People",
                "command": self.view_registered_people,
                "color": COLORS["accent_orange"],
                "desc": "Browse registered people"
            },
            {
                "text": "⚙️ Settings",
                "command": self.show_settings,
                "color": COLORS["bg_tertiary"],
                "desc": "Configure system settings"
            },
        ]
        
        # Create button grid
        for i, btn_info in enumerate(buttons):
            row = i // 2
            col = i % 2
            
            # Card frame
            card = tk.Frame(
                buttons_container,
                bg=COLORS["bg_secondary"],
                relief=tk.FLAT,
                bd=0
            )
            card.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")
            card.grid_columnconfigure(0, weight=1)
            
            # Button
            btn = ModernButton(
                card,
                text=btn_info["text"],
                bg=btn_info["color"],
                fg=COLORS["text_primary"],
                command=btn_info["command"],
                font=("Segoe UI", 12, "bold"),
                width=25,
                pady=15
            )
            btn.pack(fill=tk.X, padx=20, pady=(20, 10))
            
            # Description
            desc_label = tk.Label(
                card,
                text=btn_info["desc"],
                font=("Segoe UI", 9),
                bg=COLORS["bg_secondary"],
                fg=COLORS["text_secondary"]
            )
            desc_label.pack(pady=(0, 20))
        
        # Configure grid weights
        buttons_container.grid_columnconfigure(0, weight=1)
        buttons_container.grid_columnconfigure(1, weight=1)
        buttons_container.grid_rowconfigure(0, weight=1)
        buttons_container.grid_rowconfigure(1, weight=1)
        buttons_container.grid_rowconfigure(2, weight=1)
    
    def show_training_page(self):
        """Show the training page with modern dark theme."""
        self.clear_frame()
        
        # Header
        header_frame = tk.Frame(self.root, bg=COLORS["bg_secondary"], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        back_btn = ModernButton(
            header_frame,
            text="← Home",
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            command=self.create_homepage,
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8
        )
        back_btn.pack(side=tk.LEFT, padx=20, pady=20)
        
        title_label = tk.Label(
            header_frame,
            text="🎓 Train Model",
            font=("Segoe UI", 24, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Main content
        content_frame = tk.Frame(self.root, bg=COLORS["bg_primary"])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Left panel - Add Person
        left_card = tk.Frame(
            content_frame,
            bg=COLORS["bg_secondary"],
            relief=tk.FLAT
        )
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(
            left_card,
            text="Add New Person",
            font=("Segoe UI", 16, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(pady=(20, 15))
        
        tk.Label(
            left_card,
            text="Person Name:",
            font=("Segoe UI", 11),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        ).pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        name_entry = tk.Entry(
            left_card,
            textvariable=self.current_person_name,
            font=("Segoe UI", 11),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent_blue"]
        )
        name_entry.pack(fill=tk.X, padx=20, pady=5, ipady=8)
        
        # Buttons frame
        buttons_frame = tk.Frame(left_card, bg=COLORS["bg_secondary"])
        buttons_frame.pack(pady=20)
        
        add_photos_btn = ModernButton(
            buttons_frame,
            text="📷 Add Photos",
            bg=COLORS["accent_blue"],
            fg=COLORS["text_primary"],
            command=lambda: self.add_photos_for_person(self.current_person_name.get()),
            font=("Segoe UI", 11, "bold"),
            width=18,
            pady=12
        )
        add_photos_btn.pack(pady=5)
        
        add_video_btn = ModernButton(
            buttons_frame,
            text="🎥 Add Video",
            bg=COLORS["accent_purple"],
            fg=COLORS["text_primary"],
            command=lambda: self.add_video_for_person(self.current_person_name.get()),
            font=("Segoe UI", 11, "bold"),
            width=18,
            pady=12
        )
        add_video_btn.pack(pady=5)
        
        import_folder_btn = ModernButton(
            buttons_frame,
            text="📁 Import Folder",
            bg=COLORS["accent_orange"],
            fg=COLORS["text_primary"],
            command=self.import_from_folder,
            font=("Segoe UI", 11, "bold"),
            width=18,
            pady=12
        )
        import_folder_btn.pack(pady=5)
        
        # Right panel - Training
        right_card = tk.Frame(
            content_frame,
            bg=COLORS["bg_secondary"],
            relief=tk.FLAT
        )
        right_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        tk.Label(
            right_card,
            text="Training Configuration",
            font=("Segoe UI", 16, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(pady=(20, 15))
        
        # Face Detection Model Selection
        tk.Label(
            right_card,
            text="Face Detection Model:",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(anchor=tk.W, padx=20, pady=(10, 5))
        
        model_desc = tk.Label(
            right_card,
            text="Each model has separate training data",
            font=("Segoe UI", 9),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        )
        model_desc.pack(anchor=tk.W, padx=20, pady=(0, 10))
        
        model_dropdown_frame = tk.Frame(right_card, bg=COLORS["bg_secondary"])
        model_dropdown_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # Check if RetinaFace is available (lazy check - don't load model)
        retinaface_available = False
        retinaface_error = None
        try:
            # Just check if module can be imported, don't create detector
            import retinaface
            retinaface_available = True
        except ImportError as e:
            retinaface_available = False
            retinaface_error = str(e)
        except Exception as e:
            retinaface_available = False
            retinaface_error = f"{type(e).__name__}: {str(e)}"
        
        # Check if DeepFace is available
        deepface_available = False
        deepface_error = None
        try:
            import deepface
            deepface_available = True
        except ImportError as e:
            deepface_available = False
            deepface_error = str(e)
        except Exception as e:
            deepface_available = False
            deepface_error = f"{type(e).__name__}: {str(e)}"
        
        model_options = [
            ("YOLOv11", "yolov11", "Latest YOLO, best accuracy"),
            ("YOLOv8", "yolov8", "Stable YOLO version"),
        ]
        
        if retinaface_available:
            model_options.append(("RetinaFace", "retinaface", "Deep learning with landmarks"))
        else:
            # Show specific error if available
            if retinaface_error and "tf-keras" in retinaface_error.lower():
                model_options.append(("RetinaFace (Needs tf-keras)", "retinaface", "Install: pip install tf-keras"))
            else:
                model_options.append(("RetinaFace (Not Available)", "retinaface", "Check dependencies"))
        
        if deepface_available:
            model_options.append(("DeepFace", "deepface", "Face recognition + Emotion/Age/Race/Gender"))
        else:
            model_options.append(("DeepFace (Not Installed)", "deepface", "Install: pip install deepface"))
        
        # Model combo for training page
        model_combo = ttk.Combobox(
            model_dropdown_frame,
            textvariable=self.detection_model,
            values=[opt[1] for opt in model_options],
            state="readonly",
            font=("Segoe UI", 10),
            width=25
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Disable RetinaFace if not available
        if not retinaface_available:
            def check_retinaface_selection(*args):
                if self.detection_model.get() == "retinaface":
                    if retinaface_error and "tf-keras" in retinaface_error.lower():
                        msg = (
                            "RetinaFace requires tf-keras package.\n\n"
                            "Please install it:\n"
                            "pip install tf-keras\n\n"
                            "Switching to YOLOv11..."
                        )
                    else:
                        msg = (
                            "RetinaFace is not available.\n\n"
                            f"Error: {retinaface_error or 'Unknown error'}\n\n"
                            "Please install dependencies or use YOLOv8/YOLOv11.\n\n"
                            "Switching to YOLOv11..."
                        )
                    messagebox.showwarning("RetinaFace Not Available", msg)
                    self.detection_model.set("yolov11")
            
            # Use trace_add for Python 3.8+ compatibility
            try:
                self.detection_model.trace_add('write', check_retinaface_selection)
            except AttributeError:
                # Fallback for older Python versions
                self.detection_model.trace('w', check_retinaface_selection)
        
        # Model info label
        self.model_info_label = tk.Label(
            model_dropdown_frame,
            text="",
            font=("Segoe UI", 9),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        )
        self.model_info_label.pack(side=tk.LEFT, padx=10)
        
        def update_model_info(*args):
            model_name = self.detection_model.get()
            for opt in model_options:
                if opt[1] == model_name:
                    self.model_info_label.config(text=opt[2])
                    break
        
        def on_model_change(*args):
            model_name = self.detection_model.get()
            # Unload other models when switching (non-blocking)
            self.root.after_idle(lambda: self._unload_other_models(model_name))
            # Update status (non-blocking)
            self.root.after(100, lambda: self._update_model_status(model_name))
            # Update model info
            update_model_info()
        
        self.detection_model.trace('w', on_model_change)
        update_model_info()  # Initial update
        
        # Encoding Model selection (for face_recognition library)
        tk.Label(
            right_card,
            text="Encoding Model:",
            font=("Segoe UI", 11),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        ).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        encoding_frame = tk.Frame(right_card, bg=COLORS["bg_secondary"])
        encoding_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Radiobutton(
            encoding_frame,
            text="HOG (CPU - Faster)",
            variable=self.model_type,
            value="hog",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"]
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(
            encoding_frame,
            text="CNN (GPU - More Accurate)",
            variable=self.model_type,
            value="cnn",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"]
        ).pack(side=tk.LEFT, padx=10)
        
        # Train button
        train_btn = ModernButton(
            right_card,
            text="🚀 Train Model",
            bg=COLORS["accent_green"],
            fg=COLORS["text_primary"],
            command=self.train_model,
            font=("Segoe UI", 12, "bold"),
            width=20,
            pady=15
        )
        train_btn.pack(pady=30)
        
        # Status label
        self.training_status = tk.Label(
            right_card,
            text="",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["success"],
            wraplength=300
        )
        self.training_status.pack(pady=10)
        
        # Bottom panel - Registered People
        list_card = tk.Frame(
            content_frame,
            bg=COLORS["bg_secondary"],
            relief=tk.FLAT
        )
        list_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(
            list_card,
            text="Registered People",
            font=("Segoe UI", 16, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(pady=(20, 10))
        
        # Scrollable list
        list_container = tk.Frame(list_card, bg=COLORS["bg_secondary"])
        list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.people_listbox = tk.Listbox(
            list_container,
            font=("Segoe UI", 10),
            yscrollcommand=scrollbar.set,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["accent_blue"],
            selectforeground=COLORS["text_primary"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        self.people_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.people_listbox.yview)
        
        # Initialize people list
        self.update_people_list()
        
        # Delete button
        delete_btn = ModernButton(
            list_card,
            text="🗑️ Delete Selected",
            bg=COLORS["error"],
            fg=COLORS["text_primary"],
            command=self.delete_person,
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8
        )
        delete_btn.pack(pady=(0, 20))
    
    def _unload_other_models(self, current_model):
        """Unload models that are not currently selected."""
        for other_model in list(self.detectors.keys()):
            if other_model != current_model:
                try:
                    detector = self.detectors[other_model]
                    if hasattr(detector, 'model'):
                        del detector.model
                    del self.detectors[other_model]
                except:
                    pass
        # Keep only current model in cache
        if current_model not in self.detectors:
            self.detectors = {}
        else:
            self.detectors = {current_model: self.detectors[current_model]}
    
    def _update_model_status(self, model_name):
        """Update model status (called asynchronously to avoid lag)."""
        current_encodings = self.get_current_encodings()
        if current_encodings:
            num_people = len(set(current_encodings.get("names", [])))
            self.training_status.config(
                text=f"✓ {model_name.upper()} model selected - {num_people} person(s) trained",
                fg=COLORS["success"]
            )
        else:
            self.training_status.config(
                text=f"⚠ {model_name.upper()} model selected - Not trained yet",
                fg=COLORS["warning"]
            )
    
    def _update_homepage_status(self):
        """Update homepage status label."""
        if hasattr(self, 'status_label'):
            try:
                current_encodings = self.get_current_encodings()
                if current_encodings:
                    num_people = len(set(current_encodings.get("names", [])))
                    model_name = self.detection_model.get().upper()
                    status_text = f"✓ {model_name} Active - {num_people} person(s) registered"
                    status_color = COLORS["success"]
                else:
                    model_name = self.detection_model.get().upper()
                    status_text = f"⚠ {model_name} not trained - Train the model to get started"
                    status_color = COLORS["warning"]
                
                self.status_label.config(text=status_text, fg=status_color)
            except Exception as e:
                # Silently fail if status_label doesn't exist yet
                pass
    
    def add_photos_for_person(self, person_name):
        """Open file dialog to add photos for a person."""
        if not person_name or not person_name.strip():
            messagebox.showerror("Error", "Please enter a person name first!")
            return
        
        person_name = person_name.strip().replace(" ", "_")
        person_dir = TRAINING_DIR / person_name
        person_dir.mkdir(exist_ok=True)
        
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Photos",
            filetypes=filetypes
        )
        
        if files:
            copied = 0
            for file_path in files:
                try:
                    filename = os.path.basename(file_path)
                    dest_path = person_dir / filename
                    shutil.copy2(file_path, dest_path)
                    copied += 1
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to copy {filename}: {str(e)}")
            
            messagebox.showinfo("Success", f"Added {copied} photo(s) for {person_name}")
            self.update_people_list()
    
    def import_from_folder(self):
        """Import photos from a folder structure where subfolders are person names."""
        folder_path = filedialog.askdirectory(
            title="Select Folder with Person Subfolders",
            mustexist=True
        )
        
        if not folder_path:
            return
        
        folder_path = Path(folder_path)
        
        # Find all subfolders
        subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
        
        if not subfolders:
            messagebox.showwarning(
                "Warning",
                "No subfolders found! Please select a folder containing subfolders named after people."
            )
            return
        
        # Show progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Importing Photos")
        progress_window.geometry("500x200")
        progress_window.configure(bg=COLORS["bg_primary"])
        
        progress_label = tk.Label(
            progress_window,
            text=f"Scanning {len(subfolders)} folder(s)...",
            font=("Segoe UI", 11),
            bg=COLORS["bg_primary"],
            fg=COLORS["text_primary"]
        )
        progress_label.pack(pady=30)
        
        def import_thread():
            try:
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP'}
                total_copied = 0
                people_imported = []
                
                for subfolder in subfolders:
                    person_name = subfolder.name.strip().replace(" ", "_")
                    person_dir = TRAINING_DIR / person_name
                    person_dir.mkdir(exist_ok=True)
                    
                    # Find all image files in subfolder
                    image_files = [f for f in subfolder.iterdir() 
                                 if f.is_file() and f.suffix in image_extensions]
                    
                    if not image_files:
                        continue
                    
                    copied = 0
                    for image_file in image_files:
                        try:
                            # Copy to training directory
                            dest_path = person_dir / image_file.name
                            # If file exists, add timestamp to avoid overwrite
                            if dest_path.exists():
                                stem = dest_path.stem
                                suffix = dest_path.suffix
                                dest_path = person_dir / f"{stem}_imported{suffix}"
                            
                            shutil.copy2(image_file, dest_path)
                            copied += 1
                            total_copied += 1
                        except Exception as e:
                            print(f"Error copying {image_file}: {e}")
                    
                    if copied > 0:
                        people_imported.append(f"{person_name} ({copied} photos)")
                
                progress_window.destroy()
                
                if total_copied > 0:
                    msg = f"Successfully imported {total_copied} photo(s) from {len(people_imported)} person(s):\n\n"
                    msg += "\n".join(people_imported)
                    messagebox.showinfo("Import Complete", msg)
                    self.update_people_list()
                else:
                    messagebox.showwarning("Warning", "No image files found in subfolders!")
                    
            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Error", f"Failed to import folder: {str(e)}")
        
        threading.Thread(target=import_thread, daemon=True).start()
    
    def update_people_list(self):
        """Update the list of registered people."""
        if not hasattr(self, 'people_listbox'):
            return  # People listbox not created yet
        self.people_listbox.delete(0, tk.END)
        if TRAINING_DIR.exists():
            for person_dir in TRAINING_DIR.iterdir():
                if person_dir.is_dir():
                    num_photos = len(list(person_dir.glob("*")))
                    self.people_listbox.insert(
                        tk.END,
                        f"{person_dir.name} ({num_photos} photos)"
                    )
    
    def delete_person(self):
        """Delete selected person and their photos."""
        if not hasattr(self, 'people_listbox'):
            messagebox.showwarning("Warning", "Please go to Training page first!")
            return
        selection = self.people_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete!")
            return
        
        person_name = self.people_listbox.get(selection[0]).split(" (")[0]
        
        if messagebox.askyesno("Confirm", f"Delete {person_name} and all their photos?"):
            person_dir = TRAINING_DIR / person_name
            if person_dir.exists():
                shutil.rmtree(person_dir)
                messagebox.showinfo("Success", f"Deleted {person_name}")
                self.update_people_list()
    
    def convert_image_to_rgb(self, image_path):
        """Convert image to RGB format."""
        pil_image = Image.open(image_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return np.array(pil_image, dtype=np.uint8)
    
    def train_model(self, incremental=True):
        """Train the face recognition model with incremental support."""
        if not any(TRAINING_DIR.iterdir()):
            messagebox.showerror("Error", "No training data found! Please add people and photos first.")
            return
        
        # Check if selected model is available
        model_name = self.detection_model.get()
        if model_name == "retinaface":
            try:
                # Just verify import, don't create detector yet
                import retinaface
            except ImportError as e:
                error_msg = str(e)
                if "tf-keras" in error_msg.lower():
                    messagebox.showerror(
                        "RetinaFace Missing Dependency",
                        "RetinaFace requires tf-keras package.\n\n"
                        "Please install it:\n"
                        "pip install tf-keras\n\n"
                        "Or use YOLOv8 or YOLOv11 instead."
                    )
                else:
                    messagebox.showerror(
                        "RetinaFace Not Installed",
                        "RetinaFace package is not installed.\n\n"
                        "Please install it first:\n"
                        "pip install retina-face\n\n"
                        "Or select a different model (YOLOv8 or YOLOv11)."
                    )
                return
            except Exception as e:
                messagebox.showerror(
                    "RetinaFace Error",
                    f"RetinaFace cannot be used:\n\n{str(e)}\n\n"
                    "Please use YOLOv8 or YOLOv11 instead."
                )
                return
        elif model_name == "deepface":
            try:
                # Just verify import, don't create detector yet
                import deepface
            except ImportError as e:
                messagebox.showerror(
                    "DeepFace Not Installed",
                    "DeepFace package is not installed.\n\n"
                    "Please install it first:\n"
                    "pip install deepface\n\n"
                    "Or select a different model (YOLOv8, YOLOv11, or RetinaFace)."
                )
                return
            except Exception as e:
                messagebox.showerror(
                    "DeepFace Error",
                    f"DeepFace cannot be used:\n\n{str(e)}\n\n"
                    "Please use YOLOv8, YOLOv11, or RetinaFace instead."
                )
                return
        
        self.training_status.config(text="Training in progress... Please wait.", fg=COLORS["warning"])
        self.root.update_idletasks()  # Use update_idletasks instead of update for better performance
        
        def train_thread():
            try:
                # Get current model name
                model_name = self.detection_model.get()
                
                # Get detector early to catch any errors
                try:
                    detector = self.get_detector()
                except Exception as e:
                    self.root.after(0, lambda err=str(e): messagebox.showerror(
                        "Detector Error",
                        f"Failed to load {model_name} detector:\n\n{err}\n\nPlease try a different model."
                    ))
                    self.root.after(0, lambda: self.training_status.config(
                        text="Training failed - Detector error",
                        fg=COLORS["error"]
                    ))
                    return
                
                current_encodings = self.get_current_encodings()
                
                # Load existing encodings for incremental training
                existing_names = []
                existing_encodings = []
                if incremental and current_encodings:
                    existing_names = current_encodings.get("names", [])
                    existing_encodings = current_encodings.get("encodings", [])
                
                names = list(existing_names) if incremental else []
                encodings = list(existing_encodings) if incremental else []
                processed_count = 0
                new_count = 0
                skipped_count = 0
                error_count = 0
                error_files = []
                
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG', '.BMP'}
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
                all_files = list(TRAINING_DIR.glob("*/*"))
                
                # Filter files to process
                files_to_process = []
                for f in all_files:
                    if f.is_file() and (f.suffix in image_extensions or f.suffix in video_extensions):
                        # For incremental training, check if file is new or modified
                        file_key = str(f.relative_to(TRAINING_DIR))
                        processed_files_set = self.get_current_processed_files()
                        if incremental and file_key in processed_files_set:
                            # Check if file was modified
                            try:
                                if f.exists():
                                    current_mtime = f.stat().st_mtime
                                    # If file was modified, reprocess it
                                    if file_key not in self.processed_files or True:  # Always check
                                        files_to_process.append((f, file_key))
                                    else:
                                        skipped_count += 1
                                else:
                                    skipped_count += 1
                            except:
                                files_to_process.append((f, file_key))
                        else:
                            files_to_process.append((f, file_key))
                
                total_files = len(files_to_process)
                
                if total_files == 0:
                    if incremental and len(names) > 0:
                        # All files already processed
                        self.root.after(0, lambda: messagebox.showinfo(
                            "Info",
                            "All files are already trained! No new files to process."
                        ))
                        self.root.after(0, lambda: self.training_status.config(
                            text=f"✓ All files trained ({len(set(names))} person(s))",
                            fg=COLORS["success"]
                        ))
                    else:
                        self.root.after(0, lambda: messagebox.showerror(
                            "Error",
                            "No image files found in training directory! Please add photos first."
                        ))
                        self.root.after(0, lambda: self.training_status.config(text="", fg=COLORS["success"]))
                    return
                
                for filepath, file_key in files_to_process:
                    name = filepath.parent.name
                    try:
                        if filepath.suffix.lower() in [ext.lower() for ext in image_extensions]:
                            # Process image
                            image = self.convert_image_to_rgb(filepath)
                            
                            # Detector already loaded at start of thread
                            face_locations = detector.detect_faces(image)
                            
                            if not face_locations:
                                error_count += 1
                                error_files.append(f"{filepath.name} (no face detected)")
                                continue
                            
                            face_encodings = face_recognition.face_encodings(
                                image, face_locations
                            )
                            
                            if not face_encodings:
                                error_count += 1
                                error_files.append(f"{filepath.name} (encoding failed)")
                                continue
                            
                            for encoding in face_encodings:
                                names.append(name)
                                encodings.append(encoding)
                            
                            processed_count += 1
                            new_count += 1
                            # Mark file as processed
                            if model_name not in self.processed_files:
                                self.processed_files[model_name] = set()
                            self.processed_files[model_name].add(file_key)
                        
                        elif filepath.suffix.lower() in [ext.lower() for ext in video_extensions]:
                            # Process video - extract frames
                            # Detector already loaded at start of thread
                            frame_count = 0
                            frames_processed = 0
                            
                            cap = cv2.VideoCapture(str(filepath))
                            if not cap.isOpened():
                                error_count += 1
                                error_files.append(f"{filepath.name} (could not open)")
                                continue
                            
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_interval = max(1, int(fps / 2))  # Extract 2 frames per second
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                
                                if frame_count % frame_interval == 0:
                                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    
                                    face_locations = detector.detect_faces(rgb_frame)
                                    
                                    if face_locations:
                                        face_encodings = face_recognition.face_encodings(
                                            rgb_frame, face_locations
                                        )
                                        
                                        for encoding in face_encodings:
                                            names.append(name)
                                            encodings.append(encoding)
                                        frames_processed += 1
                                
                                frame_count += 1
                            
                            cap.release()
                            
                            if frames_processed > 0:
                                processed_count += 1
                                new_count += 1
                                # Mark file as processed
                                if model_name not in self.processed_files:
                                    self.processed_files[model_name] = set()
                                self.processed_files[model_name].add(file_key)
                            else:
                                error_count += 1
                                error_files.append(f"{filepath.name} (no faces in video)")
                    
                    except Exception as e:
                        error_count += 1
                        error_files.append(f"{filepath.name}: {str(e)}")
                        print(f"Error processing {filepath}: {e}")
                
                if not names:
                    error_msg = f"No faces found in any training images!\n\n"
                    error_msg += f"Processed: {processed_count} files\n"
                    error_msg += f"Errors: {error_count} files\n\n"
                    if error_files:
                        error_msg += "Problem files:\n" + "\n".join(error_files[:5])
                        if len(error_files) > 5:
                            error_msg += f"\n... and {len(error_files) - 5} more"
                    
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
                    self.root.after(0, lambda: self.training_status.config(
                        text="Training failed - No faces found!",
                        fg=COLORS["error"]
                    ))
                    return
                
                name_encodings = {"names": names, "encodings": encodings}
                # Save to model-specific file
                encodings_path = ENCODINGS_PATHS[model_name]
                with encodings_path.open(mode="wb") as f:
                    pickle.dump(name_encodings, f)
                
                # Update loaded encodings
                self.loaded_encodings[model_name] = name_encodings
                
                # Save processed files list
                self.save_processed_files(model_name)
                
                num_people = len(set(names))
                success_msg = f"Model trained successfully!\n\n"
                if incremental and new_count > 0:
                    success_msg += f"✓ {new_count} new file(s) processed\n"
                    if skipped_count > 0:
                        success_msg += f"✓ {skipped_count} file(s) skipped (already trained)\n"
                success_msg += f"✓ {len(names)} total face encoding(s)\n"
                success_msg += f"✓ {num_people} person(s) registered\n"
                success_msg += f"✓ {processed_count} file(s) processed successfully"
                
                if error_count > 0:
                    success_msg += f"\n⚠ {error_count} file(s) had issues"
                
                self.root.after(0, lambda msg=success_msg: messagebox.showinfo("Success", msg))
                self.root.after(0, lambda: self.training_status.config(
                    text=f"✓ Training complete! {num_people} person(s), {len(names)} encoding(s).",
                    fg=COLORS["success"]
                ))
                # Reload encodings for current model
                self.load_all_encodings()
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Training error: {error_details}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", 
                    f"Training failed: {str(e)}\n\nCheck console for details."
                ))
                self.root.after(0, lambda: self.training_status.config(
                    text="Training failed!",
                    fg=COLORS["error"]
                ))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def recognize_face_in_frame(self, face_encoding):
        """Compare face encoding with known encodings using improved distance-based matching."""
        current_encodings = self.get_current_encodings()
        if not current_encodings:
            return None
        
        # Use face_distance for more accurate matching
        face_distances = face_recognition.face_distance(
            current_encodings["encodings"], face_encoding
        )
        
        # Find the best match (lowest distance)
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]
        
        # Use a stricter threshold for better accuracy
        # Lower distance = better match (0.0 = identical, 1.0 = very different)
        threshold = 0.45  # Even stricter for better accuracy
        
        if best_distance <= threshold:
            # Get all matches within threshold and use weighted voting
            matches = face_distances <= threshold
            if matches.sum() > 0:
                # Weight votes by inverse distance (closer = more weight)
                weighted_votes = {}
                for i in range(len(current_encodings["names"])):
                    if matches[i]:
                        name = current_encodings["names"][i]
                        distance = face_distances[i]
                        # Weight: closer faces get higher weight
                        weight = 1.0 / (distance + 0.1)  # Add small value to avoid division by zero
                        if name not in weighted_votes:
                            weighted_votes[name] = 0
                        weighted_votes[name] += weight
                
                if weighted_votes:
                    # Return person with highest weighted vote
                    return max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        return None
    
    def start_live_recognition(self):
        """Start live camera recognition."""
        if not self.get_current_encodings():
            messagebox.showerror(
                "Error", "No trained model found! Please train the model first."
            )
            return
        
        camera_window = tk.Toplevel(self.root)
        camera_window.title("Live Face Recognition")
        camera_window.geometry("1400x700")
        camera_window.configure(bg=COLORS["bg_primary"])
        
        # Main container with video and transcript side by side
        main_container = tk.Frame(camera_window, bg=COLORS["bg_primary"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame (left side)
        video_frame = tk.Frame(main_container, bg=COLORS["bg_secondary"])
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Video label container (for overlay button)
        video_container = tk.Frame(video_frame, bg=COLORS["bg_secondary"])
        video_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video label
        video_label = tk.Label(video_container, bg=COLORS["bg_secondary"])
        video_label.pack(fill=tk.BOTH, expand=True)
        
        # Audio toggle button OVERLAY on video (top-right corner)
        self.audio_toggle_btn_video = tk.Button(
            video_container,
            text="🎤 Audio: OFF",
            bg=COLORS["warning"],
            fg=COLORS["text_primary"],
            activebackground=COLORS["success"],
            activeforeground=COLORS["text_primary"],
            command=lambda: self.toggle_audio(),
            font=("Segoe UI", 14, "bold"),
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            padx=20,
            pady=12
        )
        # Position button in top-right corner of video
        self.audio_toggle_btn_video.place(relx=0.98, rely=0.02, anchor=tk.NE)
        
        # Transcript frame (right side)
        transcript_frame = tk.Frame(main_container, bg=COLORS["bg_secondary"], width=400)
        transcript_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        transcript_frame.pack_propagate(False)
        
        # Transcript header
        transcript_header = tk.Frame(transcript_frame, bg=COLORS["bg_tertiary"], height=50)
        transcript_header.pack(fill=tk.X)
        transcript_header.pack_propagate(False)
        
        tk.Label(
            transcript_header,
            text="🎤 Live Transcription",
            font=("Segoe UI", 14, "bold"),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            pady=15
        ).pack()
        
        # Transcript text widget with scrollbar
        transcript_scroll = tk.Scrollbar(transcript_frame)
        transcript_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.transcript_text = tk.Text(
            transcript_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            relief=tk.FLAT,
            padx=10,
            pady=10,
            yscrollcommand=transcript_scroll.set,
            state=tk.DISABLED
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        transcript_scroll.config(command=self.transcript_text.yview)
        
        # Initial message in transcript panel
        self.transcript_text.config(state=tk.NORMAL)
        self.transcript_text.insert("1.0", "🎤 Speech-to-Text Ready\n\n")
        self.transcript_text.insert(tk.END, "Click the '🎤 Audio: OFF' button below to start recording.\n\n")
        self.transcript_text.insert(tk.END, "Your speech will be transcribed here in real-time.\n\n")
        if self.gemini_api_key.get().strip():
            self.transcript_text.insert(tk.END, "✓ Gemini API key is set - AI responses will appear here.\n")
        else:
            self.transcript_text.insert(tk.END, "⚠️ No Gemini API key - Only transcription will be shown.\n")
            self.transcript_text.insert(tk.END, "   Set your API key in Settings to enable AI chat.\n")
        self.transcript_text.config(state=tk.DISABLED)
        
        # Status label at bottom of transcript
        self.transcript_status = tk.Label(
            transcript_frame,
            text="🎤 Audio: OFF - Click '🎤 Audio: OFF' button to start recording",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["warning"],
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.transcript_status.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Control frame
        control_frame = tk.Frame(camera_window, bg=COLORS["bg_secondary"], height=60)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        status_label = tk.Label(
            control_frame,
            text="📹 Camera Active - Press 'Stop' to close",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        )
        status_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Camera controls frame
        camera_controls_frame = tk.Frame(control_frame, bg=COLORS["bg_secondary"])
        camera_controls_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Flip Horizontal button
        flip_h_btn = tk.Button(
            camera_controls_frame,
            text="↔️ Flip H",
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            activebackground=COLORS["accent_blue"],
            command=lambda: self.camera_flip_horizontal.set(not self.camera_flip_horizontal.get()),
            font=("Segoe UI", 9, "bold"),
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=5
        )
        flip_h_btn.pack(side=tk.LEFT, padx=5)
        
        # Flip Vertical button
        flip_v_btn = tk.Button(
            camera_controls_frame,
            text="↕️ Flip V",
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            activebackground=COLORS["accent_blue"],
            command=lambda: self.camera_flip_vertical.set(not self.camera_flip_vertical.get()),
            font=("Segoe UI", 9, "bold"),
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=5
        )
        flip_v_btn.pack(side=tk.LEFT, padx=5)
        
        # Rotate button
        rotate_btn = tk.Button(
            camera_controls_frame,
            text="🔄 Rotate",
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            activebackground=COLORS["accent_blue"],
            command=self.rotate_camera,
            font=("Segoe UI", 9, "bold"),
            relief=tk.RAISED,
            bd=2,
            padx=10,
            pady=5
        )
        rotate_btn.pack(side=tk.LEFT, padx=5)
        
        # Audio toggle button in control bar (using regular Button for better visibility)
        self.audio_toggle_btn = tk.Button(
            control_frame,
            text="🎤 Audio: OFF",
            bg=COLORS["warning"],
            fg=COLORS["text_primary"],
            activebackground=COLORS["success"],
            activeforeground=COLORS["text_primary"],
            command=lambda: self.toggle_audio(),
            font=("Segoe UI", 14, "bold"),
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            padx=25,
            pady=12
        )
        self.audio_toggle_btn.pack(side=tk.RIGHT, padx=15, pady=12)
        
        stop_btn = ModernButton(
            control_frame,
            text="Stop",
            bg=COLORS["error"],
            fg=COLORS["text_primary"],
            command=lambda: self.stop_camera(camera_window),
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=8
        )
        stop_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        
        self.camera_running = True
        self.video_capture = cv2.VideoCapture(self.camera_index.get())
        
        if not self.video_capture.isOpened():
            messagebox.showerror("Error", f"Could not open camera {self.camera_index.get()}")
            camera_window.destroy()
            return
        
        # Performance optimization variables
        process_frame_count = 0
        face_locations_cache = []
        face_names_cache = []
        analysis_cache = {}  # Store analysis for each recognized face
        
        # Audio recording variables
        audio_buffer = []
        last_audio_process_time = 0
        audio_process_interval = 3.0  # Process audio every 3 seconds
        
        def update_frame():
            nonlocal process_frame_count, face_locations_cache, face_names_cache, analysis_cache
            
            if not self.camera_running:
                return
            
            ret, frame = self.video_capture.read()
            if ret:
                # Apply camera transformations (flip/rotate)
                if self.camera_flip_horizontal.get():
                    frame = cv2.flip(frame, 1)  # Horizontal flip
                if self.camera_flip_vertical.get():
                    frame = cv2.flip(frame, 0)  # Vertical flip
                
                # Apply rotation
                rotation = self.camera_rotate.get()
                if rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Only process every 3rd frame for better performance
                process_frame_count += 1
                should_process = (process_frame_count % 3 == 0)
                
                if should_process:
                    # Resize for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    detector = self.get_detector()
                    face_locations_cache = detector.detect_faces(rgb_small_frame)
                    
                    # Get encodings
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations_cache
                    )
                    
                    # Recognize faces and get analysis
                    face_names_cache = []
                    analysis_cache = {}  # Reset analysis cache
                    
                    # Get DeepFace analyzer if using DeepFace model
                    deepface_analyzer = None
                    if self.detection_model.get() == "deepface":
                        try:
                            deepface_analyzer = self.get_detector()
                        except:
                            pass
                    
                    for i, face_encoding in enumerate(face_encodings):
                        name = self.recognize_face_in_frame(face_encoding)
                        name = name if name else "Unknown"
                        face_names_cache.append(name)
                        
                        # Get DeepFace analysis for recognized faces (less frequently for performance)
                        if deepface_analyzer and name != "Unknown" and (process_frame_count % 9 == 0):
                            try:
                                # Scale back to full frame size for analysis
                                scale_factor = 1 / 0.4
                                if i < len(face_locations_cache):
                                    top, right, bottom, left = face_locations_cache[i]
                                    top = int(top * scale_factor)
                                    right = int(right * scale_factor)
                                    bottom = int(bottom * scale_factor)
                                    left = int(left * scale_factor)
                                    
                                    # Extract face region
                                    face_roi = frame[max(0, top):min(frame.shape[0], bottom), 
                                                   max(0, left):min(frame.shape[1], right)]
                                    if face_roi.size > 0:
                                        import tempfile
                                        import os
                                        temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
                                        os.close(temp_fd)
                                        cv2.imwrite(temp_path, face_roi)
                                        
                                        analysis = deepface_analyzer.analyze_face(
                                            temp_path,
                                            actions=['emotion', 'age', 'gender', 'race']
                                        )
                                        
                                        if analysis:
                                            analysis_cache[name] = {
                                                'emotion': analysis.get('dominant_emotion', 'N/A'),
                                                'age': int(analysis.get('age', 0)),
                                                'gender': analysis.get('dominant_gender', 'N/A'),
                                                'race': analysis.get('dominant_race', 'N/A')
                                            }
                                        
                                        os.remove(temp_path)
                            except Exception as e:
                                pass
                
                # Draw on full-size frame using cached results
                scale_factor = 1 / 0.4  # Inverse of resize factor
                
                # Draw face bounding boxes and names
                for (top, right, bottom, left), name in zip(face_locations_cache, face_names_cache):
                    # Scale back to full frame size
                    top = int(top * scale_factor)
                    right = int(right * scale_factor)
                    bottom = int(bottom * scale_factor)
                    left = int(left * scale_factor)
                    
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                    
                    # Draw name label (simpler, analysis shown in overlay)
                    display_text = name
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
                    text_height = text_size[1] + 10
                    
                    # Draw background for name
                    cv2.rectangle(
                        frame, (left, bottom - text_height), (right, bottom), color, cv2.FILLED
                    )
                    
                    # Draw name
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(
                        frame, display_text, (left + 6, bottom - 10),
                        font, 0.7, (255, 255, 255), 2
                    )
                
                # Draw analysis overlay in top-left corner (if using DeepFace and we have analysis)
                if self.detection_model.get() == "deepface" and analysis_cache:
                    # Calculate overlay size based on number of people
                    num_people = len(analysis_cache)
                    overlay_y = 10
                    overlay_x = 10
                    overlay_width = 320
                    overlay_height = min(200, 50 + num_people * 80)  # Dynamic height
                    
                    # Semi-transparent background
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (overlay_x, overlay_y), 
                                 (overlay_x + overlay_width, overlay_y + overlay_height), 
                                 (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                    
                    # Draw title with border
                    title = "DeepFace Analysis"
                    cv2.putText(frame, title, (overlay_x + 10, overlay_y + 28),
                              cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, title, (overlay_x + 10, overlay_y + 28),
                              cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                    
                    # Draw analysis for each recognized person
                    y_offset = overlay_y + 55
                    line_height = 22
                    for name, analysis_data in analysis_cache.items():
                        if y_offset + line_height * 4 > overlay_y + overlay_height - 10:
                            break  # Don't overflow overlay
                        
                        # Person name (highlighted)
                        cv2.putText(frame, f"Person: {name}", 
                                  (overlay_x + 10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += line_height
                        
                        # Emotion
                        emotion = analysis_data.get('emotion', 'N/A')
                        cv2.putText(frame, f"  Emotion: {emotion}", 
                                  (overlay_x + 10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += line_height
                        
                        # Age and Gender
                        age = analysis_data.get('age', 0)
                        gender = analysis_data.get('gender', 'N/A')
                        cv2.putText(frame, f"  Age: {age}y | Gender: {gender}", 
                                  (overlay_x + 10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += line_height
                        
                        # Race
                        race = analysis_data.get('race', 'N/A')
                        cv2.putText(frame, f"  Race: {race}", 
                                  (overlay_x + 10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += line_height + 8  # Extra space between people
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((880, 660), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                video_label.imgtk = imgtk
                video_label.config(image=imgtk)
            
            if self.camera_running:
                camera_window.after(33, update_frame)  # ~30 FPS for better performance
        
        # Start audio recording if enabled
        if self.audio_enabled:
            self.start_audio_recording()
        
        update_frame()
    
    def rotate_camera(self):
        """Rotate camera by 90 degrees (cycles: 0 -> 90 -> 180 -> 270 -> 0)."""
        current = self.camera_rotate.get()
        next_rotation = (current + 90) % 360
        self.camera_rotate.set(next_rotation)
        print(f"Camera rotated to {next_rotation} degrees")
    
    def stop_camera(self, window):
        """Stop the camera and close the window."""
        self.camera_running = False
        if self.video_capture:
            self.video_capture.release()
        self.audio_enabled = False
        self.stop_audio_recording()
        window.destroy()
    
    def toggle_audio(self):
        """Toggle audio recording on/off."""
        self.audio_enabled = not self.audio_enabled
        
        if self.audio_enabled:
            # Update both buttons
            if hasattr(self, 'audio_toggle_btn'):
                self.audio_toggle_btn.config(text="🎤 Audio: ON", bg=COLORS["success"], fg=COLORS["text_primary"])
            if hasattr(self, 'audio_toggle_btn_video'):
                self.audio_toggle_btn_video.config(text="🎤 Audio: ON", bg=COLORS["success"], fg=COLORS["text_primary"])
            if hasattr(self, 'transcript_status'):
                self.transcript_status.config(
                    text="🎤 Audio: ON - Listening... Speak now!",
                    fg=COLORS["success"]
                )
            self.start_audio_recording()
        else:
            # Update both buttons
            if hasattr(self, 'audio_toggle_btn'):
                self.audio_toggle_btn.config(text="🎤 Audio: OFF", bg=COLORS["warning"], fg=COLORS["text_primary"])
            if hasattr(self, 'audio_toggle_btn_video'):
                self.audio_toggle_btn_video.config(text="🎤 Audio: OFF", bg=COLORS["warning"], fg=COLORS["text_primary"])
            if hasattr(self, 'transcript_status'):
                self.transcript_status.config(
                    text="🎤 Audio: OFF - Click '🎤 Audio: OFF' button to start recording",
                    fg=COLORS["warning"]
                )
            self.stop_audio_recording()
    
    def start_audio_recording(self):
        """Start recording audio from microphone."""
        if self.audio_recording:
            return
        
        if pyaudio is None:
            messagebox.showerror(
                "Audio Error",
                "PyAudio is not installed.\n\n"
                "Please install it:\n"
                "pip install pyaudio\n\n"
                "Note: On Windows, you may need to install it via:\n"
                "pip install pipwin\n"
                "pipwin install pyaudio"
            )
            self.audio_enabled = False
            if hasattr(self, 'audio_toggle_btn'):
                self.audio_toggle_btn.config(text="🎤 Audio: OFF", bg=COLORS["warning"])
            if hasattr(self, 'audio_toggle_btn_video'):
                self.audio_toggle_btn_video.config(text="🎤 Audio: OFF", bg=COLORS["warning"])
            return
        
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.audio_format,
                channels=self.audio_channels,
                rate=self.audio_sample_rate,
                input=True,
                frames_per_buffer=self.audio_chunk_size
            )
            self.audio_recording = True
            self.audio_frames = []
            
            # Clear transcript when starting and show ready message
            if hasattr(self, 'transcript_text'):
                self.transcript_text.config(state=tk.NORMAL)
                self.transcript_text.delete("1.0", tk.END)
                self.transcript_text.insert("1.0", "🎤 Audio Recording Started!\n\n")
                self.transcript_text.insert(tk.END, "Listening... Speak clearly into your microphone.\n\n")
                self.transcript_text.insert(tk.END, "Transcription will appear here every 3 seconds...\n\n")
                self.transcript_text.see(tk.END)
                self.transcript_text.config(state=tk.DISABLED)
            
            # Start audio processing thread
            threading.Thread(target=self.process_audio_loop, daemon=True).start()
            print("✓ Audio recording started")
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            messagebox.showerror("Audio Error", f"Failed to start audio recording:\n{str(e)}")
            self.audio_enabled = False
            if hasattr(self, 'audio_toggle_btn'):
                self.audio_toggle_btn.config(text="🎤 Audio: OFF", bg=COLORS["warning"])
            if hasattr(self, 'audio_toggle_btn_video'):
                self.audio_toggle_btn_video.config(text="🎤 Audio: OFF", bg=COLORS["warning"])
            if hasattr(self, 'transcript_status'):
                self.transcript_status.config(text=f"❌ Error: {str(e)}")
    
    def stop_audio_recording(self):
        """Stop recording audio."""
        self.audio_recording = False
        
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass
            self.pyaudio_instance = None
        
        self.audio_frames = []
        print("Audio recording stopped")
    
    def process_audio_loop(self):
        """Continuously record and process audio."""
        import time
        
        while self.audio_recording and self.audio_enabled:
            try:
                if self.audio_stream:
                    # Read audio data
                    data = self.audio_stream.read(self.audio_chunk_size, exception_on_overflow=False)
                    self.audio_frames.append(data)
                    
                    # Process audio every 4 seconds for better accuracy (longer context)
                    if len(self.audio_frames) >= (self.audio_sample_rate * 4) // self.audio_chunk_size:
                        self.process_audio_chunk()
                        self.audio_frames = []  # Clear buffer after processing
                
                time.sleep(0.01)  # Small delay to prevent CPU overload
            except Exception as e:
                print(f"Error in audio loop: {e}")
                break
    
    def process_audio_chunk(self):
        """Process recorded audio chunk: transcribe and send to Gemini."""
        if not self.audio_frames:
            return
        
        try:
            # Convert audio frames to numpy array
            audio_data = b''.join(self.audio_frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Normalize audio
            if audio_np.max() > 0:
                audio_np = audio_np / 32768.0
            
            # Transcribe audio with better accuracy settings
            print("Transcribing audio...")
            transcribed_text = transcribe_audio(audio_np, self.audio_sample_rate)
            
            if transcribed_text and len(transcribed_text.strip()) > 0:
                print(f"Transcribed: {transcribed_text}")
                
                # Update transcript display in UI (ensure it shows in panel)
                self.root.after(0, lambda text=transcribed_text: self.update_transcript(text))
                
                # Send to Gemini API if key is set
                gemini_key = self.gemini_api_key.get().strip()
                if gemini_key:
                    self.root.after(0, lambda: self.send_to_gemini(transcribed_text, gemini_key))
                else:
                    # Just show transcribed text if no API key
                    print("Gemini API key not set, showing transcription only")
            else:
                print("No speech detected in audio chunk")
                # Update status to show listening
                self.root.after(0, lambda: self.update_transcript_status("Listening... (no speech detected)"))
                
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            self.root.after(0, lambda: self.update_transcript_status(f"Error: {str(e)}"))
    
    def update_transcript(self, text):
        """Update the transcript display with new transcribed text."""
        if not hasattr(self, 'transcript_text'):
            return
        
        try:
            self.transcript_text.config(state=tk.NORMAL)
            
            # Add timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Add new transcription with user tag
            self.transcript_text.insert(tk.END, f"[{timestamp}] You: {text}\n", "user_tag")
            self.transcript_text.insert(tk.END, "\n")
            
            # Configure user tag styling
            self.transcript_text.tag_config("user_tag", foreground=COLORS["accent_blue"], font=("Segoe UI", 10, "bold"))
            
            # Auto-scroll to bottom
            self.transcript_text.see(tk.END)
            
            self.transcript_text.config(state=tk.DISABLED)
            
            # Force update to ensure it's visible
            self.root.update_idletasks()
            
            # Update status with more visible feedback
            gemini_key = self.gemini_api_key.get().strip()
            if gemini_key:
                self.update_transcript_status("✓ Transcribed! Sending to Gemini...")
            else:
                self.update_transcript_status("✓ Transcribed! (No Gemini API key - transcription only)")
                
        except Exception as e:
            print(f"Error updating transcript: {e}")
    
    def update_transcript_status(self, message):
        """Update the transcript status label."""
        if hasattr(self, 'transcript_status'):
            self.transcript_status.config(text=message)
    
    def send_to_gemini(self, text, api_key):
        """Send transcribed text to Gemini API and show response."""
        def gemini_thread():
            try:
                # Update status
                self.root.after(0, lambda: self.update_transcript_status("🔄 Getting Gemini response..."))
                
                response = get_gemini_response(text, api_key)
                
                # Update transcript with Gemini response
                self.root.after(0, lambda: self.add_gemini_response_to_transcript(response))
                
                # Also show popup
                self.root.after(0, lambda: self.show_gemini_response(text, response))
                
                # Update status
                self.root.after(0, lambda: self.update_transcript_status("✓ Gemini response received! Listening for more..."))
            except Exception as e:
                error_msg = f"Error getting Gemini response: {str(e)}"
                self.root.after(0, lambda: self.add_gemini_response_to_transcript(f"❌ Error: {error_msg}"))
                self.root.after(0, lambda: self.show_gemini_response(text, error_msg))
                self.root.after(0, lambda: self.update_transcript_status(f"❌ Error: {error_msg}"))
        
        threading.Thread(target=gemini_thread, daemon=True).start()
    
    def add_gemini_response_to_transcript(self, response):
        """Add Gemini response to the transcript display."""
        if not hasattr(self, 'transcript_text'):
            return
        
        try:
            self.transcript_text.config(state=tk.NORMAL)
            
            # Add Gemini response with formatting
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            self.transcript_text.insert(tk.END, f"[{timestamp}] 🤖 Gemini: {response}\n\n")
            self.transcript_text.insert(tk.END, "-" * 50 + "\n\n")
            
            # Auto-scroll to bottom
            self.transcript_text.see(tk.END)
            
            self.transcript_text.config(state=tk.DISABLED)
                
        except Exception as e:
            print(f"Error adding Gemini response to transcript: {e}")
    
    def show_gemini_response(self, user_text, gemini_response):
        """Show Gemini response in a popup window."""
        response_window = tk.Toplevel(self.root)
        response_window.title("Gemini Response")
        response_window.geometry("600x500")
        response_window.configure(bg=COLORS["bg_primary"])
        
        # Header
        header = tk.Frame(response_window, bg=COLORS["bg_secondary"], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="🤖 Gemini Response",
            font=("Segoe UI", 16, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            pady=15
        ).pack()
        
        # Content frame
        content = tk.Frame(response_window, bg=COLORS["bg_primary"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # User message
        tk.Label(
            content,
            text="You said:",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["bg_primary"],
            fg=COLORS["text_secondary"],
            anchor=tk.W
        ).pack(fill=tk.X, pady=(0, 5))
        
        user_frame = tk.Frame(content, bg=COLORS["bg_secondary"], relief=tk.FLAT)
        user_frame.pack(fill=tk.X, pady=(0, 15))
        
        user_text_widget = tk.Text(
            user_frame,
            height=3,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        user_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        user_text_widget.insert("1.0", user_text)
        user_text_widget.config(state=tk.DISABLED)
        
        # Gemini response
        tk.Label(
            content,
            text="Gemini replied:",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["bg_primary"],
            fg=COLORS["text_secondary"],
            anchor=tk.W
        ).pack(fill=tk.X, pady=(0, 5))
        
        response_frame = tk.Frame(content, bg=COLORS["bg_secondary"], relief=tk.FLAT)
        response_frame.pack(fill=tk.BOTH, expand=True)
        
        response_text_widget = tk.Text(
            response_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        response_text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        response_text_widget.insert("1.0", gemini_response)
        response_text_widget.config(state=tk.DISABLED)
        
        # Close button
        close_btn = ModernButton(
            content,
            text="Close",
            bg=COLORS["accent_blue"],
            fg=COLORS["text_primary"],
            command=response_window.destroy,
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=8
        )
        close_btn.pack(pady=(15, 0))
    
    def test_image(self):
        """Test recognition on a single image or video."""
        if not self.get_current_encodings():
            messagebox.showerror(
                "Error", "No trained model found! Please train the model first."
            )
            return
        
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Image or Video to Test",
            filetypes=filetypes
        )
        
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
            
            if file_ext in video_extensions:
                # Process video
                self.test_video(file_path)
            else:
                # Process image
                try:
                    image = self.convert_image_to_rgb(file_path)
                    
                    detector = self.get_detector()
                    face_locations = detector.detect_faces(image)
                    
                    face_encodings = face_recognition.face_encodings(
                        image, face_locations
                    )
                    
                    from PIL import ImageDraw, ImageFont
                    pillow_image = Image.fromarray(image)
                    draw = ImageDraw.Draw(pillow_image)
                    
                    try:
                        font_size = max(20, int(image.shape[0] / 30))
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()
                    
                    # Get DeepFace analyzer if using DeepFace model
                    deepface_analyzer = None
                    if self.detection_model.get() == "deepface":
                        try:
                            deepface_analyzer = self.get_detector()
                        except:
                            pass
                    
                    for bounding_box, unknown_encoding in zip(face_locations, face_encodings):
                        name = self.recognize_face_in_frame(unknown_encoding)
                        if not name:
                            name = "Unknown"
                        
                        top, right, bottom, left = bounding_box
                        color = "blue" if name != "Unknown" else "red"
                        
                        draw.rectangle(((left, top), (right, bottom)), outline=color, width=4)
                        
                        # Prepare display text
                        display_text = name
                        
                        # Add DeepFace analysis if available
                        if deepface_analyzer and name != "Unknown":
                            try:
                                # Extract face region
                                face_roi = image[top:bottom, left:right]
                                if face_roi.size > 0:
                                    import tempfile
                                    import os
                                    temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
                                    os.close(temp_fd)
                                    Image.fromarray(face_roi).save(temp_path, 'JPEG')
                                    
                                    analysis = deepface_analyzer.analyze_face(
                                        temp_path,
                                        actions=['emotion', 'age', 'gender', 'race']
                                    )
                                    
                                    if analysis:
                                        emotion = analysis.get('dominant_emotion', 'N/A')
                                        age = int(analysis.get('age', 0))
                                        gender = analysis.get('dominant_gender', 'N/A')
                                        race = analysis.get('dominant_race', 'N/A')
                                        
                                        display_text = f"{name}\n{emotion} | {age}y | {gender} | {race}"
                                    
                                    os.remove(temp_path)
                            except:
                                pass
                        
                        text_y = max(0, top - 30)
                        
                        try:
                            bbox = draw.textbbox((left, text_y), display_text.split('\n')[0], font=font)
                            text_left, text_top, text_right, text_bottom = bbox
                        except:
                            text_left, text_top, text_right, text_bottom = draw.textbbox(
                                (left, text_y), display_text.split('\n')[0]
                            )
                        
                        # Calculate height for multi-line text
                        lines = display_text.split('\n')
                        line_height = 25
                        text_height = len(lines) * line_height
                        
                        padding = 5
                        draw.rectangle(
                            ((text_left - padding, text_top - padding), 
                             (text_right + padding, text_top + text_height + padding)),
                            fill=color,
                            outline=color,
                        )
                        
                        # Draw text (handle multi-line)
                        current_y = text_top
                        for line in display_text.split('\n'):
                            draw.text(
                                (text_left, current_y),
                                line,
                                fill="white",
                                font=font,
                            )
                            current_y += line_height
                    
                    self.show_result_image(pillow_image, file_path)
                
                except Exception as e:
                    import traceback
                    error_msg = f"Failed to process image: {str(e)}\n\n"
                    error_msg += "Make sure:\n"
                    error_msg += "1. The image contains clear faces\n"
                    error_msg += "2. You have trained the model first\n"
                    error_msg += "3. The image format is supported (JPG, PNG, etc.)"
                    messagebox.showerror("Error", error_msg)
                    print(f"Full error: {traceback.format_exc()}")
    
    def test_video(self, video_path):
        """Test recognition on a video file."""
        # Create video processing window
        video_window = tk.Toplevel(self.root)
        video_window.title("Video Recognition")
        video_window.geometry("1000x700")
        video_window.configure(bg=COLORS["bg_primary"])
        
        # Video display
        video_label = tk.Label(video_window, bg=COLORS["bg_secondary"])
        video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = tk.Frame(video_window, bg=COLORS["bg_secondary"], height=60)
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        status_label = tk.Label(
            control_frame,
            text="🎬 Processing Video...",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        )
        status_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        stop_btn = ModernButton(
            control_frame,
            text="Stop",
            bg=COLORS["error"],
            fg=COLORS["text_primary"],
            command=lambda: self.stop_video_processing(video_window),
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=8
        )
        stop_btn.pack(side=tk.RIGHT, padx=20, pady=15)
        
        self.video_processing = True
        
        def process_video_frames():
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    messagebox.showerror("Error", f"Could not open video: {video_path}")
                    video_window.destroy()
                    return
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = int(1000 / fps) if fps > 0 else 33
                
                detector = self.get_detector()
                
                def process_next_frame():
                    if not self.video_processing:
                        cap.release()
                        video_window.destroy()
                        return
                    
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        status_label.config(text="✓ Video processing complete")
                        return
                    
                    # Resize for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect and recognize faces
                    face_locations = detector.detect_faces(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        name = self.recognize_face_in_frame(face_encoding)
                        face_names.append(name if name else "Unknown")
                    
                    # Draw on full frame
                    scale_factor = 2
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top = int(top * scale_factor)
                        right = int(right * scale_factor)
                        bottom = int(bottom * scale_factor)
                        left = int(left * scale_factor)
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                        
                        cv2.rectangle(
                            frame, (left, bottom - 40), (right, bottom), color, cv2.FILLED
                        )
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(
                            frame, name, (left + 6, bottom - 10),
                            font, 0.7, (255, 255, 255), 2
                        )
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((980, 600), Image.Resampling.LANCZOS)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    video_label.imgtk = imgtk
                    video_label.config(image=imgtk)
                    
                    # Schedule next frame
                    video_window.after(frame_delay, process_next_frame)
                
                process_next_frame()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process video: {str(e)}")
                video_window.destroy()
        
        threading.Thread(target=process_video_frames, daemon=True).start()
    
    def stop_video_processing(self, window):
        """Stop video processing."""
        self.video_processing = False
        window.destroy()
    
    def show_result_image(self, image, original_path):
        """Display result image in a modern Tkinter window."""
        result_window = tk.Toplevel(self.root)
        result_window.title("Face Recognition Result")
        result_window.configure(bg=COLORS["bg_primary"])
        
        img_width, img_height = image.size
        max_width = self.root.winfo_screenwidth() - 100
        max_height = self.root.winfo_screenheight() - 150
        
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        
        img_label = tk.Label(result_window, image=photo, bg=COLORS["bg_primary"])
        img_label.image = photo
        img_label.pack(padx=10, pady=10)
        
        info_frame = tk.Frame(result_window, bg=COLORS["bg_secondary"])
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        file_name = os.path.basename(original_path)
        info_label = tk.Label(
            info_frame,
            text=f"📄 {file_name}",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        )
        info_label.pack(pady=10)
        
        button_frame = tk.Frame(result_window, bg=COLORS["bg_primary"])
        button_frame.pack(pady=10)
        
        save_btn = ModernButton(
            button_frame,
            text="💾 Save Result",
            bg=COLORS["accent_blue"],
            fg=COLORS["text_primary"],
            command=lambda: self.save_result_image(image, original_path),
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8
        )
        save_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ModernButton(
            button_frame,
            text="Close",
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            command=result_window.destroy,
            font=("Segoe UI", 10, "bold"),
            padx=15,
            pady=8
        )
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def save_result_image(self, image, original_path):
        """Save the result image with face recognition labels."""
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        default_name = os.path.splitext(os.path.basename(original_path))[0] + "_recognized"
        save_path = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=filetypes
        )
        
        if save_path:
            try:
                image.save(save_path)
                messagebox.showinfo("Success", f"Result saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def view_registered_people(self):
        """Show a window with list of registered people."""
        window = tk.Toplevel(self.root)
        window.title("Registered People")
        window.geometry("500x600")
        window.configure(bg=COLORS["bg_primary"])
        
        header = tk.Frame(window, bg=COLORS["bg_secondary"], height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="👥 Registered People",
            font=("Segoe UI", 20, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            pady=25
        ).pack()
        
        content = tk.Frame(window, bg=COLORS["bg_primary"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        scrollbar = tk.Scrollbar(content)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(
            content,
            yscrollcommand=scrollbar.set,
            font=("Segoe UI", 11),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            selectbackground=COLORS["accent_blue"],
            selectforeground=COLORS["text_primary"],
            relief=tk.FLAT,
            bd=0,
            highlightthickness=0
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        current_encodings = self.get_current_encodings()
        if current_encodings:
            people = sorted(set(current_encodings["names"]))
            for person in people:
                count = current_encodings["names"].count(person)
                listbox.insert(tk.END, f"👤 {person} ({count} encoding(s))")
        else:
            listbox.insert(tk.END, "No registered people")
        
        close_btn = ModernButton(
            window,
            text="Close",
            command=window.destroy,
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=8
        )
        close_btn.pack(pady=15)
    
    def show_settings(self):
        """Show settings window with modern dark theme."""
        window = tk.Toplevel(self.root)
        window.title("Settings")
        window.geometry("600x550")
        window.configure(bg=COLORS["bg_primary"])
        window.transient(self.root)  # Make it modal
        window.grab_set()  # Focus on this window
        
        header = tk.Frame(window, bg=COLORS["bg_secondary"], height=70)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="⚙️ Settings",
            font=("Segoe UI", 20, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            pady=20
        ).pack()
        
        content = tk.Frame(window, bg=COLORS["bg_primary"])
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Camera index
        camera_frame = tk.Frame(content, bg=COLORS["bg_secondary"], relief=tk.FLAT)
        camera_frame.pack(pady=15, padx=0, fill=tk.X)
        
        tk.Label(
            camera_frame,
            text="📹 Camera Index:",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(anchor=tk.W, padx=20, pady=(15, 5))
        
        spinbox_frame = tk.Frame(camera_frame, bg=COLORS["bg_secondary"])
        spinbox_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Label(
            spinbox_frame,
            text="Device:",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"]
        ).pack(side=tk.LEFT)
        
        camera_spinbox = tk.Spinbox(
            spinbox_frame,
            from_=0,
            to=5,
            textvariable=self.camera_index,
            width=10,
            font=("Segoe UI", 10),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            buttonbackground=COLORS["accent_blue"],
            relief=tk.FLAT
        )
        camera_spinbox.pack(side=tk.RIGHT)
        
        # Model type
        model_frame = tk.Frame(content, bg=COLORS["bg_secondary"], relief=tk.FLAT)
        model_frame.pack(pady=15, padx=0, fill=tk.X)
        
        tk.Label(
            model_frame,
            text="🤖 Detection Model:",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"]
        ).pack(anchor=tk.W, padx=20, pady=(15, 10))
        
        radio_frame = tk.Frame(model_frame, bg=COLORS["bg_secondary"])
        radio_frame.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        tk.Radiobutton(
            radio_frame,
            text="HOG (CPU - Faster)",
            variable=self.model_type,
            value="hog",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"]
        ).pack(anchor=tk.W, padx=10)
        
        tk.Radiobutton(
            radio_frame,
            text="CNN (GPU - More Accurate)",
            variable=self.model_type,
            value="cnn",
            font=("Segoe UI", 10),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_primary"],
            selectcolor=COLORS["bg_tertiary"],
            activebackground=COLORS["bg_secondary"],
            activeforeground=COLORS["text_primary"]
        ).pack(anchor=tk.W, padx=10)
        
        # Gemini API Key Section (More Prominent)
        gemini_frame = tk.Frame(content, bg=COLORS["bg_secondary"], relief=tk.FLAT, bd=2)
        gemini_frame.pack(pady=20, padx=0, fill=tk.X)
        
        # Header with icon
        gemini_header = tk.Frame(gemini_frame, bg=COLORS["accent_blue"], height=40)
        gemini_header.pack(fill=tk.X)
        gemini_header.pack_propagate(False)
        
        tk.Label(
            gemini_header,
            text="🤖 Gemini API Key (For Voice Chat)",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["accent_blue"],
            fg=COLORS["text_primary"]
        ).pack(pady=10)
        
        # Instructions
        tk.Label(
            gemini_frame,
            text="Get your free API key from: https://makersuite.google.com/app/apikey",
            font=("Segoe UI", 9),
            bg=COLORS["bg_secondary"],
            fg=COLORS["text_secondary"],
            cursor="hand2"
        ).pack(anchor=tk.W, padx=20, pady=(15, 5))
        
        # Entry field
        gemini_entry = tk.Entry(
            gemini_frame,
            textvariable=self.gemini_api_key,
            font=("Segoe UI", 11),
            bg=COLORS["bg_tertiary"],
            fg=COLORS["text_primary"],
            insertbackground=COLORS["text_primary"],
            relief=tk.FLAT,
            show="*",  # Hide API key
            width=50
        )
        gemini_entry.pack(fill=tk.X, padx=20, pady=(5, 10))
        
        # Status indicator
        api_status = tk.Label(
            gemini_frame,
            text="⚠️ API key not set - Voice chat will show transcription only",
            font=("Segoe UI", 9, "italic"),
            bg=COLORS["bg_secondary"],
            fg=COLORS["warning"]
        )
        api_status.pack(anchor=tk.W, padx=20, pady=(0, 15))
        
        # Update status based on current key
        if self.gemini_api_key.get().strip():
            api_status.config(
                text="✓ API key is set - Voice chat enabled",
                fg=COLORS["success"]
            )
        
        close_btn = ModernButton(
            content,
            text="💾 Save & Close",
            command=lambda: self.save_settings(window),
            bg=COLORS["accent_green"],
            fg=COLORS["text_primary"],
            font=("Segoe UI", 11, "bold"),
            padx=20,
            pady=10
        )
        close_btn.pack(pady=20)
    
    def load_gemini_api_key(self):
        """Load Gemini API key from file if exists."""
        try:
            key_file = Path("output/gemini_api_key.txt")
            if key_file.exists():
                with key_file.open("r") as f:
                    self.gemini_api_key.set(f.read().strip())
        except Exception as e:
            print(f"Error loading Gemini API key: {e}")
    
    def save_gemini_api_key(self):
        """Save Gemini API key to file."""
        try:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            key_file = output_dir / "gemini_api_key.txt"
            with key_file.open("w") as f:
                f.write(self.gemini_api_key.get().strip())
        except Exception as e:
            print(f"Error saving Gemini API key: {e}")
    
    def save_settings(self, window):
        """Save settings and close window."""
        # Save Gemini API key
        self.save_gemini_api_key()
        
        window.destroy()
        messagebox.showinfo("Success", "Settings saved!")


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
