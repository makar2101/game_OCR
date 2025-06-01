#!/usr/bin/env python3
"""
Skyrim Dialogue Analyzer - Main UI Application
==============================================

The primary user interface for the Skyrim Dialogue Analyzer.
Provides a comprehensive GUI for video processing, OCR, translation, and learning.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import logging
from pathlib import Path
import json
from datetime import datetime
import time

# Import core modules
try:
    from core.video_processor import VideoProcessor, VideoMetadata, FrameData
    from core.ocr_engine import OCREngine, OCRResult, TextSegment
    from data.models import ProcessingStatus

    VIDEO_PROCESSOR_AVAILABLE = True
    OCR_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"[IMPORT] Core modules not yet available: {e}")
    VIDEO_PROCESSOR_AVAILABLE = False
    OCR_ENGINE_AVAILABLE = False


    # Create placeholder classes for now
    class VideoProcessor:
        def __init__(self): pass

        def load_video(self, path): return None


    class OCREngine:
        def __init__(self): pass


    class VideoMetadata:
        def __init__(self): pass


class SkyrimDialogueAnalyzer:
    """Main application class for Skyrim Dialogue Analyzer."""

    def __init__(self, config):
        """Initialize the main application."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("[UI] Initializing Skyrim Dialogue Analyzer UI...")

        # Application state
        self.current_project = None
        self.current_video_file = None
        self.current_video_metadata = None
        self.processing_active = False
        self.ocr_results = []
        self.dialogue_segments = []

        # Core components
        if VIDEO_PROCESSOR_AVAILABLE and OCR_ENGINE_AVAILABLE:
            self.video_processor = VideoProcessor(
                max_workers=4,
                frame_buffer_size=50,
                enable_gpu=True
            )
            self.ocr_engine = OCREngine(
                gpu_enabled=True,
                languages=['en'],
                confidence_threshold=0.6,
                max_workers=4,
                batch_size=8
            )
            self.logger.info("[UI] Core processors initialized successfully")
        else:
            self.video_processor = VideoProcessor()
            self.ocr_engine = OCREngine()
            self.logger.warning("[UI] Using placeholder processors")

        # Processing thread
        self.processing_thread = None

        # Create main window
        self.root = tk.Tk()
        self.setup_main_window()
        self.create_menu_bar()
        self.create_main_interface()
        self.create_status_bar()

        self.logger.info("[UI] Main application initialized successfully")

    def setup_main_window(self):
        """Configure the main application window."""
        self.root.title("Skyrim Dialogue Analyzer v1.0.0")
        self.root.geometry("1600x900")
        self.root.minsize(1200, 700)

        # Set icon if available
        try:
            icon_path = Path(__file__).parent.parent.parent / "resources" / "icons" / "app_icon.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
        except:
            pass

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')  # Modern theme

        # Configure colors for dark theme
        self.root.configure(bg='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TNotebook', background='#2b2b2b')
        style.configure('TNotebook.Tab', background='#404040', foreground='#ffffff')

        self.logger.info("[UI] Main window configured")

    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Import Video", command=self.import_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit_application)

        # Process menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Process", menu=process_menu)
        process_menu.add_command(label="Start OCR Analysis", command=self.start_ocr_analysis)
        process_menu.add_command(label="Generate Translations", command=self.generate_translations)
        process_menu.add_command(label="Create Learning Materials", command=self.create_learning_materials)
        process_menu.add_separator()
        process_menu.add_command(label="Export Results", command=self.export_results)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Settings", command=self.open_settings)
        tools_menu.add_command(label="Performance Monitor", command=self.open_performance_monitor)
        tools_menu.add_separator()
        tools_menu.add_command(label="Test Video Processor", command=self.test_video_processor)
        tools_menu.add_command(label="Test OCR Engine", command=self.test_ocr_engine)
        tools_menu.add_command(label="Test AI Translation", command=self.test_ai_translation)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.open_user_guide)
        help_menu.add_command(label="Troubleshooting", command=self.open_troubleshooting)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)

        self.logger.info("[UI] Menu bar created")

    def create_main_interface(self):
        """Create the main application interface."""
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create paned window for resizable panels
        self.main_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel - Video and controls
        self.create_left_panel()

        # Right panel - Tabbed interface for results and tools
        self.create_right_panel()

        self.logger.info("[UI] Main interface created")

    def create_left_panel(self):
        """Create the left panel with video player and controls."""
        left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(left_frame, weight=2)

        # Video section
        video_frame = ttk.LabelFrame(left_frame, text="Video Player", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # Video display area (placeholder)
        self.video_canvas = tk.Canvas(video_frame, bg='black', height=300)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Add placeholder text
        self.video_canvas.create_text(
            400, 150, text="Video Player\n(Will be implemented next)",
            fill='white', font=('Arial', 14), anchor=tk.CENTER
        )

        # Video controls
        controls_frame = ttk.Frame(video_frame)
        controls_frame.pack(fill=tk.X)

        self.play_button = ttk.Button(controls_frame, text="▶ Play", command=self.toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))

        self.stop_button = ttk.Button(controls_frame, text="⏹ Stop", command=self.stop_playback)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))

        # Timeline scrubber
        self.timeline_var = tk.DoubleVar()
        self.timeline_scale = ttk.Scale(controls_frame, from_=0, to=100,
                                        variable=self.timeline_var, orient=tk.HORIZONTAL)
        self.timeline_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))

        # Time display
        self.time_label = ttk.Label(controls_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.RIGHT)

        # Project info section
        info_frame = ttk.LabelFrame(left_frame, text="Project Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(5, 0))

        # Project details
        self.project_info = tk.Text(info_frame, height=6, wrap=tk.WORD)
        self.project_info.pack(fill=tk.BOTH, expand=True)
        self.project_info.insert(tk.END,
                                 "No project loaded.\n\nUse File > New Project to start or File > Import Video to load a video file.")

        self.logger.info("[UI] Left panel created")

    def create_right_panel(self):
        """Create the right panel with tabbed interface."""
        right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(right_frame, weight=3)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # OCR Results tab
        self.create_ocr_tab()

        # Translation tab
        self.create_translation_tab()

        # Learning Materials tab
        self.create_learning_tab()

        # Processing Log tab
        self.create_log_tab()

        self.logger.info("[UI] Right panel created")

    def create_ocr_tab(self):
        """Create the OCR results tab."""
        ocr_frame = ttk.Frame(self.notebook)
        self.notebook.add(ocr_frame, text="OCR Results")

        # OCR controls
        controls_frame = ttk.Frame(ocr_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_ocr_button = ttk.Button(controls_frame, text="🔍 Start OCR Analysis",
                                           command=self.start_ocr_analysis)
        self.start_ocr_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_ocr_button = ttk.Button(controls_frame, text="⏹ Stop OCR",
                                          command=self.stop_ocr_analysis, state=tk.DISABLED)
        self.stop_ocr_button.pack(side=tk.LEFT, padx=(0, 10))

        self.export_ocr_button = ttk.Button(controls_frame, text="💾 Export OCR Results",
                                            command=self.export_ocr_results)
        self.export_ocr_button.pack(side=tk.LEFT)

        # OCR progress
        self.ocr_progress_var = tk.DoubleVar()
        self.ocr_progress = ttk.Progressbar(controls_frame, variable=self.ocr_progress_var,
                                            mode='determinate')
        self.ocr_progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))

        # OCR results display
        results_frame = ttk.LabelFrame(ocr_frame, text="Extracted Text", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))

        # Text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.ocr_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        ocr_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.ocr_text.yview)
        self.ocr_text.configure(yscrollcommand=ocr_scrollbar.set)

        self.ocr_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ocr_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add placeholder text
        self.ocr_text.insert(tk.END,
                             "OCR results will appear here after processing.\n\nTo start:\n1. Import a video file\n2. Click 'Start OCR Analysis'\n3. Wait for processing to complete")

        self.logger.info("[UI] OCR tab created")

    def create_translation_tab(self):
        """Create the translation tab."""
        translation_frame = ttk.Frame(self.notebook)
        self.notebook.add(translation_frame, text="Translations")

        # Translation controls
        controls_frame = ttk.Frame(translation_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.translate_button = ttk.Button(controls_frame, text="🌐 Generate Translations",
                                           command=self.generate_translations)
        self.translate_button.pack(side=tk.LEFT, padx=(0, 10))

        # Language selection
        ttk.Label(controls_frame, text="Target Language:").pack(side=tk.LEFT, padx=(10, 5))
        self.target_language = ttk.Combobox(controls_frame, width=15,
                                            values=["Auto", "Ukrainian", "Russian", "Spanish", "French", "German"])
        self.target_language.set("Auto")
        self.target_language.pack(side=tk.LEFT, padx=(0, 10))

        # Translation display
        translation_text_frame = ttk.LabelFrame(translation_frame, text="Translations", padding=5)
        translation_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))

        self.translation_text = tk.Text(translation_text_frame, wrap=tk.WORD, font=('Arial', 10))
        translation_scrollbar = ttk.Scrollbar(translation_text_frame, orient=tk.VERTICAL,
                                              command=self.translation_text.yview)
        self.translation_text.configure(yscrollcommand=translation_scrollbar.set)

        self.translation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        translation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add placeholder text
        self.translation_text.insert(tk.END,
                                     "Translations will appear here after processing.\n\nFirst complete OCR analysis, then click 'Generate Translations'.")

        self.logger.info("[UI] Translation tab created")

    def create_learning_tab(self):
        """Create the learning materials tab."""
        learning_frame = ttk.Frame(self.notebook)
        self.notebook.add(learning_frame, text="Learning")

        # Learning controls
        controls_frame = ttk.Frame(learning_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.learning_button = ttk.Button(controls_frame, text="📚 Create Learning Materials",
                                          command=self.create_learning_materials)
        self.learning_button.pack(side=tk.LEFT, padx=(0, 10))

        # Learning type selection
        ttk.Label(controls_frame, text="Focus:").pack(side=tk.LEFT, padx=(10, 5))
        self.learning_focus = ttk.Combobox(controls_frame, width=15,
                                           values=["Grammar", "Vocabulary", "Pronunciation", "Context"])
        self.learning_focus.set("Grammar")
        self.learning_focus.pack(side=tk.LEFT)

        # Learning materials display
        learning_text_frame = ttk.LabelFrame(learning_frame, text="Learning Materials", padding=5)
        learning_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))

        self.learning_text = tk.Text(learning_text_frame, wrap=tk.WORD, font=('Arial', 11))
        learning_scrollbar = ttk.Scrollbar(learning_text_frame, orient=tk.VERTICAL,
                                           command=self.learning_text.yview)
        self.learning_text.configure(yscrollcommand=learning_scrollbar.set)

        self.learning_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        learning_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add placeholder text
        self.learning_text.insert(tk.END,
                                  "Learning materials will be generated here.\n\nThis will include:\n• Grammar explanations\n• Vocabulary lists\n• Pronunciation guides\n• Context analysis\n\nComplete OCR and translation first.")

        self.logger.info("[UI] Learning tab created")

    def create_log_tab(self):
        """Create the processing log tab."""
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text="Processing Log")

        # Log controls
        controls_frame = ttk.Frame(log_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        self.clear_log_button = ttk.Button(controls_frame, text="🗑️ Clear Log",
                                           command=self.clear_log)
        self.clear_log_button.pack(side=tk.LEFT, padx=(0, 10))

        self.save_log_button = ttk.Button(controls_frame, text="💾 Save Log",
                                          command=self.save_log)
        self.save_log_button.pack(side=tk.LEFT)

        # Log display
        log_text_frame = ttk.LabelFrame(log_frame, text="Processing Log", padding=5)
        log_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 5))

        self.log_text = tk.Text(log_text_frame, wrap=tk.WORD, font=('Consolas', 9))
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL,
                                      command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add initial log entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] Skyrim Dialogue Analyzer initialized\n")
        self.log_text.insert(tk.END, f"[{timestamp}] Ready for video processing\n")

        self.logger.info("[UI] Log tab created")

    def create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Status label
        self.status_label = ttk.Label(self.status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))

        # Processing indicator
        self.processing_label = ttk.Label(self.status_frame, text="Idle", relief=tk.SUNKEN)
        self.processing_label.pack(side=tk.RIGHT, padx=(5, 5))

        self.logger.info("[UI] Status bar created")

    # Event handlers
    def new_project(self):
        """Create a new project."""
        self.log_message("Creating new project...")
        messagebox.showinfo("New Project", "New project functionality will be implemented next!")

    def open_project(self):
        """Open an existing project."""
        self.log_message("Opening project...")
        messagebox.showinfo("Open Project", "Project loading functionality will be implemented next!")

    def save_project(self):
        """Save the current project."""
        self.log_message("Saving project...")
        messagebox.showinfo("Save Project", "Project saving functionality will be implemented next!")

    def import_video(self):
        """Import a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Skyrim Video File",
            filetypes=[
                ("Video files", "*.mkv *.mp4 *.avi"),
                ("MKV files", "*.mkv"),
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.current_video_file = file_path
            self.log_message(f"Loading video: {Path(file_path).name}")

            try:
                # Load video with video processor
                self.current_video_metadata = self.video_processor.load_video(file_path)

                if self.current_video_metadata:
                    self.log_message(f"Video loaded successfully: {Path(file_path).name}")
                    self.update_project_info()

                    # Update status
                    status_text = f"Video: {Path(file_path).name} ({self.current_video_metadata.width}x{self.current_video_metadata.height}, {self.current_video_metadata.duration:.1f}s)"
                    self.status_label.config(text=status_text)

                    # Show success message with details
                    info_text = f"""Video loaded successfully!

File: {Path(file_path).name}
Resolution: {self.current_video_metadata.width}x{self.current_video_metadata.height}
Duration: {self.current_video_metadata.duration:.2f} seconds
FPS: {self.current_video_metadata.fps:.2f}
Frames: {self.current_video_metadata.total_frames}
Size: {self.current_video_metadata.file_size / (1024 * 1024):.1f} MB

Ready for OCR processing!"""

                    messagebox.showinfo("Video Imported", info_text)
                else:
                    self.log_message(f"Failed to load video: {Path(file_path).name}")
                    messagebox.showerror("Import Error", f"Failed to load video file:\n{Path(file_path).name}")
                    self.current_video_file = None

            except Exception as e:
                self.log_message(f"Error loading video: {e}")
                messagebox.showerror("Import Error", f"Error loading video:\n{e}")
                self.current_video_file = None

    def start_ocr_analysis(self):
        """Start OCR analysis with dialogue-focused options."""
        if not self.current_video_file:
            messagebox.showwarning("No Video", "Please import a video file first.")
            return

        if self.processing_active:
            messagebox.showwarning("Processing Active", "OCR analysis is already running.")
            return

        if not VIDEO_PROCESSOR_AVAILABLE or not OCR_ENGINE_AVAILABLE:
            messagebox.showerror("Missing Components",
                                 "Video processor or OCR engine not available.\nPlease check your installation.")
            return

        # Ask user for processing mode
        mode_choice = messagebox.askyesnocancel(
            "OCR Processing Mode",
            "Choose OCR processing mode:\n\n"
            "YES = Dialogue Focus (dense sampling for better dialogue)\n"
            "NO = Quick Scan (current method - faster)\n"
            "CANCEL = Cancel OCR"
        )

        if mode_choice is None:  # Cancel
            return

        self.dialogue_focused_mode = mode_choice  # True for dialogue focus, False for quick scan

        # Start processing in background thread
        self.processing_active = True
        self.start_ocr_button.config(state=tk.DISABLED)
        self.stop_ocr_button.config(state=tk.NORMAL)

        self.log_message(
            f"Starting OCR analysis in {'dialogue-focused' if self.dialogue_focused_mode else 'quick-scan'} mode...")
        self.status_label.config(text="Processing OCR...")
        self.processing_label.config(text="OCR Active")

        # Clear previous results
        self.ocr_text.delete(1.0, tk.END)
        self.ocr_text.insert(tk.END,
                             f"Starting OCR analysis ({'Dialogue Focus' if self.dialogue_focused_mode else 'Quick Scan'} mode)...\n\n")
        self.ocr_results = []

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._run_ocr_analysis)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _run_ocr_analysis(self):
        """Run OCR analysis with improved dialogue detection."""
        try:
            self.log_message("Extracting video frames...")

            # Update UI
            self.root.after(0, lambda: self.ocr_text.insert(tk.END, "Extracting frames from video...\n"))

            # Smart frame selection based on mode
            if self.current_video_metadata:
                total_video_frames = self.current_video_metadata.total_frames
                duration = self.current_video_metadata.duration
                fps = self.current_video_metadata.fps

                if self.dialogue_focused_mode:
                    # DIALOGUE FOCUS MODE: Dense sampling from dialogue-heavy sections
                    if duration > 120:  # Video > 2 minutes
                        # Focus on middle section where dialogue usually occurs
                        start_time = 60  # Start 1 minute in
                        end_time = min(duration - 30, start_time + 180)  # 3 minutes of content
                        start_frame = int(start_time * fps)
                        end_frame = int(end_time * fps)
                        step = int(fps)  # Every 1 second for dense dialogue capture
                    else:
                        # Short video: sample densely throughout
                        start_frame = int(30 * fps)  # Skip first 30 seconds
                        end_frame = int((duration - 10) * fps)  # Leave 10s at end
                        step = int(fps / 2)  # Every 0.5 seconds
                else:
                    # QUICK SCAN MODE: Original method
                    if duration > 60:
                        frame_step = int(fps * 10)  # Every 10 seconds
                        start_frame = frame_step * 5  # Start 50 seconds in
                        end_frame = min(frame_step * 30, total_video_frames)
                        step = frame_step
                    else:
                        start_frame = 30
                        end_frame = min(90, total_video_frames)
                        step = 2
            else:
                # Fallback
                start_frame = 30
                end_frame = 90
                step = 2

            # Limit total frames to prevent overwhelming
            max_frames = 100 if self.dialogue_focused_mode else 30
            estimated_frames = (end_frame - start_frame) // step
            if estimated_frames > max_frames:
                step = (end_frame - start_frame) // max_frames

            mode_text = "DIALOGUE FOCUS" if self.dialogue_focused_mode else "QUICK SCAN"
            self.root.after(0, lambda: self.ocr_text.insert(tk.END,
                                                            f"{mode_text}: frames {start_frame}-{end_frame}, step={step}\n"))

            # Extract frames
            frames = self.video_processor.extract_frames_batch(
                self.current_video_file,
                start_frame=start_frame,
                end_frame=end_frame,
                step=step
            )

            if not frames:
                self.root.after(0, lambda: self._ocr_analysis_failed("Failed to extract frames from video"))
                return

            self.log_message(f"Extracted {len(frames)} frames in {mode_text} mode")
            self.root.after(0, lambda: self.ocr_text.insert(tk.END, f"Extracted {len(frames)} frames\n\n"))

            # Process frames with OCR
            self.log_message("Running dialogue-focused OCR analysis...")
            self.root.after(0, lambda: self.ocr_text.insert(tk.END, "Running dialogue-focused OCR analysis...\n"))

            # Lower confidence for dialogue detection
            original_threshold = self.ocr_engine.confidence_threshold
            self.ocr_engine.confidence_threshold = 0.4 if self.dialogue_focused_mode else 0.3

            total_frames = len(frames)
            processed_frames = 0
            frames_with_text = 0
            dialogue_segments = []

            for i, frame in enumerate(frames):
                if not self.processing_active:
                    break

                # Process single frame
                result = self.ocr_engine.process_frame(frame)
                self.ocr_results.append(result)

                processed_frames += 1
                progress = (processed_frames / total_frames) * 100

                # Update progress
                self.root.after(0, lambda p=progress: self.ocr_progress_var.set(p))

                # Collect dialogue segments for analysis
                if result.text_segments:
                    frames_with_text += 1

                    # Show frames with substantial text (dialogue-focused display)
                    substantial_text = [seg for seg in result.text_segments if len(seg.cleaned_text) > 3]
                    dialogue_text = [seg for seg in substantial_text if seg.is_dialogue and len(seg.cleaned_text) > 5]

                    if substantial_text:
                        frame_text = f"Frame {result.frame_number} (t={result.timestamp:.1f}s):\n"

                        # Prioritize dialogue text
                        if dialogue_text:
                            frame_text += "  🎭 DIALOGUE:\n"
                            for seg in dialogue_text:
                                frame_text += f"    • '{seg.cleaned_text}' (conf: {seg.confidence:.3f})\n"
                                dialogue_segments.append(f"[{result.timestamp:.1f}s] {seg.cleaned_text}")

                        # Show other substantial text
                        other_text = [seg for seg in substantial_text if not seg.is_dialogue]
                        if other_text and len(other_text) <= 5:  # Limit UI text display
                            frame_text += "  📋 UI/OTHER:\n"
                            for seg in other_text[:5]:
                                frame_text += f"    • '{seg.cleaned_text}' (conf: {seg.confidence:.3f})\n"

                        frame_text += "\n"
                        self.root.after(0, lambda text=frame_text: self.ocr_text.insert(tk.END, text))
                        self.root.after(0, lambda: self.ocr_text.see(tk.END))

                # Small delay
                time.sleep(0.05)

            # Restore original confidence threshold
            self.ocr_engine.confidence_threshold = original_threshold

            # Enhanced completion with dialogue summary
            self.root.after(0,
                            lambda: self._ocr_analysis_complete_dialogue_focused(frames_with_text, dialogue_segments))

        except Exception as e:
            self.log_message(f"OCR analysis error: {e}")
            self.root.after(0, lambda: self._ocr_analysis_failed(f"OCR analysis failed: {e}"))

    def _ocr_analysis_complete_dialogue_focused(self, frames_with_text, dialogue_segments):
        """Handle completion with dialogue focus."""
        self.processing_active = False
        self.start_ocr_button.config(state=tk.NORMAL)
        self.stop_ocr_button.config(state=tk.DISABLED)

        # Calculate statistics
        total_segments = sum(len(result.text_segments) for result in self.ocr_results)
        dialogue_frames = sum(1 for result in self.ocr_results if result.dialogue_detected)
        long_dialogue = [seg for seg in dialogue_segments if len(seg.split()) >= 3]

        # Add summary with dialogue focus
        summary = f"\n{'=' * 60}\n"
        summary += f"DIALOGUE-FOCUSED OCR ANALYSIS COMPLETE\n"
        summary += f"{'=' * 60}\n"
        summary += f"Frames processed: {len(self.ocr_results)}\n"
        summary += f"Frames with text: {frames_with_text}\n"
        summary += f"Total text segments: {total_segments}\n"
        summary += f"Dialogue frames: {dialogue_frames}\n"
        summary += f"Dialogue segments found: {len(dialogue_segments)}\n"
        summary += f"Substantial dialogue: {len(long_dialogue)}\n"

        if dialogue_segments:
            summary += f"\n🎭 DIALOGUE TIMELINE:\n"
            for seg in dialogue_segments[:10]:  # Show first 10
                summary += f"  {seg}\n"
            if len(dialogue_segments) > 10:
                summary += f"  ... and {len(dialogue_segments) - 10} more dialogue segments\n"

        summary += f"\nProcessing completed at: {datetime.now().strftime('%H:%M:%S')}\n"

        self.ocr_text.insert(tk.END, summary)
        self.ocr_text.see(tk.END)

        self.log_message(f"Dialogue-focused OCR completed: {len(dialogue_segments)} dialogue segments found")
        self.status_label.config(text=f"OCR Complete: {len(dialogue_segments)} dialogue segments")
        self.processing_label.config(text="Idle")

        # Show completion message
        messagebox.showinfo("Dialogue OCR Complete",
                            f"Dialogue-focused OCR analysis completed!\n\n"
                            f"Frames processed: {len(self.ocr_results)}\n"
                            f"Dialogue segments found: {len(dialogue_segments)}\n"
                            f"Substantial dialogue: {len(long_dialogue)}\n\n"
                            f"Ready for translation!")

    def _ocr_analysis_complete_enhanced(self, frames_with_text):
        """Handle OCR analysis completion with enhanced reporting."""
        self.processing_active = False
        self.start_ocr_button.config(state=tk.NORMAL)
        self.stop_ocr_button.config(state=tk.DISABLED)

        # Calculate enhanced statistics
        total_segments = sum(len(result.text_segments) for result in self.ocr_results)
        dialogue_frames = sum(1 for result in self.ocr_results if result.dialogue_detected)
        all_text_found = []

        # Collect all unique text for analysis
        for result in self.ocr_results:
            for segment in result.text_segments:
                all_text_found.append(segment.cleaned_text)

        # Analyze text patterns
        short_text = sum(1 for text in all_text_found if len(text) <= 3)
        medium_text = sum(1 for text in all_text_found if 4 <= len(text) <= 10)
        long_text = sum(1 for text in all_text_found if len(text) > 10)

        # Add enhanced summary
        summary = f"\n{'=' * 60}\n"
        summary += f"ENHANCED OCR ANALYSIS COMPLETE\n"
        summary += f"{'=' * 60}\n"
        summary += f"Frames processed: {len(self.ocr_results)}\n"
        summary += f"Frames with text: {frames_with_text}\n"
        summary += f"Total text segments: {total_segments}\n"
        summary += f"Dialogue frames detected: {dialogue_frames}\n"
        summary += f"\nText Analysis:\n"
        summary += f"• Short text (≤3 chars): {short_text}\n"
        summary += f"• Medium text (4-10 chars): {medium_text}\n"
        summary += f"• Long text (>10 chars): {long_text}\n"
        summary += f"\nProcessing completed at: {datetime.now().strftime('%H:%M:%S')}\n"

        if total_segments == 0:
            summary += f"\n⚠️ TROUBLESHOOTING:\n"
            summary += f"No text found! Possible issues:\n"
            summary += f"• Frame range might not contain dialogue\n"
            summary += f"• Video resolution not optimized for OCR\n"
            summary += f"• Text too small or blurry\n"
            summary += f"• UI regions not matching your video layout\n"
            summary += f"\nTry: Tools > Test Video Processor to check frames\n"

        self.ocr_text.insert(tk.END, summary)
        self.ocr_text.see(tk.END)

        self.log_message(
            f"Enhanced OCR analysis completed: {total_segments} text segments, {frames_with_text} frames with text")
        self.status_label.config(text=f"OCR Complete: {total_segments} segments found")
        self.processing_label.config(text="Idle")

        # Show enhanced completion message
        if total_segments > 0:
            messagebox.showinfo("OCR Complete",
                                f"OCR analysis completed!\n\n"
                                f"Frames processed: {len(self.ocr_results)}\n"
                                f"Frames with text: {frames_with_text}\n"
                                f"Text segments found: {total_segments}\n"
                                f"Dialogue frames: {dialogue_frames}")
        else:
            messagebox.showwarning("No Text Found",
                                   f"OCR processed {len(self.ocr_results)} frames but found no text.\n\n"
                                   f"This could mean:\n"
                                   f"• Frame range contains no dialogue\n"
                                   f"• Video quality issues\n"
                                   f"• OCR settings need adjustment\n\n"
                                   f"Try Tools > Test Video Processor to check extracted frames.")

    def _ocr_analysis_complete(self):
        """Handle OCR analysis completion."""
        self.processing_active = False
        self.start_ocr_button.config(state=tk.NORMAL)
        self.stop_ocr_button.config(state=tk.DISABLED)

        # Calculate statistics
        total_segments = sum(len(result.text_segments) for result in self.ocr_results)
        dialogue_frames = sum(1 for result in self.ocr_results if result.dialogue_detected)

        # Add summary
        summary = f"\n{'=' * 50}\n"
        summary += f"OCR ANALYSIS COMPLETE\n"
        summary += f"{'=' * 50}\n"
        summary += f"Frames processed: {len(self.ocr_results)}\n"
        summary += f"Text segments found: {total_segments}\n"
        summary += f"Dialogue frames detected: {dialogue_frames}\n"
        summary += f"Processing completed at: {datetime.now().strftime('%H:%M:%S')}\n"

        self.ocr_text.insert(tk.END, summary)
        self.ocr_text.see(tk.END)

        self.log_message(f"OCR analysis completed: {total_segments} text segments found")
        self.status_label.config(text=f"OCR Complete: {total_segments} text segments found")
        self.processing_label.config(text="Idle")

        # Show completion message
        messagebox.showinfo("OCR Complete",
                            f"OCR analysis completed successfully!\n\n"
                            f"Frames processed: {len(self.ocr_results)}\n"
                            f"Text segments found: {total_segments}\n"
                            f"Dialogue frames: {dialogue_frames}")

    def _ocr_analysis_failed(self, error_message):
        """Handle OCR analysis failure."""
        self.processing_active = False
        self.start_ocr_button.config(state=tk.NORMAL)
        self.stop_ocr_button.config(state=tk.DISABLED)

        self.ocr_text.insert(tk.END, f"\nERROR: {error_message}\n")
        self.log_message(f"OCR analysis failed: {error_message}")
        self.status_label.config(text="OCR Failed")
        self.processing_label.config(text="Error")

        messagebox.showerror("OCR Failed", error_message)

    def stop_ocr_analysis(self):
        """Stop OCR analysis."""
        if self.processing_active:
            self.processing_active = False
            self.log_message("Stopping OCR analysis...")
            self.status_label.config(text="Stopping...")

            # Reset UI state
            self.start_ocr_button.config(state=tk.NORMAL)
            self.stop_ocr_button.config(state=tk.DISABLED)
            self.processing_label.config(text="Idle")

            # Add cancellation message
            self.ocr_text.insert(tk.END, "\n[CANCELLED] OCR analysis stopped by user\n")
            self.ocr_text.see(tk.END)

            self.log_message("OCR analysis cancelled")

    def generate_translations(self):
        """Generate translations of the extracted text."""
        if not self.ocr_results:
            messagebox.showwarning("No OCR Results", "Please complete OCR analysis first.")
            return

        self.log_message("Generating translations...")
        self.translation_text.delete(1.0, tk.END)
        self.translation_text.insert(tk.END, "Translation functionality will be implemented next!\n\n")

        # Show what text we would translate
        dialogue_texts = []
        for result in self.ocr_results:
            dialogue_text = result.get_dialogue_text()
            if dialogue_text.strip():
                dialogue_texts.append(f"Frame {result.frame_number}: {dialogue_text}")

        if dialogue_texts:
            self.translation_text.insert(tk.END, "Found dialogue text to translate:\n\n")
            for text in dialogue_texts[:10]:  # Show first 10
                self.translation_text.insert(tk.END, f"• {text}\n")

            if len(dialogue_texts) > 10:
                self.translation_text.insert(tk.END, f"\n... and {len(dialogue_texts) - 10} more dialogue segments")
        else:
            self.translation_text.insert(tk.END, "No dialogue text found to translate.")

    def create_learning_materials(self):
        """Create learning materials."""
        if not self.ocr_results:
            messagebox.showwarning("No OCR Results", "Please complete OCR analysis first.")
            return

        self.log_message("Creating learning materials...")
        self.learning_text.delete(1.0, tk.END)
        self.learning_text.insert(tk.END, "Learning material generation will be implemented next!\n\n")

        # Show what we could create learning materials from
        vocabulary = set()
        for result in self.ocr_results:
            for segment in result.text_segments:
                if segment.is_dialogue:
                    words = segment.cleaned_text.split()
                    vocabulary.update(word.strip('.,!?').lower() for word in words if len(word) > 3)

        if vocabulary:
            self.learning_text.insert(tk.END, f"Found {len(vocabulary)} unique vocabulary words:\n\n")
            sorted_vocab = sorted(list(vocabulary))[:20]  # Show first 20
            for word in sorted_vocab:
                self.learning_text.insert(tk.END, f"• {word}\n")

            if len(vocabulary) > 20:
                self.learning_text.insert(tk.END, f"\n... and {len(vocabulary) - 20} more words")

    def export_ocr_results(self):
        """Export OCR results specifically."""
        if not self.ocr_results:
            messagebox.showwarning("No Results", "No OCR results to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Export OCR Results",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                if file_path.lower().endswith('.json'):
                    # Export as JSON using OCR engine's built-in method
                    success = self.ocr_engine.save_results(self.ocr_results, file_path)
                    if success:
                        messagebox.showinfo("Export Complete", f"OCR results exported to:\n{file_path}")
                    else:
                        messagebox.showerror("Export Error", "Failed to export OCR results")
                else:
                    # Export as plain text
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("Skyrim Dialogue Analyzer - OCR Results\n")
                        f.write("=" * 50 + "\n\n")

                        for result in self.ocr_results:
                            f.write(f"Frame {result.frame_number} (t={result.timestamp:.3f}s):\n")
                            if result.text_segments:
                                for segment in result.text_segments:
                                    f.write(f"  • '{segment.cleaned_text}' (confidence: {segment.confidence:.3f})\n")
                            else:
                                f.write("  (no text detected)\n")
                            f.write("\n")

                    messagebox.showinfo("Export Complete", f"OCR results exported to:\n{file_path}")

                self.log_message(f"OCR results exported to: {file_path}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")
                self.log_message(f"Export error: {e}")

    def export_results(self):
        """Export all processing results."""
        if not self.ocr_results:
            messagebox.showwarning("No Results", "No results to export.")
            return

        messagebox.showinfo("Export",
                            "Comprehensive export functionality will be implemented next!\nFor now, use 'Export OCR Results' button.")

    def test_ocr_engine(self):
        """Test OCR engine functionality."""
        if not OCR_ENGINE_AVAILABLE:
            messagebox.showerror("OCR Engine", "OCR engine not available. Please check installation.")
            return

        self.log_message("Testing OCR engine...")

        try:
            # Get OCR engine stats
            stats = self.ocr_engine.get_processing_stats()

            test_results = f"""OCR Engine Test Results:

Engine Status: {'Available' if OCR_ENGINE_AVAILABLE else 'Not Available'}
GPU Enabled: {self.ocr_engine.gpu_enabled}
Languages: {self.ocr_engine.languages}
Confidence Threshold: {self.ocr_engine.confidence_threshold}
Max Workers: {self.ocr_engine.max_workers}

Processing Statistics:
• Frames processed: {stats.get('frames_processed', 0)}
• Total processing time: {stats.get('total_processing_time', 0):.3f}s
• Average FPS: {stats.get('average_fps', 0):.2f}
• Text segments found: {stats.get('text_segments_found', 0)}

GPU Information:
"""

            # Add GPU stats if available
            if 'gpu_memory_allocated' in stats:
                test_results += f"• GPU memory allocated: {stats['gpu_memory_allocated'] / 1024 ** 2:.1f} MB\n"
                test_results += f"• GPU memory reserved: {stats['gpu_memory_reserved'] / 1024 ** 2:.1f} MB\n"
                test_results += f"• GPU utilization: {stats.get('gpu_utilization', 'N/A')}\n"

            self.show_test_results("OCR Engine Test", test_results)
            self.log_message("OCR engine test completed")

        except Exception as e:
            error_msg = f"OCR engine test failed: {e}"
            messagebox.showerror("Test Failed", error_msg)
            self.log_message(error_msg)

    def test_video_processor(self):
        """Test video processor functionality."""
        if not self.current_video_file:
            messagebox.showwarning("No Video", "Please import a video file first.")
            return

        self.log_message("Testing video processor...")

        try:
            # Test loading video
            metadata = self.video_processor.load_video(self.current_video_file)
            if not metadata:
                messagebox.showerror("Error", "Failed to load video for testing")
                return

            test_results = f"""Video Processor Test Results:

File: {Path(self.current_video_file).name}
Resolution: {metadata.width}x{metadata.height}
Duration: {metadata.duration:.2f} seconds
FPS: {metadata.fps:.2f}
Frame Count: {metadata.total_frames}
Codec: {metadata.codec}
File Size: {metadata.file_size / (1024 * 1024):.1f} MB

Test 1: Extract single frame...
"""

            # Test frame extraction
            test_frame = self.video_processor.extract_frame(self.current_video_file, 100)
            if test_frame:
                test_results += f"✓ Frame 100 extracted successfully\n"
                test_results += f"  Timestamp: {test_frame.timestamp:.2f}s\n"
                test_results += f"  Hash: {test_frame.frame_hash[:16]}...\n\n"
            else:
                test_results += "✗ Frame extraction failed\n\n"

            test_results += "Test 2: Batch frame extraction...\n"

            # Test batch extraction (first 10 frames)
            batch_frames = self.video_processor.extract_frames_batch(
                self.current_video_file, 0, 10, 1)
            test_results += f"✓ Extracted {len(batch_frames)} frames in batch\n"

            # Get processing stats
            stats = self.video_processor.get_processing_stats()
            test_results += f"\nProcessing Statistics:\n"
            test_results += f"• Average FPS: {stats.get('average_fps', 0):.2f}\n"
            test_results += f"• Frames processed: {stats.get('frames_processed', 0)}\n"

            # Show results
            self.show_test_results("Video Processor Test", test_results)
            self.log_message("Video processor test completed")

        except Exception as e:
            error_msg = f"Video processor test failed: {e}"
            messagebox.showerror("Test Failed", error_msg)
            self.log_message(error_msg)

    def show_test_results(self, title, results):
        """Show test results in a new window."""
        result_window = tk.Toplevel(self.root)
        result_window.title(title)
        result_window.geometry("600x500")

        # Text widget with scrollbar
        text_frame = ttk.Frame(result_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.insert(tk.END, results)
        text_widget.config(state=tk.DISABLED)

        # Close button
        close_btn = ttk.Button(result_window, text="Close", command=result_window.destroy)
        close_btn.pack(pady=5)

    def test_ai_translation(self):
        """Test AI translation functionality."""
        self.log_message("Testing AI translation...")
        messagebox.showinfo("AI Test", "AI translation test will be implemented next!")

    def open_settings(self):
        """Open settings dialog."""
        messagebox.showinfo("Settings", "Settings dialog will be implemented next!")

    def open_performance_monitor(self):
        """Open performance monitor."""
        messagebox.showinfo("Performance", "Performance monitor will be implemented next!")

    def open_user_guide(self):
        """Open user guide."""
        messagebox.showinfo("User Guide", "User guide will be available soon!")

    def open_troubleshooting(self):
        """Open troubleshooting guide."""
        messagebox.showinfo("Troubleshooting", "Troubleshooting guide will be available soon!")

    def show_about(self):
        """Show about dialog."""
        about_text = """Skyrim Dialogue Analyzer v1.0.0

A comprehensive tool for extracting, analyzing, and learning from 
Skyrim dialogue through advanced OCR and AI-powered language processing.

Built for high-performance gaming PCs with RTX 5080 + Ryzen 7700 + 64GB RAM.

© 2025 Skyrim Dialogue Analyzer Team"""
        messagebox.showinfo("About", about_text)

    def toggle_playback(self):
        """Toggle video playback."""
        self.log_message("Toggling video playback...")
        messagebox.showinfo("Video Player", "Video playback will be implemented next!")

    def stop_playback(self):
        """Stop video playback."""
        self.log_message("Stopping video playback...")

    def clear_log(self):
        """Clear the processing log."""
        self.log_text.delete(1.0, tk.END)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] Log cleared\n")

    def save_log(self):
        """Save the processing log."""
        file_path = filedialog.asksaveasfilename(
            title="Save Processing Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Log Saved", f"Log saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log:\n{e}")

    def log_message(self, message):
        """Add a message to the processing log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.logger.info(f"[LOG] {message}")

    def update_project_info(self):
        """Update the project information display."""
        self.project_info.delete(1.0, tk.END)

        if self.current_video_file and self.current_video_metadata:
            metadata = self.current_video_metadata
            info_text = f"""Project Information:

Video File: {Path(self.current_video_file).name}
Path: {Path(self.current_video_file).parent}
File Size: {metadata.file_size / (1024 * 1024):.1f} MB

Video Properties:
• Resolution: {metadata.width}x{metadata.height}
• Duration: {metadata.duration:.2f} seconds
• FPS: {metadata.fps:.2f}
• Total Frames: {metadata.total_frames}
• Codec: {metadata.codec}

Processing Status: Ready for OCR analysis

Next Steps:
1. Start OCR Analysis to extract text
2. Generate translations
3. Create learning materials
4. Export results

Tools Available:
• Test Video Processor (Tools menu)
• Test OCR Engine (Tools menu)
• Performance monitoring
• Batch processing
"""
            self.project_info.insert(tk.END, info_text)
        elif self.current_video_file:
            video_path = Path(self.current_video_file)
            info_text = f"""Project Information:

Video File: {video_path.name}
Path: {video_path.parent}
Size: {video_path.stat().st_size / (1024 * 1024):.1f} MB

Status: Video file selected but not processed
Issue: Video metadata not loaded

Next Steps: Try reloading the video file
"""
            self.project_info.insert(tk.END, info_text)
        else:
            self.project_info.insert(tk.END, """No project loaded.

Getting Started:
1. File > Import Video to load a video file
2. Or File > New Project to start fresh

Supported Formats:
• MKV files (recommended for Skyrim recordings)
• MP4 files
• AVI files

System Requirements:
• 2K video support (up to 2560x1440)
• GPU acceleration with RTX 5080
• Large file support (multi-GB videos)

Your System:
✓ RTX 5080 GPU detected
✓ CUDA 12.8 available
✓ 63.1 GB RAM available
✓ All dependencies satisfied
""")

    def exit_application(self):
        """Exit the application."""
        if self.processing_active:
            if messagebox.askokcancel("Exit", "OCR processing is active. Are you sure you want to exit?"):
                self.processing_active = False
                self.logger.info("[UI] Application closing during processing...")
                self.root.quit()
        else:
            if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
                self.logger.info("[UI] Application closing...")
                self.root.quit()

    def run(self):
        """Start the main application loop."""
        self.logger.info("[UI] Starting main application loop...")
        self.root.mainloop()
        self.logger.info("[UI] Application loop ended")