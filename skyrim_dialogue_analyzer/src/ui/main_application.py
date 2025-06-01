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

# Import core modules (will be created later)
try:
    from core.video_processor import VideoProcessor
    from core.ocr_engine import OCREngine
    from core.ai_translator import AITranslator
    from data.database_manager import DatabaseManager
    from data.session_manager import SessionManager
except ImportError as e:
    logging.warning(f"[IMPORT] Core modules not yet available: {e}")


    # Create placeholder classes for now
    class VideoProcessor:
        def __init__(self): pass


    class OCREngine:
        def __init__(self): pass


    class AITranslator:
        def __init__(self): pass


    class DatabaseManager:
        def __init__(self): pass


    class SessionManager:
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
        self.processing_active = False
        self.ocr_results = []
        self.dialogue_segments = []

        # Core components (placeholders for now)
        self.video_processor = VideoProcessor()
        self.ocr_engine = OCREngine()
        self.ai_translator = AITranslator()
        self.db_manager = DatabaseManager()
        self.session_manager = SessionManager()

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

    # Event handlers (placeholder implementations)
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

            # Load video with video processor
            success = self.video_processor.load_video(file_path)

            if success:
                self.log_message(f"Video loaded successfully: {Path(file_path).name}")
                self.update_project_info()

                # Update status with video info
                metadata = self.video_processor.metadata
                if metadata:
                    status_text = f"Video: {metadata.filename} ({metadata.width}x{metadata.height}, {metadata.duration:.1f}s)"
                    self.status_label.config(text=status_text)

                    # Show success message with details
                    info_text = f"""Video loaded successfully!

File: {metadata.filename}
Resolution: {metadata.width}x{metadata.height}
Duration: {metadata.duration:.2f} seconds
FPS: {metadata.fps:.2f}
Frames: {metadata.frame_count}
Size: {metadata.file_size / (1024 * 1024):.1f} MB

Ready for processing!"""

                    messagebox.showinfo("Video Imported", info_text)
                else:
                    messagebox.showinfo("Video Imported", f"Video file loaded:\n{Path(file_path).name}")
            else:
                self.log_message(f"Failed to load video: {Path(file_path).name}")
                messagebox.showerror("Import Error", f"Failed to load video file:\n{Path(file_path).name}")
                self.current_video_file = None

    def start_ocr_analysis(self):
        """Start OCR analysis of the video."""
        if not self.current_video_file:
            messagebox.showwarning("No Video", "Please import a video file first.")
            return

        self.log_message("Starting OCR analysis...")
        self.status_label.config(text="Processing OCR...")
        self.processing_label.config(text="OCR Active")

        # TODO: Implement actual OCR processing
        messagebox.showinfo("OCR Analysis", "OCR processing will be implemented with the core modules!")

    def generate_translations(self):
        """Generate translations of the extracted text."""
        self.log_message("Generating translations...")
        messagebox.showinfo("Translation", "Translation functionality will be implemented next!")

    def create_learning_materials(self):
        """Create learning materials."""
        self.log_message("Creating learning materials...")
        messagebox.showinfo("Learning Materials", "Learning material generation will be implemented next!")

    def export_results(self):
        """Export processing results."""
        self.log_message("Exporting results...")
        messagebox.showinfo("Export", "Export functionality will be implemented next!")

    def export_ocr_results(self):
        """Export OCR results specifically."""
        self.log_message("Exporting OCR results...")
        messagebox.showinfo("Export OCR", "OCR export will be implemented next!")

    def toggle_playback(self):
        """Toggle video playback."""
        self.log_message("Toggling video playback...")

    def stop_playback(self):
        """Stop video playback."""
        self.log_message("Stopping video playback...")

    def open_settings(self):
        """Open settings dialog."""
        messagebox.showinfo("Settings", "Settings dialog will be implemented next!")

    def open_performance_monitor(self):
        """Open performance monitor."""
        messagebox.showinfo("Performance", "Performance monitor will be implemented next!")

    def test_ocr_engine(self):
        """Test OCR engine functionality."""
        self.log_message("Testing OCR engine...")
        messagebox.showinfo("OCR Test", "OCR engine test will be implemented next!")

    def test_video_processor(self):
        """Test video processor functionality."""
        if not self.current_video_file:
            messagebox.showwarning("No Video", "Please import a video file first.")
            return

        self.log_message("Testing video processor...")

        # Test loading video
        success = self.video_processor.load_video(self.current_video_file)
        if not success:
            messagebox.showerror("Error", "Failed to load video for testing")
            return

        # Get metadata
        metadata = self.video_processor.metadata
        if metadata:
            test_results = f"""Video Processor Test Results:

File: {metadata.filename}
Resolution: {metadata.width}x{metadata.height}
Duration: {metadata.duration:.2f} seconds
FPS: {metadata.fps:.2f}
Frame Count: {metadata.frame_count}
Codec: {metadata.codec}
File Size: {metadata.file_size / (1024 * 1024):.1f} MB

Test 1: Extract single frame...
"""

            # Test frame extraction
            test_frame = self.video_processor.extract_frame(100)
            if test_frame:
                test_results += f"✓ Frame 100 extracted successfully\n"
                test_results += f"  Timestamp: {test_frame.timestamp:.2f}s\n"
                test_results += f"  Hash: {test_frame.frame_hash}\n\n"
            else:
                test_results += "✗ Frame extraction failed\n\n"

            test_results += "Test 2: Batch frame extraction...\n"

            # Test batch extraction (first 10 frames)
            batch_frames = self.video_processor.extract_frames_batch(0, 10, 1)
            test_results += f"✓ Extracted {len(batch_frames)} frames in batch\n"

            # Show results
            self.log_message("Video processor test completed")

            # Display in a new window
            self.show_test_results("Video Processor Test", test_results)
        else:
            messagebox.showerror("Error", "No metadata available")

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

        if self.current_video_file and hasattr(self.video_processor, 'metadata') and self.video_processor.metadata:
            metadata = self.video_processor.metadata
            info_text = f"""Project Information:

Video File: {metadata.filename}
Path: {Path(metadata.file_path).parent}
File Size: {metadata.file_size / (1024 * 1024):.1f} MB

Video Properties:
• Resolution: {metadata.width}x{metadata.height}
• Duration: {metadata.duration:.2f} seconds
• FPS: {metadata.fps:.2f}
• Total Frames: {metadata.frame_count}
• Codec: {metadata.codec}
• Bitrate: {metadata.bitrate / 1000:.0f} kbps (estimated)

Processing Status: Ready for OCR analysis
Cache Status: {len(self.video_processor.frame_cache)} frames cached

Next Steps:
1. Start OCR Analysis to extract text
2. Generate translations
3. Create learning materials
4. Export results

Tools Available:
• Test Video Processor (Tools menu)
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
""")

    def exit_application(self):
        """Exit the application."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.logger.info("[UI] Application closing...")
            self.root.quit()

    def run(self):
        """Start the main application loop."""
        self.logger.info("[UI] Starting main application loop...")
        self.root.mainloop()
        self.logger.info("[UI] Application loop ended")