#!/usr/bin/env python3
"""
Skyrim Dialogue Analyzer - Main Entry Point
===========================================

A comprehensive desktop application for extracting, analyzing, and learning from
Skyrim dialogue through advanced OCR and AI-powered language processing.

Built for high-performance gaming PCs with RTX 5080 + Ryzen 7700 + 64GB RAM.

Author: Skyrim Dialogue Analyzer Team
Version: 1.0.0
"""

import sys
import os
import traceback
import logging
from pathlib import Path

# Add src directory to Python path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def setup_logging():
    """Configure comprehensive logging for debugging."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure root logger with UTF-8 encoding
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "skyrim_analyzer.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("[GAME] SKYRIM DIALOGUE ANALYZER STARTING")
    logger.info("=" * 60)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Source path: {src_path}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")

    return logger


def check_dependencies():
    """Check if all required dependencies are available."""
    logger = logging.getLogger(__name__)
    logger.info("[CHECK] Checking dependencies...")

    required_modules = [
        'cv2',  # OpenCV
        'easyocr',  # EasyOCR
        'PIL',  # Pillow
        'numpy',  # NumPy
        'requests',  # Requests
        'yaml',  # PyYAML
        'sqlite3',  # SQLite (built-in)
        'tkinter',  # Tkinter (built-in)
        'ffmpeg'  # FFmpeg-python
    ]

    missing_modules = []

    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
                logger.info(f"[OK] OpenCV version: {cv2.__version__}")
            elif module == 'easyocr':
                import easyocr
                logger.info(f"[OK] EasyOCR available")
            elif module == 'PIL':
                import PIL
                logger.info(f"[OK] Pillow version: {PIL.__version__}")
            elif module == 'numpy':
                import numpy as np
                logger.info(f"[OK] NumPy version: {np.__version__}")
            elif module == 'requests':
                import requests
                logger.info(f"[OK] Requests version: {requests.__version__}")
            elif module == 'yaml':
                import yaml
                logger.info(f"[OK] PyYAML available")
            elif module == 'sqlite3':
                import sqlite3
                logger.info(f"[OK] SQLite3 version: {sqlite3.sqlite_version}")
            elif module == 'tkinter':
                import tkinter as tk
                logger.info(f"[OK] Tkinter version: {tk.TkVersion}")
            elif module == 'ffmpeg':
                import ffmpeg
                logger.info(f"[OK] FFmpeg-python available")

        except ImportError as e:
            logger.error(f"[MISSING] Module: {module} - {e}")
            missing_modules.append(module)

    if missing_modules:
        logger.error(f"[ERROR] Missing dependencies: {missing_modules}")
        logger.error("Please install missing dependencies using:")
        logger.error("pip install -r requirements.txt")
        return False

    logger.info("[SUCCESS] All dependencies satisfied!")
    return True


def check_system_requirements():
    """Check system requirements and GPU availability."""
    logger = logging.getLogger(__name__)
    logger.info("[SYSTEM] Checking system requirements...")

    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024 ** 3)
        logger.info(f"[RAM] Available RAM: {memory_gb:.1f} GB")

        if memory_gb < 16:
            logger.warning("[WARNING] Less than 16GB RAM detected. Performance may be limited.")
        elif memory_gb >= 32:
            logger.info("[EXCELLENT] 32GB+ RAM available for optimal performance")
    except ImportError:
        logger.warning("[WARNING] psutil not available, cannot check memory")

    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[GPU] Available: {gpu_name} (Count: {gpu_count})")
            logger.info(f"[CUDA] Version: {torch.version.cuda}")
        else:
            logger.warning("[WARNING] CUDA not available. OCR will use CPU (slower performance)")
    except ImportError:
        logger.warning("[WARNING] PyTorch not available, cannot check GPU")

    # Check disk space
    try:
        disk_usage = psutil.disk_usage(project_root)
        free_gb = disk_usage.free / (1024 ** 3)
        logger.info(f"[DISK] Available disk space: {free_gb:.1f} GB")

        if free_gb < 10:
            logger.warning("[WARNING] Less than 10GB free space. Consider cleaning up.")
    except:
        logger.warning("[WARNING] Could not check disk space")


def create_project_directories():
    """Create necessary project directories."""
    logger = logging.getLogger(__name__)
    logger.info("[SETUP] Creating project directories...")

    directories = [
        "data/projects",
        "data/cache",
        "data/models",
        "data/backups",
        "output/video_clips",
        "output/screenshots",
        "output/transcripts",
        "output/translations",
        "output/learning_materials",
        "output/reports",
        "logs"
    ]

    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"[DIR] Created/verified: {full_path}")

    logger.info("[SUCCESS] All directories created successfully!")


def load_configuration():
    """Load application configuration."""
    logger = logging.getLogger(__name__)
    logger.info("[CONFIG] Loading configuration...")

    config_file = project_root / "config.ini"

    if not config_file.exists():
        logger.info("[CONFIG] Creating default config.ini...")
        default_config = """[DEFAULT]
# Skyrim Dialogue Analyzer Configuration
app_name = Skyrim Dialogue Analyzer
version = 1.0.0
debug_mode = true

[VIDEO]
# Video processing settings
default_fps = 30
max_resolution = 2560x1440
supported_formats = mkv,mp4,avi

[OCR]
# OCR Engine settings
use_gpu = true
languages = en
confidence_threshold = 0.6
batch_size = 8

[AI]
# AI Translation settings
ollama_endpoint = http://localhost:11434
default_model = llama3.2
translation_language = auto

[UI]
# User interface settings
theme = dark
window_size = 1600x900
auto_save_interval = 300

[PERFORMANCE]
# Performance optimization
max_threads = 8
cache_size_mb = 1024
gpu_memory_limit = 8192
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(default_config)
        logger.info("[SUCCESS] Default configuration created")

    # Load configuration
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file)

    logger.info(f"[SUCCESS] Configuration loaded from {config_file}")
    return config


def main():
    """Main application entry point."""
    try:
        # Setup logging first
        logger = setup_logging()

        logger.info("[INIT] Initializing Skyrim Dialogue Analyzer...")

        # Check dependencies
        if not check_dependencies():
            logger.error("[ERROR] Dependency check failed. Exiting.")
            return 1

        # Check system requirements
        check_system_requirements()

        # Create directories
        create_project_directories()

        # Load configuration
        config = load_configuration()

        logger.info("[IMPORT] Importing core modules...")

        # Import main application (will be created next)
        try:
            from ui.main_application import SkyrimDialogueAnalyzer
            logger.info("[SUCCESS] UI module imported successfully")
        except ImportError as e:
            logger.error(f"[ERROR] Failed to import UI module: {e}")
            logger.error("This is expected if main_application.py doesn't exist yet")
            logger.info("[TEST] Creating placeholder UI for testing...")

            # Create a simple test window for now
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.title("Skyrim Dialogue Analyzer - Test Mode")
            root.geometry("800x600")

            label = tk.Label(root,
                             text="[GAME] Skyrim Dialogue Analyzer\n\nCore system initialized successfully!\nUI components will be implemented next.",
                             font=("Arial", 14), pady=50)
            label.pack(expand=True)

            def show_logs():
                log_file = project_root / "logs" / "skyrim_analyzer.log"
                if log_file.exists():
                    import subprocess
                    subprocess.run(['notepad.exe', str(log_file)], check=False)

            btn_logs = tk.Button(root, text="[LOG] View Logs", command=show_logs, font=("Arial", 12))
            btn_logs.pack(pady=10)

            def close_app():
                logger.info("[CLOSE] Application closing...")
                root.quit()

            btn_close = tk.Button(root, text="[EXIT] Close", command=close_app, font=("Arial", 12))
            btn_close.pack(pady=5)

            logger.info("[UI] Test UI window created")
            logger.info("[START] Starting application main loop...")

            root.protocol("WM_DELETE_WINDOW", close_app)
            root.mainloop()

            return 0

        # If UI module exists, start the real application
        logger.info("[UI] Starting main application...")
        app = SkyrimDialogueAnalyzer(config)
        app.run()

        logger.info("[SUCCESS] Application completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("[INTERRUPT] Application interrupted by user")
        return 0

    except Exception as e:
        logger.error("[CRITICAL] CRITICAL ERROR occurred:")
        logger.error(f"Error: {e}")
        logger.error("Traceback:")
        logger.error(traceback.format_exc())

        # Show error dialog if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Critical Error",
                                 f"Skyrim Dialogue Analyzer encountered a critical error:\n\n{e}\n\nCheck logs for details.")
        except:
            pass

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)