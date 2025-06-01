#!/usr/bin/env python3
"""
Skyrim Dialogue Analyzer - Dependency Installer
==============================================

Automatically installs all required dependencies for the Skyrim Dialogue Analyzer.
This script handles both CPU and GPU installations with proper error handling.
"""

import sys
import subprocess
import os
import logging
from pathlib import Path


def setup_logging():
    """Setup logging for installation process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_pip_command(command, logger):
    """Run a pip command with proper error handling."""
    try:
        logger.info(f"[INSTALL] Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info(f"[SUCCESS] Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Command failed with return code {e.returncode}")
        logger.error(f"[ERROR] stdout: {e.stdout}")
        logger.error(f"[ERROR] stderr: {e.stderr}")
        return False


def check_gpu_support():
    """Check if NVIDIA GPU is available for CUDA support."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    return False


def install_basic_dependencies(logger):
    """Install basic Python dependencies."""
    logger.info("[PHASE 1] Installing basic dependencies...")

    basic_packages = [
        "numpy>=1.21.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "requests>=2.25.0",
        "tqdm>=4.62.0",
        "python-dateutil>=2.8.0",
        "pathlib2>=2.3.6"
    ]

    for package in basic_packages:
        command = [sys.executable, "-m", "pip", "install", package]
        if not run_pip_command(command, logger):
            logger.error(f"[ERROR] Failed to install {package}")
            return False

    logger.info("[SUCCESS] Basic dependencies installed")
    return True


def install_opencv(logger):
    """Install OpenCV with proper configuration."""
    logger.info("[PHASE 2] Installing OpenCV...")

    # Try opencv-python first (CPU version)
    command = [sys.executable, "-m", "pip", "install", "opencv-python>=4.5.0"]
    if run_pip_command(command, logger):
        logger.info("[SUCCESS] OpenCV installed successfully")
        return True
    else:
        logger.error("[ERROR] Failed to install OpenCV")
        return False


def install_easyocr(logger, gpu_support=False):
    """Install EasyOCR with appropriate torch backend."""
    logger.info("[PHASE 3] Installing EasyOCR and PyTorch...")

    if gpu_support:
        logger.info("[GPU] Installing PyTorch with CUDA support...")
        # Install PyTorch with CUDA support first
        torch_command = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        if not run_pip_command(torch_command, logger):
            logger.warning("[WARNING] Failed to install CUDA PyTorch, falling back to CPU version")
            gpu_support = False

    if not gpu_support:
        logger.info("[CPU] Installing PyTorch CPU version...")
        torch_command = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        if not run_pip_command(torch_command, logger):
            logger.error("[ERROR] Failed to install PyTorch")
            return False

    # Install EasyOCR
    easyocr_command = [sys.executable, "-m", "pip", "install", "easyocr>=1.6.0"]
    if run_pip_command(easyocr_command, logger):
        logger.info("[SUCCESS] EasyOCR installed successfully")
        return True
    else:
        logger.error("[ERROR] Failed to install EasyOCR")
        return False


def install_video_processing(logger):
    """Install video processing dependencies."""
    logger.info("[PHASE 4] Installing video processing libraries...")

    packages = [
        "ffmpeg-python>=0.2.0"
    ]

    for package in packages:
        command = [sys.executable, "-m", "pip", "install", package]
        if not run_pip_command(command, logger):
            logger.error(f"[ERROR] Failed to install {package}")
            return False

    logger.info("[SUCCESS] Video processing libraries installed")
    return True


def install_development_tools(logger):
    """Install development and testing tools."""
    logger.info("[PHASE 5] Installing development tools...")

    dev_packages = [
        "pytest>=6.0.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
        "psutil"  # For system monitoring
    ]

    for package in dev_packages:
        command = [sys.executable, "-m", "pip", "install", package]
        if not run_pip_command(command, logger):
            logger.warning(f"[WARNING] Failed to install {package} (non-critical)")

    logger.info("[SUCCESS] Development tools installed")
    return True


def verify_installation(logger):
    """Verify that all packages are installed correctly."""
    logger.info("[VERIFY] Verifying installation...")

    test_imports = {
        'cv2': 'OpenCV',
        'easyocr': 'EasyOCR',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'requests': 'Requests',
        'yaml': 'PyYAML',
        'ffmpeg': 'FFmpeg-python',
        'torch': 'PyTorch'
    }

    failed_imports = []

    for module, name in test_imports.items():
        try:
            __import__(module)
            logger.info(f"[OK] {name} imported successfully")
        except ImportError as e:
            logger.error(f"[FAIL] {name} import failed: {e}")
            failed_imports.append(name)

    if failed_imports:
        logger.error(f"[ERROR] Failed imports: {failed_imports}")
        return False

    logger.info("[SUCCESS] All packages verified successfully!")
    return True


def check_ffmpeg_binary():
    """Check if FFmpeg binary is available in system PATH."""
    logger = logging.getLogger(__name__)
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("[OK] FFmpeg binary found in system PATH")
            return True
    except FileNotFoundError:
        pass

    logger.warning("[WARNING] FFmpeg binary not found in system PATH")
    logger.warning("Please download FFmpeg from: https://ffmpeg.org/download.html")
    logger.warning("And add it to your system PATH for video processing features")
    return False


def main():
    """Main installation function."""
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("[INSTALLER] Skyrim Dialogue Analyzer Dependency Installer")
    logger.info("=" * 60)

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("[ERROR] Python 3.8+ required. Current version: {}.{}".format(
            sys.version_info.major, sys.version_info.minor))
        return 1

    logger.info(f"[PYTHON] Version: {sys.version}")

    # Check for GPU support
    gpu_support = check_gpu_support()
    if gpu_support:
        logger.info("[GPU] NVIDIA GPU detected - will install CUDA support")
    else:
        logger.info("[CPU] No NVIDIA GPU detected - will install CPU-only versions")

    # Install packages in phases
    success = True

    # Phase 1: Basic dependencies
    if not install_basic_dependencies(logger):
        success = False

    # Phase 2: OpenCV
    if success and not install_opencv(logger):
        success = False

    # Phase 3: EasyOCR and PyTorch
    if success and not install_easyocr(logger, gpu_support):
        success = False

    # Phase 4: Video processing
    if success and not install_video_processing(logger):
        success = False

    # Phase 5: Development tools
    if success:
        install_development_tools(logger)  # Non-critical

    # Verify installation
    if success:
        success = verify_installation(logger)

    # Check FFmpeg binary
    check_ffmpeg_binary()

    logger.info("=" * 60)
    if success:
        logger.info("[SUCCESS] Installation completed successfully!")
        logger.info("You can now run: python main.py")
    else:
        logger.error("[ERROR] Installation failed. Please check the errors above.")
        logger.error("You may need to install some packages manually.")
    logger.info("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    input("\nPress Enter to exit...")
    sys.exit(exit_code)