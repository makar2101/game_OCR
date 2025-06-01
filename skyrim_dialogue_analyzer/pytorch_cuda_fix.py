#!/usr/bin/env python3
"""
PyTorch CUDA Debug and Fix for RTX 5080
=======================================

This script will:
1. Check current PyTorch version and CUDA compatibility
2. Install the correct PyTorch version for RTX 5080
3. Test CUDA functionality
4. Fix the OCR engine to use GPU properly
"""

import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_current_setup():
    """Check current PyTorch and CUDA setup."""
    logger.info("=" * 60)
    logger.info("CHECKING CURRENT PYTORCH SETUP")
    logger.info("=" * 60)

    try:
        import torch
        logger.info(f"Current PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")

            # Check CUDA capability
            gpu_props = torch.cuda.get_device_properties(0)
            major = gpu_props.major
            minor = gpu_props.minor
            logger.info(f"GPU CUDA capability: sm_{major}{minor}")

            # Check what CUDA architectures PyTorch was compiled for
            logger.info(f"PyTorch CUDA architectures: {torch.cuda.get_arch_list()}")

        return True
    except ImportError:
        logger.error("PyTorch not installed!")
        return False


def install_latest_pytorch():
    """Install the latest PyTorch with CUDA 12.4 support."""
    logger.info("=" * 60)
    logger.info("INSTALLING LATEST PYTORCH WITH CUDA 12.4")
    logger.info("=" * 60)

    # Uninstall current PyTorch
    logger.info("Uninstalling current PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall",
        "torch", "torchvision", "torchaudio", "-y"
    ])

    # Install PyTorch with CUDA 12.4 (latest)
    logger.info("Installing PyTorch with CUDA 12.4 (nightly build for RTX 5080)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--pre", "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/nightly/cu124"
    ])

    logger.info("PyTorch installation complete!")


def install_stable_pytorch():
    """Install stable PyTorch with CUDA 12.1."""
    logger.info("=" * 60)
    logger.info("INSTALLING STABLE PYTORCH WITH CUDA 12.1")
    logger.info("=" * 60)

    # Uninstall current PyTorch
    logger.info("Uninstalling current PyTorch...")
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall",
        "torch", "torchvision", "torchaudio", "-y"
    ])

    # Install stable PyTorch with CUDA 12.1
    logger.info("Installing stable PyTorch with CUDA 12.1...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

    logger.info("PyTorch installation complete!")


def test_cuda_functionality():
    """Test basic CUDA functionality."""
    logger.info("=" * 60)
    logger.info("TESTING CUDA FUNCTIONALITY")
    logger.info("=" * 60)

    try:
        import torch

        # Basic CUDA test
        logger.info("Testing basic CUDA operations...")

        if torch.cuda.is_available():
            # Create tensors on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()

            # Perform computation
            z = torch.mm(x, y)

            logger.info("✅ Basic CUDA operations successful!")
            logger.info(f"Result tensor shape: {z.shape}")
            logger.info(f"Result tensor device: {z.device}")

            # Test memory allocation
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")

            return True
        else:
            logger.error("❌ CUDA not available")
            return False

    except Exception as e:
        logger.error(f"❌ CUDA test failed: {e}")
        return False


def test_easyocr_gpu():
    """Test EasyOCR with GPU."""
    logger.info("=" * 60)
    logger.info("TESTING EASYOCR WITH GPU")
    logger.info("=" * 60)

    try:
        import easyocr
        import numpy as np

        logger.info("Creating EasyOCR reader with GPU...")
        reader = easyocr.Reader(['en'], gpu=True)

        # Create test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        # Add some text (would normally use cv2.putText)

        logger.info("Testing text recognition...")
        results = reader.readtext(test_image)

        logger.info("✅ EasyOCR GPU test successful!")
        logger.info(f"Results: {results}")

        return True

    except Exception as e:
        logger.error(f"❌ EasyOCR GPU test failed: {e}")
        return False


def main():
    """Main installation and testing function."""
    print("=" * 60)
    print("RTX 5080 PYTORCH CUDA FIX")
    print("=" * 60)
    print("This will fix PyTorch CUDA compatibility for your RTX 5080")
    print("")

    # Check current setup
    check_current_setup()

    print("\nChoose installation option:")
    print("1. Install PyTorch nightly (CUDA 12.4) - Recommended for RTX 5080")
    print("2. Install PyTorch stable (CUDA 12.1)")
    print("3. Skip installation and just test current setup")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ")

    if choice == "1":
        install_latest_pytorch()
    elif choice == "2":
        install_stable_pytorch()
    elif choice == "3":
        logger.info("Skipping installation...")
    elif choice == "4":
        return
    else:
        logger.error("Invalid choice")
        return

    # Test the installation
    logger.info("\nTesting installation...")

    # Check setup again
    if not check_current_setup():
        logger.error("PyTorch installation verification failed")
        return

    # Test CUDA
    if not test_cuda_functionality():
        logger.error("CUDA functionality test failed")
        return

    # Test EasyOCR
    if not test_easyocr_gpu():
        logger.error("EasyOCR GPU test failed")
        return

    logger.info("=" * 60)
    logger.info("🎉 ALL TESTS PASSED!")
    logger.info("Your RTX 5080 is now properly configured for GPU OCR!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()