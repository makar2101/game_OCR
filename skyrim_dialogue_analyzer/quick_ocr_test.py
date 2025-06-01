#!/usr/bin/env python3
"""
Quick OCR Engine Test
====================

Quick test to verify OCR engine is working correctly.
Run this after installing dependencies.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported."""
    logger.info("Testing imports...")

    try:
        import torch
        import easyocr
        import cv2
        import numpy as np
        logger.info("✓ All core dependencies imported successfully")

        # Test CUDA availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info(f"✓ CUDA available: {gpu_name} (CUDA {cuda_version})")
        else:
            logger.warning("⚠ CUDA not available - will use CPU")

        return True

    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_ocr_engine():
    """Test OCR engine initialization and basic functionality."""
    logger.info("Testing OCR engine...")

    try:
        from core.ocr_engine import OCREngine, BoundingBox, TextSegment
        from core.video_processor import FrameData

        logger.info("✓ OCR engine modules imported successfully")

        # Initialize OCR engine
        logger.info("Initializing OCR engine...")
        ocr_engine = OCREngine(
            gpu_enabled=True,
            languages=['en'],
            confidence_threshold=0.6,
            max_workers=2
        )
        logger.info("✓ OCR engine initialized successfully")

        # Create test frame
        import numpy as np
        import cv2

        test_frame = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
        cv2.putText(test_frame, "Hello, Skyrim!", (400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)
        cv2.putText(test_frame, "Dragonborn", (400, 800),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 100), 3)

        frame_data = FrameData(
            frame_number=1,
            timestamp=0.033,
            frame_array=test_frame,
            frame_hash="test_hash"
        )

        # Process frame
        logger.info("Processing test frame...")
        result = ocr_engine.process_frame(frame_data)

        logger.info(f"✓ Frame processed successfully")
        logger.info(f"  Processing time: {result.processing_time:.3f}s")
        logger.info(f"  Text segments found: {len(result.text_segments)}")
        logger.info(f"  Dialogue detected: {result.dialogue_detected}")

        if result.text_segments:
            logger.info("  Text found:")
            for i, segment in enumerate(result.text_segments):
                logger.info(f"    {i + 1}. '{segment.cleaned_text}' (confidence: {segment.confidence:.3f})")

        # Get processing stats
        stats = ocr_engine.get_processing_stats()
        logger.info(f"✓ Processing stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"✗ OCR engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_integration():
    """Test integration with video processor."""
    logger.info("Testing video processor integration...")

    try:
        from core.video_processor import VideoProcessor
        from core.ocr_engine import OCREngine

        logger.info("✓ Video processor integration modules imported")

        # This would normally test with actual video files
        # For now, just verify the classes work together
        video_processor = VideoProcessor(max_workers=2)
        ocr_engine = OCREngine(max_workers=2)

        logger.info("✓ Video processor and OCR engine initialized together")

        return True

    except Exception as e:
        logger.error(f"✗ Video integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("SKYRIM DIALOGUE ANALYZER - QUICK OCR TEST")
    logger.info("=" * 60)

    all_passed = True

    # Test 1: Imports
    logger.info("\n[TEST 1] Testing imports...")
    if not test_imports():
        all_passed = False

    # Test 2: OCR Engine
    logger.info("\n[TEST 2] Testing OCR engine...")
    if not test_ocr_engine():
        all_passed = False

    # Test 3: Video Integration
    logger.info("\n[TEST 3] Testing video integration...")
    if not test_video_integration():
        all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! OCR Engine is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python main.py (to test with the full UI)")
        logger.info("2. Import a video file and try OCR analysis")
        logger.info("3. Check the OCR results panel for extracted text")
    else:
        logger.error("❌ SOME TESTS FAILED! Check the errors above.")
        logger.error("\nTroubleshooting:")
        logger.error("1. Make sure all dependencies are installed")
        logger.error("2. Check that CUDA drivers are up to date")
        logger.error("3. Verify EasyOCR can access the GPU")

    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)