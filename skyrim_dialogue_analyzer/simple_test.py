#!/usr/bin/env python3
"""
Simple Test Script for VideoProcessor
Only uses videos from test_videos/ directory
Save this as: simple_test.py in skyrim_dialogue_analyzer/ directory
"""

import sys
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from core.video_processor import VideoProcessor, logger, TEST_VIDEOS_DIR, OUTPUT_DIR


def test_video_processor_simple():
    """Simple test that only uses videos from test_videos/ directory"""
    logger.info("=" * 60)
    logger.info("SIMPLE VIDEO PROCESSOR TEST")
    logger.info("=" * 60)

    # Check for videos in test_videos directory
    video_files = []
    if TEST_VIDEOS_DIR.exists():
        for ext in ['.mkv', '.mp4', '.avi', '.mov']:
            video_files.extend(list(TEST_VIDEOS_DIR.glob(f"*{ext}")))

    if not video_files:
        logger.error("[X] No video files found in test_videos/")
        logger.error("Please add a video file to test_videos/ directory")
        logger.error("Supported formats: .mkv, .mp4, .avi, .mov")
        return False

    # Use the first video found
    test_video = video_files[0]
    logger.info(f"[OK] Found video: {test_video.name}")
    logger.info(f"[PATH] Full path: {test_video}")

    # Initialize processor
    processor = VideoProcessor(
        max_workers=4,
        frame_buffer_size=50,
        enable_gpu=True,
        debug_mode=True
    )

    # Test 1: Load video metadata
    logger.info("\n" + "=" * 40)
    logger.info("TEST 1: Loading video metadata")
    logger.info("=" * 40)

    metadata = processor.load_video(str(test_video))
    if not metadata:
        logger.error("[X] Failed to load video metadata")
        return False

    logger.info("[OK] Video metadata loaded successfully")

    # Test 2: Extract first 10 frames
    logger.info("\n" + "=" * 40)
    logger.info("TEST 2: Extracting frames")
    logger.info("=" * 40)

    # Extract first 10 frames for testing
    frames = processor.extract_frames_batch(
        str(test_video),
        start_frame=0,
        end_frame=10,
        step=1
    )

    if not frames:
        logger.error("[X] Failed to extract frames")
        return False

    logger.info(f"[OK] Extracted {len(frames)} frames successfully")

    # Test 3: Save frames to output
    logger.info("\n" + "=" * 40)
    logger.info("TEST 3: Saving frames")
    logger.info("=" * 40)

    saved_paths = processor.save_frames_to_output(frames, "simple_test")

    if saved_paths:
        logger.info(f"[OK] Saved {len(saved_paths)} frames")
        logger.info(f"[PATH] Output directory: {OUTPUT_DIR / 'simple_test'}")
    else:
        logger.warning("[!] No frames were saved")

    # Test 4: Show processing stats
    logger.info("\n" + "=" * 40)
    logger.info("TEST 4: Processing statistics")
    logger.info("=" * 40)

    stats = processor.get_processing_stats()
    logger.info("[STATS] Processing statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")

    logger.info("\n" + "=" * 60)
    logger.info("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"[PATH] Check results in: {OUTPUT_DIR}")

    return True


if __name__ == "__main__":
    success = test_video_processor_simple()
    if success:
        print("\n[OK] Video processor is working perfectly!")
        print(f"[PATH] Check your results in the output/ directory")
    else:
        print("\n[X] Test failed. Make sure you have a video file in test_videos/")
        print("   Supported formats: .mkv, .mp4, .avi, .mov")