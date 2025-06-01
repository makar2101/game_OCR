#!/usr/bin/env python3
"""
Helper script to add video files to test_videos directory
Run this to easily add a video file for testing
"""

import os
import shutil
from pathlib import Path


def find_and_copy_video():
    """Find video files and copy one to test_videos"""

    # Get project directories
    project_root = Path(__file__).parent
    test_videos_dir = project_root / "test_videos"

    # Create test_videos directory
    test_videos_dir.mkdir(exist_ok=True)

    print("Video File Helper for Skyrim Dialogue Analyzer")
    print("=" * 50)
    print(f"Test videos directory: {test_videos_dir}")

    # Check if we already have videos
    existing_videos = []
    for ext in ['.mkv', '.mp4', '.avi', '.mov']:
        existing_videos.extend(list(test_videos_dir.glob(f"*{ext}")))

    if existing_videos:
        print(f"\nFound {len(existing_videos)} existing video(s):")
        for video in existing_videos:
            print(f"  - {video.name} ({video.stat().st_size / (1024 * 1024):.1f} MB)")
        print("\nYou can run the test now with: python simple_test.py")
        return

    print("\nNo videos found in test_videos/")
    print("\nPlease do ONE of the following:")
    print("\n1. MANUAL COPY (Recommended):")
    print("   - Copy any video file to the test_videos/ folder")
    print("   - Supported formats: .mkv, .mp4, .avi, .mov")
    print(f"   - Full path: {test_videos_dir.absolute()}")

    print("\n2. DRAG AND DROP:")
    print("   - Drag your video file to the test_videos/ folder in Windows Explorer")

    print("\n3. COMMAND LINE:")
    print('   copy "C:\\path\\to\\your\\video.mp4" "test_videos\\"')

    print("\n" + "=" * 50)
    print("After adding a video, run: python simple_test.py")


if __name__ == "__main__":
    find_and_copy_video()