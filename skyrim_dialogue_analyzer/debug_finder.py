#!/usr/bin/env python3
"""
Debug script to find out why video files aren't being detected
"""

import os
from pathlib import Path


def debug_video_detection():
    """Debug video file detection"""
    print("=" * 60)
    print("DEBUG: Video File Detection")
    print("=" * 60)

    # Get project root
    project_root = Path(__file__).parent
    test_videos_dir = project_root / "test_videos"

    print(f"Project root: {project_root}")
    print(f"Test videos dir: {test_videos_dir}")
    print(f"Test videos dir exists: {test_videos_dir.exists()}")

    if test_videos_dir.exists():
        print(f"Test videos dir absolute path: {test_videos_dir.absolute()}")

        # List ALL files in test_videos directory
        all_files = list(test_videos_dir.iterdir())
        print(f"\nALL files in test_videos: {len(all_files)}")

        for i, file in enumerate(all_files):
            print(f"  {i + 1}. {file.name}")
            print(f"      - Is file: {file.is_file()}")
            print(f"      - Size: {file.stat().st_size if file.is_file() else 'N/A'} bytes")
            print(f"      - Extension: {file.suffix}")

        # Check specific extensions
        print(f"\n" + "=" * 40)
        print("CHECKING SPECIFIC EXTENSIONS:")

        extensions = ['.mkv', '.mp4', '.avi', '.mov']

        for ext in extensions:
            matching_files = list(test_videos_dir.glob(f"*{ext}"))
            print(f"\n{ext} files: {len(matching_files)}")
            for file in matching_files:
                print(f"  - {file.name}")

        # Check case-insensitive
        print(f"\n" + "=" * 40)
        print("CASE-INSENSITIVE CHECK:")

        for ext in extensions:
            # Check both lowercase and uppercase
            lower_files = list(test_videos_dir.glob(f"*{ext.lower()}"))
            upper_files = list(test_videos_dir.glob(f"*{ext.upper()}"))

            all_ext_files = lower_files + upper_files
            print(f"\n{ext} (case-insensitive): {len(all_ext_files)}")
            for file in all_ext_files:
                print(f"  - {file.name}")

        # Manual check for the specific file we saw
        specific_file = test_videos_dir / "2025-05-31 18-44-15.mkv"
        print(f"\n" + "=" * 40)
        print("SPECIFIC FILE CHECK:")
        print(f"Looking for: {specific_file}")
        print(f"File exists: {specific_file.exists()}")

        if specific_file.exists():
            print(f"File size: {specific_file.stat().st_size} bytes")
            print(f"File extension: '{specific_file.suffix}'")
            print(f"Is file: {specific_file.is_file()}")

    else:
        print("ERROR: test_videos directory doesn't exist!")

        # Try to create it
        print("Attempting to create test_videos directory...")
        try:
            test_videos_dir.mkdir(exist_ok=True)
            print(f"Created: {test_videos_dir}")
        except Exception as e:
            print(f"Failed to create directory: {e}")


if __name__ == "__main__":
    debug_video_detection()