"""
Core Video Processing Module for Skyrim Dialogue Analyzer
Handles MKV frame extraction, metadata analysis, and video preprocessing
Optimized for RTX 5080 + Ryzen 7700 + 64GB RAM setup

File Structure:
skyrim_dialogue_analyzer/
├── src/core/video_processor.py  # This file
├── test_videos/                 # Put your .mkv files here
├── output/                      # Generated output files
└── logs/                        # Debug logs
"""

import cv2
import numpy as np
import os
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass, asdict
import sys

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_VIDEOS_DIR = PROJECT_ROOT / "test_videos"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create necessary directories
for directory in [TEST_VIDEOS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Configure detailed logging
log_file = LOGS_DIR / "video_processor_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log the directory structure for debugging
logger.info(f"PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"TEST_VIDEOS_DIR: {TEST_VIDEOS_DIR}")
logger.info(f"OUTPUT_DIR: {OUTPUT_DIR}")
logger.info(f"LOGS_DIR: {LOGS_DIR}")


@dataclass
class VideoMetadata:
    """Video metadata container"""
    filepath: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    file_size: int
    creation_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_to_file(self, output_path: str) -> bool:
        """Save metadata to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Metadata saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            return False


@dataclass
class FrameData:
    """Individual frame data container"""
    frame_number: int
    timestamp: float
    frame_array: np.ndarray
    frame_hash: str
    is_changed: bool = False
    text_regions: List[Dict] = None

    def __post_init__(self):
        if self.text_regions is None:
            self.text_regions = []

    def save_frame_image(self, output_dir: str) -> str:
        """Save frame as image file and return path"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"frame_{self.frame_number:06d}_{self.timestamp:.3f}.png"
            filepath = os.path.join(output_dir, filename)

            success = cv2.imwrite(filepath, self.frame_array)
            if success:
                logger.debug(f"Frame saved: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to save frame: {filepath}")
                return ""
        except Exception as e:
            logger.error(f"Error saving frame: {str(e)}")
            return ""


class VideoProcessor:
    """
    High-performance video processor optimized for Skyrim dialogue extraction
    Supports 2K MKV files with GPU acceleration and multi-threading
    """

    def __init__(self,
                 max_workers: int = 8,
                 frame_buffer_size: int = 100,
                 enable_gpu: bool = True,
                 debug_mode: bool = True):
        """
        Initialize VideoProcessor with performance optimizations

        Args:
            max_workers: Number of parallel processing threads
            frame_buffer_size: Maximum frames to keep in memory
            enable_gpu: Whether to use GPU acceleration
            debug_mode: Enable detailed debugging output
        """
        logger.info(f"Initializing VideoProcessor with {max_workers} workers")
        logger.debug(f"Frame buffer size: {frame_buffer_size}")
        logger.debug(f"GPU enabled: {enable_gpu}")

        self.max_workers = max_workers
        self.frame_buffer_size = frame_buffer_size
        self.enable_gpu = enable_gpu
        self.debug_mode = debug_mode

        # Performance tracking
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'average_fps': 0,
            'memory_usage': 0
        }

        # Thread safety
        self._lock = threading.Lock()
        self._frame_cache = {}
        self._last_frame_hash = None

        # Check for available video files
        self._scan_available_videos()

        logger.info("VideoProcessor initialized successfully")

    def _scan_available_videos(self):
        """Scan for available video files in test directory"""
        logger.info(f"Scanning for video files in: {TEST_VIDEOS_DIR}")

        video_extensions = ['.mkv', '.mp4', '.avi', '.mov']
        found_videos = []

        if TEST_VIDEOS_DIR.exists():
            for ext in video_extensions:
                videos = list(TEST_VIDEOS_DIR.glob(f"*{ext}"))
                found_videos.extend(videos)

        logger.info(f"Found {len(found_videos)} video files:")
        for video in found_videos:
            logger.info(f"  - {video.name} ({video.stat().st_size / (1024*1024):.1f} MB)")

        if not found_videos:
            logger.warning(f"No video files found in {TEST_VIDEOS_DIR}")
            logger.warning("Please add .mkv files to the test_videos/ directory")

    def find_video_by_name(self, partial_name: str) -> Optional[str]:
        """Find video file by partial name match"""
        logger.debug(f"Searching for video with name containing: '{partial_name}'")

        if not TEST_VIDEOS_DIR.exists():
            logger.error(f"Test videos directory not found: {TEST_VIDEOS_DIR}")
            return None

        video_extensions = ['.mkv', '.mp4', '.avi', '.mov']

        for ext in video_extensions:
            for video_file in TEST_VIDEOS_DIR.glob(f"*{partial_name}*{ext}"):
                logger.info(f"Found matching video: {video_file}")
                return str(video_file)

        logger.warning(f"No video found matching: '{partial_name}'")
        return None

    def load_video(self, video_path: str) -> Optional[VideoMetadata]:
        """
        Load video file and extract metadata

        Args:
            video_path: Path to the MKV video file OR just filename

        Returns:
            VideoMetadata object or None if failed
        """
        # If only filename provided, look in test_videos directory
        if not os.path.exists(video_path):
            logger.info(f"File not found at {video_path}, searching in test_videos/")
            found_video = self.find_video_by_name(video_path)
            if found_video:
                video_path = found_video
            else:
                logger.error(f"Video file not found: {video_path}")
                return None

        logger.info(f"Loading video: {video_path}")

        try:
            # Open video with OpenCV
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                logger.error("This could be due to:")
                logger.error("  1. Unsupported codec")
                logger.error("  2. Corrupted file")
                logger.error("  3. Missing OpenCV video support")
                return None

            # Extract metadata
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            # Get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])

            # File size
            file_size = os.path.getsize(video_path)

            logger.info(f"Video metadata extracted:")
            logger.info(f"  Resolution: {width}x{height}")
            logger.info(f"  FPS: {fps}")
            logger.info(f"  Total frames: {total_frames}")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Codec: {codec}")
            logger.info(f"  File size: {file_size / (1024 * 1024):.2f} MB")

            # Validate for Skyrim requirements
            if width == 2560 and height == 1440:
                logger.info("✓ Video is 2K resolution (optimal for Skyrim)")
            else:
                logger.warning(f"⚠ Video is not 2K (2560x1440). Actual: {width}x{height}")

            if abs(fps - 30.0) <= 1.0:
                logger.info("✓ Video is ~30 FPS (optimal for Skyrim)")
            else:
                logger.warning(f"⚠ Video is not 30 FPS. Actual: {fps}")

            if video_path.lower().endswith('.mkv'):
                logger.info("✓ Video is MKV format (optimal)")
            else:
                logger.warning(f"⚠ Video is not MKV format")

            metadata = VideoMetadata(
                filepath=video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec,
                file_size=file_size
            )

            cap.release()

            # Save metadata to output directory
            metadata_file = OUTPUT_DIR / f"{Path(video_path).stem}_metadata.json"
            metadata.save_to_file(str(metadata_file))

            logger.info("Video loaded successfully")
            return metadata

        except Exception as e:
            logger.error(f"Error loading video: {str(e)}", exc_info=True)
            return None

    def extract_frame(self, video_path: str, frame_number: int) -> Optional[FrameData]:
        """
        Extract a specific frame from video

        Args:
            video_path: Path to video file
            frame_number: Frame number to extract

        Returns:
            FrameData object or None if failed
        """
        logger.debug(f"Extracting frame {frame_number} from {Path(video_path).name}")

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Failed to open video for frame extraction: {video_path}")
                return None

            # Validate frame number
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_number >= total_frames:
                logger.warning(f"Frame {frame_number} is beyond video length ({total_frames} frames)")
                cap.release()
                return None

            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_number}")
                cap.release()
                return None

            # Calculate timestamp
            fps = cap.get(cv2.CAP_PROP_FPS)
            timestamp = frame_number / fps if fps > 0 else 0

            # Generate frame hash for change detection
            frame_hash = self._calculate_frame_hash(frame)

            # Check if frame changed from previous
            is_changed = self._is_frame_changed(frame_hash)

            logger.debug(f"Frame {frame_number} extracted successfully")
            logger.debug(f"  Timestamp: {timestamp:.3f}s")
            logger.debug(f"  Hash: {frame_hash[:16]}...")
            logger.debug(f"  Changed: {is_changed}")

            frame_data = FrameData(
                frame_number=frame_number,
                timestamp=timestamp,
                frame_array=frame,
                frame_hash=frame_hash,
                is_changed=is_changed
            )

            cap.release()
            return frame_data

        except Exception as e:
            logger.error(f"Error extracting frame {frame_number}: {str(e)}", exc_info=True)
            return None

    def extract_frames_batch(self,
                             video_path: str,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None,
                             step: int = 1) -> List[FrameData]:
        """
        Extract multiple frames in parallel

        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all)
            step: Frame step size

        Returns:
            List of FrameData objects
        """
        logger.info(f"Starting batch frame extraction")
        logger.info(f"  Video: {Path(video_path).name}")
        logger.info(f"  Range: {start_frame} to {end_frame or 'end'}")
        logger.info(f"  Step: {step}")

        start_time = time.time()

        # Get video metadata for validation
        metadata = self.load_video(video_path)
        if not metadata:
            logger.error("Failed to load video metadata")
            return []

        if end_frame is None:
            end_frame = metadata.total_frames

        # Validate range
        end_frame = min(end_frame, metadata.total_frames)
        frame_numbers = list(range(start_frame, end_frame, step))

        logger.info(f"Processing {len(frame_numbers)} frames with {self.max_workers} workers")

        frames = []
        processed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all frame extraction tasks
            future_to_frame = {
                executor.submit(self.extract_frame, video_path, frame_num): frame_num
                for frame_num in frame_numbers
            }

            # Process completed tasks
            for future in as_completed(future_to_frame):
                frame_num = future_to_frame[future]

                try:
                    frame_data = future.result()
                    if frame_data:
                        frames.append(frame_data)
                        processed_count += 1

                        if processed_count % 100 == 0:
                            elapsed = time.time() - start_time
                            fps = processed_count / elapsed
                            logger.info(f"Processed {processed_count}/{len(frame_numbers)} frames, {fps:.2f} FPS")

                except Exception as e:
                    logger.error(f"Error processing frame {frame_num}: {str(e)}")

        # Sort frames by frame number
        frames.sort(key=lambda x: x.frame_number)

        total_time = time.time() - start_time
        avg_fps = len(frames) / total_time if total_time > 0 else 0

        logger.info(f"Batch extraction completed:")
        logger.info(f"  Frames extracted: {len(frames)}")
        logger.info(f"  Processing time: {total_time:.2f}s")
        logger.info(f"  Average FPS: {avg_fps:.2f}")

        # Update stats
        self.processing_stats['frames_processed'] += len(frames)
        self.processing_stats['total_processing_time'] += total_time
        self.processing_stats['average_fps'] = avg_fps

        return frames

    def _calculate_frame_hash(self, frame: np.ndarray) -> str:
        """Calculate hash for frame change detection"""
        try:
            # Convert to grayscale for hash calculation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Resize to small size for faster hashing
            resized = cv2.resize(gray, (64, 64))

            # Calculate hash
            frame_hash = hash(resized.tobytes())
            return str(frame_hash)

        except Exception as e:
            logger.error(f"Error calculating frame hash: {str(e)}")
            return "0"

    def _is_frame_changed(self, frame_hash: str) -> bool:
        """Check if frame changed from previous"""
        with self._lock:
            if self._last_frame_hash is None:
                self._last_frame_hash = frame_hash
                return True

            changed = frame_hash != self._last_frame_hash
            self._last_frame_hash = frame_hash
            return changed

    def preprocess_frame_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better OCR accuracy

        Args:
            frame: Input frame array

        Returns:
            Preprocessed frame
        """
        logger.debug("Preprocessing frame for OCR")

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            # Apply threshold for better text visibility
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            logger.debug("Frame preprocessing completed")
            return thresh

        except Exception as e:
            logger.error(f"Error preprocessing frame: {str(e)}")
            return frame

    def save_frames_to_output(self, frames: List[FrameData], subfolder: str = "frames") -> List[str]:
        """
        Save all frames to output directory

        Args:
            frames: List of FrameData objects
            subfolder: Subfolder name in output directory

        Returns:
            List of saved file paths
        """
        output_subdir = OUTPUT_DIR / subfolder
        output_subdir.mkdir(exist_ok=True)

        logger.info(f"Saving {len(frames)} frames to {output_subdir}")

        saved_paths = []
        for frame in frames:
            saved_path = frame.save_frame_image(str(output_subdir))
            if saved_path:
                saved_paths.append(saved_path)

        logger.info(f"Saved {len(saved_paths)} frames successfully")
        return saved_paths

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()

    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'average_fps': 0,
            'memory_usage': 0
        }
        logger.info("Processing statistics reset")


# Enhanced test function with file discovery
def test_video_processor():
    """Test function with extensive debugging and automatic file discovery"""
    logger.info("=" * 60)
    logger.info("Starting VideoProcessor test with file discovery")
    logger.info("=" * 60)

    # Initialize processor
    processor = VideoProcessor(
        max_workers=4,
        frame_buffer_size=50,
        enable_gpu=True,
        debug_mode=True
    )

    # Try to find any video file automatically
    video_files = []
    if TEST_VIDEOS_DIR.exists():
        for ext in ['.mkv', '.mp4', '.avi', '.mov']:
            video_files.extend(list(TEST_VIDEOS_DIR.glob(f"*{ext}")))

    if video_files:
        # Use the first video found
        test_video_path = str(video_files[0])
        logger.info(f"Using video file: {test_video_path}")

        # Load video metadata
        metadata = processor.load_video(test_video_path)

        if metadata:
            logger.info("✓ Metadata loaded successfully")

            # Extract first 30 frames (1 second at 30fps)
            logger.info("Extracting first 30 frames for testing...")
            frames = processor.extract_frames_batch(
                test_video_path,
                start_frame=0,
                end_frame=30,
                step=1
            )

            logger.info(f"✓ Extracted {len(frames)} frames")

            # Save frames to output directory
            if frames:
                saved_paths = processor.save_frames_to_output(frames, "test_frames")
                logger.info(f"✓ Saved {len(saved_paths)} frame images")

            # Get processing stats
            stats = processor.get_processing_stats()
            logger.info(f"Processing stats: {stats}")

        else:
            logger.error("✗ Failed to load metadata")
    else:
        logger.warning("No video files found for testing")
        logger.warning(f"Please add video files to: {TEST_VIDEOS_DIR}")
        logger.warning("Supported formats: .mkv, .mp4, .avi, .mov")

    logger.info("=" * 60)
    logger.info("VideoProcessor test completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_video_processor()