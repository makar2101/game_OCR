#!/usr/bin/env python3
"""
Skyrim Dialogue Analyzer - Video Processing Engine
=================================================

High-performance video processing engine optimized for Skyrim gameplay videos.
Handles frame extraction, metadata analysis, and video segmentation with GPU acceleration.

Optimized for 2K video at 30fps with RTX 5080 + Ryzen 7700 + 64GB RAM.
"""

import cv2
import numpy as np
import threading
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any
import json
from datetime import datetime, timedelta
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib

# Video processing utilities
try:
    import ffmpeg

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    logging.warning("[VIDEO] FFmpeg-python not available, some features may be limited")


@dataclass
class VideoFrame:
    """Represents a single video frame with metadata."""
    frame_number: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int
    frame_hash: str = ""
    is_dialogue_frame: bool = False
    confidence: float = 0.0

    def __post_init__(self):
        """Calculate frame hash after initialization."""
        if self.frame_hash == "":
            self.frame_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate unique hash for frame comparison."""
        # Use a smaller version for hash calculation to improve performance
        small_frame = cv2.resize(self.image, (64, 36))
        return hashlib.md5(small_frame.tobytes()).hexdigest()[:16]


@dataclass
class VideoMetadata:
    """Video file metadata and processing information."""
    file_path: str
    filename: str
    file_size: int
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    bitrate: int
    creation_time: Optional[datetime] = None
    processing_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'filename': self.filename,
            'file_size': self.file_size,
            'duration': self.duration,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'codec': self.codec,
            'bitrate': self.bitrate,
            'creation_time': self.creation_time.isoformat() if self.creation_time else None,
            'processing_time': self.processing_time.isoformat() if self.processing_time else None
        }


class VideoProcessor:
    """High-performance video processing engine for Skyrim dialogue analysis."""

    def __init__(self, max_workers: int = None, cache_size: int = 1000):
        """
        Initialize the video processor.

        Args:
            max_workers: Maximum number of worker threads (default: CPU count)
            cache_size: Maximum number of frames to keep in memory cache
        """
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or mp.cpu_count()
        self.cache_size = cache_size

        # Processing state
        self.current_video = None
        self.metadata = None
        self.is_processing = False
        self.processing_progress = 0.0
        self.cancel_requested = False

        # Frame cache for performance
        self.frame_cache = {}
        self.cache_order = []

        # Callbacks for progress reporting
        self.progress_callback = None
        self.frame_callback = None
        self.completion_callback = None

        # Performance monitoring
        self.processing_stats = {
            'frames_processed': 0,
            'processing_start_time': None,
            'estimated_completion': None,
            'fps_processing_rate': 0.0
        }

        self.logger.info(f"[VIDEO] Video processor initialized with {self.max_workers} workers")

    def set_callbacks(self, progress_callback: Callable = None,
                      frame_callback: Callable = None,
                      completion_callback: Callable = None):
        """Set callback functions for progress reporting."""
        self.progress_callback = progress_callback
        self.frame_callback = frame_callback
        self.completion_callback = completion_callback
        self.logger.info("[VIDEO] Callbacks configured")

    def load_video(self, video_path: str) -> bool:
        """
        Load and analyze video file metadata.

        Args:
            video_path: Path to the video file

        Returns:
            bool: True if video loaded successfully
        """
        try:
            video_path = Path(video_path)
            if not video_path.exists():
                self.logger.error(f"[VIDEO] File not found: {video_path}")
                return False

            self.logger.info(f"[VIDEO] Loading video: {video_path.name}")

            # Open video with OpenCV
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.logger.error(f"[VIDEO] Could not open video file: {video_path}")
                return False

            # Extract basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            # Try to get codec information
            fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])

            cap.release()

            # Get additional metadata with FFmpeg if available
            bitrate = 0
            creation_time = None

            if FFMPEG_AVAILABLE:
                try:
                    probe = ffmpeg.probe(str(video_path))

                    # Get bitrate
                    if 'bit_rate' in probe['format']:
                        bitrate = int(probe['format']['bit_rate'])

                    # Get creation time
                    if 'creation_time' in probe['format']['tags']:
                        creation_time = datetime.fromisoformat(
                            probe['format']['tags']['creation_time'].replace('Z', '+00:00')
                        )
                except Exception as e:
                    self.logger.warning(f"[VIDEO] Could not get extended metadata: {e}")

            # Create metadata object
            self.metadata = VideoMetadata(
                file_path=str(video_path),
                filename=video_path.name,
                file_size=video_path.stat().st_size,
                duration=duration,
                fps=fps,
                frame_count=frame_count,
                width=width,
                height=height,
                codec=codec,
                bitrate=bitrate,
                creation_time=creation_time,
                processing_time=datetime.now()
            )

            self.current_video = str(video_path)

            self.logger.info(f"[VIDEO] Video loaded successfully:")
            self.logger.info(f"  Resolution: {width}x{height}")
            self.logger.info(f"  Duration: {duration:.2f}s ({frame_count} frames)")
            self.logger.info(f"  FPS: {fps:.2f}")
            self.logger.info(f"  Codec: {codec}")
            self.logger.info(f"  File size: {video_path.stat().st_size / (1024 * 1024):.1f} MB")

            return True

        except Exception as e:
            self.logger.error(f"[VIDEO] Error loading video: {e}")
            return False

    def extract_frame(self, frame_number: int) -> Optional[VideoFrame]:
        """
        Extract a specific frame from the video.

        Args:
            frame_number: Frame number to extract (0-based)

        Returns:
            VideoFrame object or None if extraction failed
        """
        if not self.current_video or not self.metadata:
            self.logger.error("[VIDEO] No video loaded")
            return None

        # Check cache first
        if frame_number in self.frame_cache:
            self.logger.debug(f"[VIDEO] Frame {frame_number} retrieved from cache")
            return self.frame_cache[frame_number]

        try:
            cap = cv2.VideoCapture(self.current_video)
            if not cap.isOpened():
                self.logger.error("[VIDEO] Could not open video for frame extraction")
                return None

            # Seek to the desired frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            cap.release()

            if not ret:
                self.logger.warning(f"[VIDEO] Could not read frame {frame_number}")
                return None

            # Calculate timestamp
            timestamp = frame_number / self.metadata.fps

            # Create VideoFrame object
            video_frame = VideoFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                image=frame,
                width=frame.shape[1],
                height=frame.shape[0]
            )

            # Add to cache
            self._add_to_cache(frame_number, video_frame)

            return video_frame

        except Exception as e:
            self.logger.error(f"[VIDEO] Error extracting frame {frame_number}: {e}")
            return None

    def extract_frames_batch(self, start_frame: int, end_frame: int,
                             step: int = 1) -> List[VideoFrame]:
        """
        Extract multiple frames efficiently in batch.

        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (exclusive)
            step: Step size between frames

        Returns:
            List of VideoFrame objects
        """
        if not self.current_video or not self.metadata:
            self.logger.error("[VIDEO] No video loaded")
            return []

        frames = []
        frame_numbers = list(range(start_frame, min(end_frame, self.metadata.frame_count), step))

        self.logger.info(f"[VIDEO] Extracting {len(frame_numbers)} frames in batch")

        try:
            cap = cv2.VideoCapture(self.current_video)
            if not cap.isOpened():
                self.logger.error("[VIDEO] Could not open video for batch extraction")
                return []

            for i, frame_num in enumerate(frame_numbers):
                # Check if we should cancel
                if self.cancel_requested:
                    self.logger.info("[VIDEO] Batch extraction cancelled")
                    break

                # Check cache first
                if frame_num in self.frame_cache:
                    frames.append(self.frame_cache[frame_num])
                    continue

                # Seek and read frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame_image = cap.read()

                if ret:
                    timestamp = frame_num / self.metadata.fps
                    video_frame = VideoFrame(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        image=frame_image,
                        width=frame_image.shape[1],
                        height=frame_image.shape[0]
                    )

                    frames.append(video_frame)
                    self._add_to_cache(frame_num, video_frame)

                    # Report progress
                    if self.progress_callback:
                        progress = (i + 1) / len(frame_numbers) * 100
                        self.progress_callback(progress)

                # Update processing stats
                self.processing_stats['frames_processed'] += 1

            cap.release()

            self.logger.info(f"[VIDEO] Batch extraction completed: {len(frames)} frames extracted")
            return frames

        except Exception as e:
            self.logger.error(f"[VIDEO] Error in batch frame extraction: {e}")
            return frames

    def process_video_async(self, frame_step: int = 30,
                            dialogue_detection: bool = True) -> threading.Thread:
        """
        Process entire video asynchronously with progress reporting.

        Args:
            frame_step: Step size between frames (30 = every second at 30fps)
            dialogue_detection: Whether to perform basic dialogue detection

        Returns:
            Threading.Thread object for the processing task
        """

        def process_worker():
            self.logger.info("[VIDEO] Starting asynchronous video processing")
            self.is_processing = True
            self.cancel_requested = False
            self.processing_stats['processing_start_time'] = time.time()
            self.processing_stats['frames_processed'] = 0

            try:
                if not self.current_video or not self.metadata:
                    self.logger.error("[VIDEO] No video loaded for processing")
                    return

                total_frames = self.metadata.frame_count
                frames_to_process = list(range(0, total_frames, frame_step))

                self.logger.info(f"[VIDEO] Processing {len(frames_to_process)} frames "
                                 f"(every {frame_step} frames)")

                # Process frames in batches for efficiency
                batch_size = min(100, len(frames_to_process))
                processed_frames = []

                for i in range(0, len(frames_to_process), batch_size):
                    if self.cancel_requested:
                        self.logger.info("[VIDEO] Processing cancelled by user")
                        break

                    batch_start = frames_to_process[i]
                    batch_end = frames_to_process[min(i + batch_size, len(frames_to_process) - 1)]

                    # Extract batch of frames
                    batch_frames = self.extract_frames_batch(batch_start, batch_end + 1, frame_step)

                    # Perform dialogue detection if requested
                    if dialogue_detection:
                        batch_frames = self._detect_dialogue_in_batch(batch_frames)

                    processed_frames.extend(batch_frames)

                    # Update progress
                    progress = len(processed_frames) / len(frames_to_process) * 100
                    self.processing_progress = progress

                    if self.progress_callback:
                        self.progress_callback(progress)

                    # Call frame callback for each frame
                    if self.frame_callback:
                        for frame in batch_frames:
                            self.frame_callback(frame)

                    # Update processing rate
                    elapsed_time = time.time() - self.processing_stats['processing_start_time']
                    if elapsed_time > 0:
                        self.processing_stats['fps_processing_rate'] = len(processed_frames) / elapsed_time

                # Processing completed
                self.is_processing = False
                self.processing_progress = 100.0

                processing_time = time.time() - self.processing_stats['processing_start_time']
                self.logger.info(f"[VIDEO] Processing completed in {processing_time:.2f} seconds")
                self.logger.info(f"[VIDEO] Processed {len(processed_frames)} frames")
                self.logger.info(f"[VIDEO] Processing rate: {self.processing_stats['fps_processing_rate']:.1f} fps")

                if self.completion_callback:
                    self.completion_callback(processed_frames)

            except Exception as e:
                self.logger.error(f"[VIDEO] Error during processing: {e}")
                self.is_processing = False

        # Start processing thread
        thread = threading.Thread(target=process_worker, daemon=True)
        thread.start()
        return thread

    def _detect_dialogue_in_batch(self, frames: List[VideoFrame]) -> List[VideoFrame]:
        """
        Perform basic dialogue detection on a batch of frames.
        This is a simplified implementation - can be enhanced later.

        Args:
            frames: List of VideoFrame objects

        Returns:
            List of VideoFrame objects with dialogue detection results
        """
        for frame in frames:
            # Simple dialogue detection based on image analysis
            # This is a placeholder - will be enhanced with proper OCR integration

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

            # Look for text-like regions (high contrast areas)
            # This is a very basic approach
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Count significant contours that might represent text
            significant_contours = [c for c in contours if cv2.contourArea(c) > 50]

            # Simple heuristic: if there are many small contours, might be text
            if len(significant_contours) > 20:
                frame.is_dialogue_frame = True
                frame.confidence = min(len(significant_contours) / 100.0, 1.0)
            else:
                frame.is_dialogue_frame = False
                frame.confidence = 0.0

        return frames

    def _add_to_cache(self, frame_number: int, frame: VideoFrame):
        """Add frame to cache with LRU eviction."""
        if len(self.frame_cache) >= self.cache_size:
            # Remove oldest frame
            oldest_frame = self.cache_order.pop(0)
            del self.frame_cache[oldest_frame]

        self.frame_cache[frame_number] = frame
        self.cache_order.append(frame_number)

    def get_frame_at_timestamp(self, timestamp: float) -> Optional[VideoFrame]:
        """
        Get frame at specific timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            VideoFrame at the specified timestamp
        """
        if not self.metadata:
            return None

        frame_number = int(timestamp * self.metadata.fps)
        return self.extract_frame(frame_number)

    def create_video_segment(self, start_time: float, end_time: float,
                             output_path: str) -> bool:
        """
        Create a video segment from the original video.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path for the output video file

        Returns:
            bool: True if segment created successfully
        """
        if not FFMPEG_AVAILABLE:
            self.logger.error("[VIDEO] FFmpeg not available for video segmentation")
            return False

        if not self.current_video:
            self.logger.error("[VIDEO] No video loaded")
            return False

        try:
            self.logger.info(f"[VIDEO] Creating segment: {start_time}s - {end_time}s")

            # Use FFmpeg to create segment
            (
                ffmpeg
                .input(self.current_video, ss=start_time, t=end_time - start_time)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )

            self.logger.info(f"[VIDEO] Segment created successfully: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"[VIDEO] Error creating video segment: {e}")
            return False

    def cancel_processing(self):
        """Cancel ongoing processing."""
        self.cancel_requested = True
        self.logger.info("[VIDEO] Processing cancellation requested")

    def clear_cache(self):
        """Clear the frame cache."""
        self.frame_cache.clear()
        self.cache_order.clear()
        self.logger.info("[VIDEO] Frame cache cleared")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()
        stats['is_processing'] = self.is_processing
        stats['progress'] = self.processing_progress
        stats['cache_size'] = len(self.frame_cache)
        return stats

    def save_metadata(self, output_path: str):
        """Save video metadata to JSON file."""
        if not self.metadata:
            self.logger.warning("[VIDEO] No metadata to save")
            return

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
            self.logger.info(f"[VIDEO] Metadata saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"[VIDEO] Error saving metadata: {e}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.cancel_processing()
        self.clear_cache()