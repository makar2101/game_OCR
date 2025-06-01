#!/usr/bin/env python3
"""
OCR Engine for Skyrim Dialogue Analyzer - GPU ONLY VERSION
==========================================================

High-performance OCR engine optimized for Skyrim dialogue extraction.
Uses EasyOCR with GPU acceleration for real-time text recognition.
This version FORCES GPU usage and provides detailed debugging.

Optimized for RTX 5080 + Ryzen 7700 + 64GB RAM setup.

Features:
- FORCED GPU-accelerated EasyOCR
- Detailed CUDA debugging
- Skyrim-specific UI element detection
- Batch processing with multi-threading
- Confidence-based filtering
- Text region optimization
- Performance monitoring

Author: Skyrim Dialogue Analyzer Team
"""

import cv2
import numpy as np
import easyocr
import logging
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import torch
import os

# Set environment variables for better CUDA debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Import project modules
try:
    from .video_processor import FrameData
except ImportError:
    # Fallback for standalone testing
    from dataclasses import dataclass


    @dataclass
    class FrameData:
        frame_number: int
        timestamp: float
        frame_array: np.ndarray
        frame_hash: str
        is_changed: bool = False

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for text regions."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    region_type: str = "dialogue"  # dialogue, menu, subtitle, character_name

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class TextSegment:
    """Individual text segment with metadata."""
    text: str
    confidence: float
    bounding_box: BoundingBox
    language: str = "en"
    cleaned_text: str = ""
    is_dialogue: bool = False
    character_name: Optional[str] = None

    def __post_init__(self):
        if not self.cleaned_text:
            self.cleaned_text = self._clean_text(self.text)

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())

        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\.\,\!\?\:\;\-\'\"]', '', cleaned)

        # Fix common OCR mistakes for Skyrim
        replacements = {
            'Draqonborn': 'Dragonborn',
            'Skynm': 'Skyrim',
            'Whrterun': 'Whiterun',
            'Solrtude': 'Solitude',
            'Wrndhelm': 'Windhelm',
            '|': 'I',  # Common OCR mistake
            '0': 'O',  # In names/dialogue
        }

        for wrong, correct in replacements.items():
            cleaned = cleaned.replace(wrong, correct)

        return cleaned


@dataclass
class OCRResult:
    """Complete OCR result for a frame."""
    frame_number: int
    timestamp: float
    frame_hash: str
    text_segments: List[TextSegment]
    processing_time: float
    total_confidence: float
    dialogue_detected: bool = False
    character_speaking: Optional[str] = None

    def __post_init__(self):
        # Calculate overall confidence
        if self.text_segments:
            self.total_confidence = sum(seg.confidence for seg in self.text_segments) / len(self.text_segments)

            # Check if any segment contains dialogue
            self.dialogue_detected = any(seg.is_dialogue for seg in self.text_segments)

            # Try to detect character name
            self.character_speaking = self._detect_character_name()
        else:
            self.total_confidence = 0.0

    def _detect_character_name(self) -> Optional[str]:
        """Detect character name from text segments."""
        for segment in self.text_segments:
            if segment.character_name:
                return segment.character_name

        # Look for patterns that might be character names
        for segment in self.text_segments:
            text = segment.cleaned_text.strip()
            # Character names are often short, capitalized, and appear at dialogue start
            if (len(text.split()) <= 2 and
                    text[0].isupper() and
                    segment.bounding_box.region_type == "character_name"):
                return text

        return None

    def get_dialogue_text(self) -> str:
        """Get combined dialogue text."""
        dialogue_segments = [seg for seg in self.text_segments if seg.is_dialogue]
        return " ".join(seg.cleaned_text for seg in dialogue_segments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists if any
        return result


class SkyrimUIDetector:
    """Skyrim-specific UI element detection."""

    def __init__(self):
        """Initialize UI detector with Skyrim-specific regions."""
        logger.info("[OCR] Initializing Skyrim UI detector...")

        # Define Skyrim UI regions for 2560x1440 resolution
        # These coordinates are based on typical Skyrim UI layout
        self.ui_regions = {
            "dialogue_main": {
                "coords": (200, 800, 2360, 1200),  # Main dialogue area
                "priority": 1,
                "type": "dialogue"
            },
            "character_name": {
                "coords": (200, 750, 800, 800),  # Character name area
                "priority": 2,
                "type": "character_name"
            },
            "subtitle": {
                "coords": (400, 1200, 2160, 1350),  # Subtitle area
                "priority": 3,
                "type": "subtitle"
            },
            "menu_text": {
                "coords": (100, 100, 2460, 700),  # Menu areas
                "priority": 4,
                "type": "menu"
            }
        }

        logger.info(f"[OCR] Loaded {len(self.ui_regions)} UI regions")

    def detect_text_regions(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect potential text regions in Skyrim UI."""
        height, width = frame.shape[:2]
        regions = []

        # Scale regions based on actual frame size
        scale_x = width / 2560
        scale_y = height / 1440

        for region_name, region_data in self.ui_regions.items():
            x1, y1, x2, y2 = region_data["coords"]

            # Scale coordinates
            scaled_x1 = int(x1 * scale_x)
            scaled_y1 = int(y1 * scale_y)
            scaled_x2 = int(x2 * scale_x)
            scaled_y2 = int(y2 * scale_y)

            # Ensure coordinates are within frame bounds
            scaled_x1 = max(0, min(scaled_x1, width))
            scaled_y1 = max(0, min(scaled_y1, height))
            scaled_x2 = max(0, min(scaled_x2, width))
            scaled_y2 = max(0, min(scaled_y2, height))

            # Check if region contains potential text (simple brightness check)
            region_roi = frame[scaled_y1:scaled_y2, scaled_x1:scaled_x2]

            if region_roi.size > 0:
                # Convert to grayscale for analysis
                if len(region_roi.shape) == 3:
                    gray_roi = cv2.cvtColor(region_roi, cv2.COLOR_BGR2GRAY)
                else:
                    gray_roi = region_roi

                # Check if region has sufficient contrast (likely contains text)
                contrast = gray_roi.std()

                if contrast > 15:  # Threshold for text presence
                    bbox = BoundingBox(
                        x1=scaled_x1,
                        y1=scaled_y1,
                        x2=scaled_x2,
                        y2=scaled_y2,
                        confidence=min(contrast / 100.0, 1.0),
                        region_type=region_data["type"]
                    )
                    regions.append(bbox)

        # Sort regions by priority
        region_priority = {"dialogue": 1, "character_name": 2, "subtitle": 3, "menu": 4}
        regions.sort(key=lambda r: region_priority.get(r.region_type, 5))

        logger.debug(f"[OCR] Detected {len(regions)} potential text regions")
        return regions


class OCREngine:
    """
    High-performance OCR engine optimized for Skyrim dialogue extraction.
    GPU ONLY VERSION - Forces GPU usage for debugging.
    """

    def __init__(self,
                 gpu_enabled: bool = True,
                 languages: List[str] = ['en'],
                 confidence_threshold: float = 0.6,
                 max_workers: int = 4,
                 batch_size: int = 8):
        """
        Initialize OCR Engine with FORCED GPU acceleration.

        Args:
            gpu_enabled: Whether to use GPU acceleration (forced to True)
            languages: List of language codes for recognition
            confidence_threshold: Minimum confidence for text acceptance
            max_workers: Number of worker threads
            batch_size: Batch size for processing
        """
        # Force GPU usage for debugging
        self.gpu_enabled = True
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Performance tracking
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'average_fps': 0,
            'text_segments_found': 0,
            'gpu_utilization': 0
        }

        # Thread safety
        self._lock = threading.Lock()

        # Initialize components
        logger.info("[OCR] Initializing OCR Engine (GPU ONLY VERSION)...")
        self._debug_cuda_setup()
        self._initialize_easyocr()
        self._initialize_ui_detector()

        logger.info("[OCR] OCR Engine initialized successfully")

    def _debug_cuda_setup(self):
        """Debug CUDA setup thoroughly."""
        logger.info("[CUDA] Debugging CUDA setup...")

        try:
            import torch

            logger.info(f"[CUDA] PyTorch version: {torch.__version__}")
            logger.info(f"[CUDA] CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                logger.info(f"[CUDA] CUDA version: {torch.version.cuda}")
                logger.info(f"[CUDA] GPU count: {torch.cuda.device_count()}")

                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_props = torch.cuda.get_device_properties(i)
                    logger.info(f"[CUDA] GPU {i}: {gpu_name}")
                    logger.info(f"[CUDA] CUDA capability: sm_{gpu_props.major}{gpu_props.minor}")
                    logger.info(f"[CUDA] Total memory: {gpu_props.total_memory / 1024 ** 3:.1f} GB")

                # Check compiled CUDA architectures
                arch_list = torch.cuda.get_arch_list()
                logger.info(f"[CUDA] PyTorch compiled for: {arch_list}")

                # Test basic CUDA operation
                logger.info("[CUDA] Testing basic CUDA operation...")
                x = torch.tensor([1.0, 2.0]).cuda()
                y = x * 2
                logger.info(f"[CUDA] Basic test result: {y}")
                logger.info("[CUDA] ✅ Basic CUDA operation successful")

            else:
                logger.error("[CUDA] ❌ CUDA not available!")
                raise RuntimeError("CUDA not available")

        except Exception as e:
            logger.error(f"[CUDA] ❌ CUDA debug failed: {e}")
            raise

    def _initialize_easyocr(self):
        """Initialize EasyOCR with FORCED GPU settings."""
        logger.info("[OCR] Initializing EasyOCR with FORCED GPU...")

        try:
            # Check GPU availability first
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available - cannot initialize GPU OCR")

            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            logger.info(f"[OCR] FORCED GPU acceleration: {gpu_name}")
            logger.info(f"[OCR] CUDA version: {torch.version.cuda}")

            # Set CUDA device explicitly
            torch.cuda.set_device(0)
            logger.info("[OCR] CUDA device set to GPU 0")

            # Clear CUDA cache
            torch.cuda.empty_cache()
            logger.info("[OCR] CUDA cache cleared")

            # Initialize EasyOCR reader with GPU FORCED
            logger.info("[OCR] Creating EasyOCR Reader with gpu=True...")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=True,  # FORCED
                verbose=True  # Enable verbose for debugging
            )

            logger.info(f"[OCR] ✅ EasyOCR initialized successfully with GPU")
            logger.info(f"[OCR] Languages: {self.languages}")
            logger.info(f"[OCR] GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")

        except Exception as e:
            logger.error(f"[OCR] ❌ Failed to initialize EasyOCR with GPU: {e}")
            logger.error(f"[OCR] Error type: {type(e).__name__}")
            logger.error(f"[OCR] GPU memory before error: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")
            raise

    def _initialize_ui_detector(self):
        """Initialize Skyrim UI detector."""
        self.ui_detector = SkyrimUIDetector()
        logger.info("[OCR] Skyrim UI detector initialized")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for optimal OCR accuracy.

        Args:
            frame: Input frame array

        Returns:
            Preprocessed frame
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            # Apply adaptive threshold for better text visibility
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to clean up text
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.error(f"[OCR] Error in frame preprocessing: {e}")
            return frame

    def extract_text_from_region(self, frame: np.ndarray, bbox: BoundingBox) -> List[TextSegment]:
        """
        Extract text from a specific region.

        Args:
            frame: Input frame
            bbox: Bounding box defining the region

        Returns:
            List of text segments found in the region
        """
        try:
            # Extract region
            region = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]

            if region.size == 0:
                return []

            # Preprocess region for OCR
            processed_region = self.preprocess_frame(region)

            # Log GPU status before OCR
            logger.debug(f"[OCR] GPU memory before readtext: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")

            # Run EasyOCR on the region
            results = self.reader.readtext(processed_region)

            # Log GPU status after OCR
            logger.debug(f"[OCR] GPU memory after readtext: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")

            text_segments = []

            for (region_bbox, text, confidence) in results:
                if confidence >= self.confidence_threshold and text.strip():
                    # Convert relative coordinates to absolute
                    region_x1 = bbox.x1 + int(min(point[0] for point in region_bbox))
                    region_y1 = bbox.y1 + int(min(point[1] for point in region_bbox))
                    region_x2 = bbox.x1 + int(max(point[0] for point in region_bbox))
                    region_y2 = bbox.y1 + int(max(point[1] for point in region_bbox))

                    # Create bounding box for this text
                    text_bbox = BoundingBox(
                        x1=region_x1,
                        y1=region_y1,
                        x2=region_x2,
                        y2=region_y2,
                        confidence=confidence,
                        region_type=bbox.region_type
                    )

                    # Determine if this is dialogue text
                    is_dialogue = self._is_dialogue_text(text, bbox.region_type)

                    # Extract character name if applicable
                    character_name = None
                    if bbox.region_type == "character_name":
                        character_name = text.strip()

                    # Create text segment
                    segment = TextSegment(
                        text=text,
                        confidence=confidence,
                        bounding_box=text_bbox,
                        is_dialogue=is_dialogue,
                        character_name=character_name
                    )

                    text_segments.append(segment)

            return text_segments

        except Exception as e:
            logger.error(f"[OCR] Error extracting text from region: {e}")
            return []

    def _is_dialogue_text(self, text: str, region_type: str) -> bool:
        """
        Determine if text is dialogue based on content and region.

        Args:
            text: Extracted text
            region_type: Type of UI region

        Returns:
            True if text appears to be dialogue
        """
        if region_type in ["dialogue", "subtitle"]:
            # Check for dialogue indicators
            text_lower = text.lower().strip()

            # Skip menu items and UI elements
            ui_keywords = [
                'continue', 'save', 'load', 'options', 'quit',
                'inventory', 'map', 'journal', 'stats',
                'level up', 'perk', 'skill'
            ]

            if any(keyword in text_lower for keyword in ui_keywords):
                return False

            # Dialogue usually has reasonable length and sentence structure
            if len(text.strip()) > 10 and any(c in text for c in '.!?'):
                return True

            # Short responses might still be dialogue
            if region_type == "dialogue" and len(text.strip()) > 3:
                return True

        return False

    def process_frame(self, frame_data: FrameData) -> OCRResult:
        """
        Process a single frame for text extraction.

        Args:
            frame_data: Frame data from video processor

        Returns:
            OCR result with extracted text
        """
        start_time = time.time()

        logger.debug(f"[OCR] Processing frame {frame_data.frame_number}")

        try:
            # Log GPU status
            logger.debug(f"[OCR] GPU memory at frame start: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")

            # Detect text regions in frame
            text_regions = self.ui_detector.detect_text_regions(frame_data.frame_array)

            # Extract text from each region
            all_text_segments = []

            for region in text_regions:
                segments = self.extract_text_from_region(frame_data.frame_array, region)
                all_text_segments.extend(segments)

            processing_time = time.time() - start_time

            # Create OCR result
            result = OCRResult(
                frame_number=frame_data.frame_number,
                timestamp=frame_data.timestamp,
                frame_hash=frame_data.frame_hash,
                text_segments=all_text_segments,
                processing_time=processing_time,
                total_confidence=0.0  # Will be calculated in __post_init__
            )

            # Update statistics
            with self._lock:
                self.processing_stats['frames_processed'] += 1
                self.processing_stats['total_processing_time'] += processing_time
                self.processing_stats['text_segments_found'] += len(all_text_segments)

                if self.processing_stats['total_processing_time'] > 0:
                    self.processing_stats['average_fps'] = (
                            self.processing_stats['frames_processed'] /
                            self.processing_stats['total_processing_time']
                    )

            logger.debug(f"[OCR] Frame {frame_data.frame_number} processed: "
                         f"{len(all_text_segments)} segments, {processing_time:.3f}s")

            return result

        except Exception as e:
            logger.error(f"[OCR] Error processing frame {frame_data.frame_number}: {e}")
            logger.error(f"[OCR] GPU memory at error: {torch.cuda.memory_allocated(0) / 1024 ** 2:.1f} MB")

            # Return empty result
            processing_time = time.time() - start_time
            return OCRResult(
                frame_number=frame_data.frame_number,
                timestamp=frame_data.timestamp,
                frame_hash=frame_data.frame_hash,
                text_segments=[],
                processing_time=processing_time,
                total_confidence=0.0
            )

    def process_frames_batch(self, frames: List[FrameData]) -> List[OCRResult]:
        """
        Process multiple frames in parallel.

        Args:
            frames: List of frame data objects

        Returns:
            List of OCR results
        """
        logger.info(f"[OCR] Processing batch of {len(frames)} frames")
        start_time = time.time()

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all frame processing tasks
            future_to_frame = {
                executor.submit(self.process_frame, frame): frame
                for frame in frames
            }

            # Collect results as they complete
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]

                try:
                    result = future.result()
                    results.append(result)

                except Exception as e:
                    logger.error(f"[OCR] Error processing frame {frame.frame_number}: {e}")

                    # Add empty result for failed frame
                    processing_time = 0.0
                    empty_result = OCRResult(
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        frame_hash=frame.frame_hash,
                        text_segments=[],
                        processing_time=processing_time,
                        total_confidence=0.0
                    )
                    results.append(empty_result)

        # Sort results by frame number
        results.sort(key=lambda r: r.frame_number)

        total_time = time.time() - start_time
        avg_fps = len(frames) / total_time if total_time > 0 else 0

        logger.info(f"[OCR] Batch processing completed: {len(results)} frames, "
                    f"{total_time:.3f}s, {avg_fps:.2f} FPS")

        return results

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            stats = self.processing_stats.copy()

            # Add GPU information
            if torch.cuda.is_available():
                try:
                    stats['gpu_memory_allocated'] = torch.cuda.memory_allocated(0)
                    stats['gpu_memory_reserved'] = torch.cuda.memory_reserved(0)
                    stats['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory
                    stats[
                        'gpu_utilization'] = f"{torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100:.1f}%"
                except Exception as e:
                    logger.warning(f"[OCR] Could not get GPU stats: {e}")

        return stats

    def reset_stats(self):
        """Reset processing statistics."""
        with self._lock:
            self.processing_stats = {
                'frames_processed': 0,
                'total_processing_time': 0,
                'average_fps': 0,
                'text_segments_found': 0,
                'gpu_utilization': 0
            }
        logger.info("[OCR] Processing statistics reset")

    def save_results(self, results: List[OCRResult], output_path: str) -> bool:
        """
        Save OCR results to JSON file.

        Args:
            results: List of OCR results
            output_path: Output file path

        Returns:
            True if saved successfully
        """
        try:
            # Convert results to dictionaries
            results_data = {
                'metadata': {
                    'total_frames': len(results),
                    'processing_stats': self.get_processing_stats(),
                    'ocr_settings': {
                        'languages': self.languages,
                        'confidence_threshold': self.confidence_threshold,
                        'gpu_enabled': self.gpu_enabled
                    }
                },
                'results': [result.to_dict() for result in results]
            }

            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[OCR] Results saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"[OCR] Error saving results: {e}")
            return False


def test_ocr_engine():
    """Test function for OCR engine."""
    logger.info("=" * 60)
    logger.info("[TEST] Starting OCR Engine GPU-ONLY test")
    logger.info("=" * 60)

    # Initialize OCR engine
    ocr_engine = OCREngine(
        gpu_enabled=True,
        languages=['en'],
        confidence_threshold=0.6,
        max_workers=4
    )

    # Test with dummy frame data (for testing without video)
    logger.info("[TEST] Creating test frame...")

    # Create a test image with text
    test_frame = np.ones((1440, 2560, 3), dtype=np.uint8) * 255

    # Add some text to the frame (simulating Skyrim dialogue)
    cv2.putText(test_frame, "Hello, traveler!", (220, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(test_frame, "Lydia", (220, 780),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

    # Create test frame data
    test_frame_data = FrameData(
        frame_number=1,
        timestamp=0.033,
        frame_array=test_frame,
        frame_hash="test_hash_123"
    )

    # Process the frame
    logger.info("[TEST] Processing test frame...")
    result = ocr_engine.process_frame(test_frame_data)

    # Display results
    logger.info(f"[TEST] OCR Results:")
    logger.info(f"  Frame: {result.frame_number}")
    logger.info(f"  Timestamp: {result.timestamp:.3f}s")
    logger.info(f"  Processing time: {result.processing_time:.3f}s")
    logger.info(f"  Text segments found: {len(result.text_segments)}")
    logger.info(f"  Dialogue detected: {result.dialogue_detected}")
    logger.info(f"  Character speaking: {result.character_speaking}")

    for i, segment in enumerate(result.text_segments):
        logger.info(f"  Segment {i + 1}:")
        logger.info(f"    Text: '{segment.text}'")
        logger.info(f"    Cleaned: '{segment.cleaned_text}'")
        logger.info(f"    Confidence: {segment.confidence:.3f}")
        logger.info(f"    Is dialogue: {segment.is_dialogue}")
        logger.info(f"    Region type: {segment.bounding_box.region_type}")

    # Test batch processing
    logger.info("[TEST] Testing batch processing...")
    batch_results = ocr_engine.process_frames_batch([test_frame_data] * 3)
    logger.info(f"[TEST] Batch processed {len(batch_results)} frames")

    # Get statistics
    stats = ocr_engine.get_processing_stats()
    logger.info(f"[TEST] Processing statistics: {stats}")

    logger.info("=" * 60)
    logger.info("[TEST] OCR Engine GPU-ONLY test completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Run test if executed directly
    test_ocr_engine()