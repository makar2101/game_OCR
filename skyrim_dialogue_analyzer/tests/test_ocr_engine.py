#!/usr/bin/env python3
"""
OCR Engine Test Suite
====================

Comprehensive testing for the OCR Engine module including:
- Unit tests for core functionality
- Integration tests with video processor
- Performance benchmarks
- GPU acceleration validation
- Skyrim-specific dialogue detection tests

Author: Skyrim Dialogue Analyzer Team
"""

import unittest
import numpy as np
import cv2
import time
import tempfile
import json
from pathlib import Path
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from core.ocr_engine import OCREngine, OCRResult, TextSegment, BoundingBox, SkyrimUIDetector
    from core.video_processor import FrameData, VideoProcessor
    from data.models import ProcessingStatus, TextRegionType
except ImportError as e:
    logging.error(f"Import error: {e}")
    logging.error("Make sure all required modules are installed and in the correct path")
    sys.exit(1)

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestOCREngine(unittest.TestCase):
    """Test cases for OCR Engine functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test class with OCR engine instance."""
        logger.info("✓ Frame preprocessing test passed")

    def test_single_frame_processing(self):
        """Test processing of a single frame."""
        logger.info("Testing single frame processing...")

        frame_data = self.test_frames[0]  # Simple dialogue frame

        # Process frame
        start_time = time.time()
        result = self.ocr_engine.process_frame(frame_data)
        processing_time = time.time() - start_time

        # Validate result
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.frame_number, frame_data.frame_number)
        self.assertEqual(result.timestamp, frame_data.timestamp)
        self.assertEqual(result.frame_hash, frame_data.frame_hash)
        self.assertGreater(result.processing_time, 0)

        # Check for text detection
        self.assertIsInstance(result.text_segments, list)

        if result.text_segments:
            logger.info(f"✓ Found {len(result.text_segments)} text segments:")
            for i, segment in enumerate(result.text_segments):
                logger.info(f"  Segment {i + 1}: '{segment.cleaned_text}' (confidence: {segment.confidence:.3f})")

            # Validate text segments
            for segment in result.text_segments:
                self.assertIsInstance(segment, TextSegment)
                self.assertGreater(segment.confidence, 0)
                self.assertIsInstance(segment.text, str)
                self.assertIsInstance(segment.cleaned_text, str)
                self.assertIsInstance(segment.bounding_box, BoundingBox)

        logger.info(f"✓ Single frame processing completed in {processing_time:.3f}s")

    def test_batch_processing(self):
        """Test batch processing of multiple frames."""
        logger.info("Testing batch frame processing...")

        # Process all test frames
        start_time = time.time()
        results = self.ocr_engine.process_frames_batch(self.test_frames)
        total_time = time.time() - start_time

        # Validate results
        self.assertEqual(len(results), len(self.test_frames))
        self.assertGreater(total_time, 0)

        fps = len(results) / total_time
        logger.info(f"✓ Batch processing: {len(results)} frames in {total_time:.3f}s ({fps:.2f} FPS)")

        # Check results are properly ordered
        for i, result in enumerate(results):
            self.assertEqual(result.frame_number, self.test_frames[i].frame_number)
            self.assertIsInstance(result, OCRResult)

        # Count total text segments found
        total_segments = sum(len(result.text_segments) for result in results)
        dialogue_frames = sum(1 for result in results if result.dialogue_detected)

        logger.info(f"✓ Total text segments found: {total_segments}")
        logger.info(f"✓ Dialogue frames detected: {dialogue_frames}/{len(results)}")

    def test_dialogue_detection(self):
        """Test dialogue detection accuracy."""
        logger.info("Testing dialogue detection...")

        # Process frames with known dialogue content
        dialogue_frame = self.test_frames[0]  # Has "Hello, traveler!"
        menu_frame = self.test_frames[2]  # Has menu items

        dialogue_result = self.ocr_engine.process_frame(dialogue_frame)
        menu_result = self.ocr_engine.process_frame(menu_frame)

        # Check dialogue detection
        if dialogue_result.text_segments:
            dialogue_segments = [seg for seg in dialogue_result.text_segments if seg.is_dialogue]
            logger.info(f"✓ Dialogue frame: {len(dialogue_segments)} dialogue segments detected")

            # Should find dialogue text
            dialogue_text = dialogue_result.get_dialogue_text()
            self.assertIsInstance(dialogue_text, str)
            if dialogue_text:
                logger.info(f"  Dialogue text: '{dialogue_text}'")

        if menu_result.text_segments:
            menu_dialogue_segments = [seg for seg in menu_result.text_segments if seg.is_dialogue]
            logger.info(f"✓ Menu frame: {len(menu_dialogue_segments)} dialogue segments detected")

            # Menu frame should have fewer dialogue segments
            self.assertLessEqual(len(menu_dialogue_segments), len(dialogue_result.text_segments))

    def test_character_name_detection(self):
        """Test character name detection."""
        logger.info("Testing character name detection...")

        # Test frame with character name
        frame_with_character = self.test_frames[1]  # Has "Whiterun Guard"
        result = self.ocr_engine.process_frame(frame_with_character)

        if result.character_speaking:
            logger.info(f"✓ Character detected: '{result.character_speaking}'")
            self.assertIsInstance(result.character_speaking, str)
            self.assertGreater(len(result.character_speaking), 0)

        # Check for character name segments
        character_segments = [seg for seg in result.text_segments if seg.character_name]
        if character_segments:
            logger.info(f"✓ Found {len(character_segments)} character name segments")
            for seg in character_segments:
                logger.info(f"  Character: '{seg.character_name}'")

    def test_confidence_filtering(self):
        """Test confidence threshold filtering."""
        logger.info("Testing confidence filtering...")

        # Test with different confidence thresholds
        high_confidence_engine = OCREngine(
            gpu_enabled=self.ocr_engine.gpu_enabled,
            confidence_threshold=0.8,
            max_workers=1
        )

        low_confidence_engine = OCREngine(
            gpu_enabled=self.ocr_engine.gpu_enabled,
            confidence_threshold=0.3,
            max_workers=1
        )

        test_frame = self.test_frames[0]

        high_result = high_confidence_engine.process_frame(test_frame)
        low_result = low_confidence_engine.process_frame(test_frame)

        # Low confidence should find more or equal segments
        self.assertGreaterEqual(len(low_result.text_segments), len(high_result.text_segments))

        # All high confidence segments should have confidence >= 0.8
        for segment in high_result.text_segments:
            self.assertGreaterEqual(segment.confidence, 0.8)

        logger.info(f"✓ High confidence (≥0.8): {len(high_result.text_segments)} segments")
        logger.info(f"✓ Low confidence (≥0.3): {len(low_result.text_segments)} segments")

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        logger.info("Testing text cleaning...")

        # Create a text segment with messy text
        bbox = BoundingBox(0, 0, 100, 50, 0.9)
        segment = TextSegment(
            text="  Hello,   world!  \n\n",
            confidence=0.9,
            bounding_box=bbox
        )

        # Check that text was cleaned
        self.assertEqual(segment.cleaned_text, "Hello, world!")
        self.assertGreater(segment.word_count, 0)

        # Test Skyrim-specific corrections
        skyrim_segment = TextSegment(
            text="Draqonborn from Skynm",
            confidence=0.8,
            bounding_box=bbox
        )

        self.assertIn("Dragonborn", skyrim_segment.cleaned_text)
        self.assertIn("Skyrim", skyrim_segment.cleaned_text)

        logger.info("✓ Text cleaning test passed")

    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        logger.info("Testing performance benchmarks...")

        # Reset stats
        self.ocr_engine.reset_stats()

        # Process frames multiple times for better benchmark
        benchmark_frames = self.test_frames * 3  # 12 frames total

        start_time = time.time()
        results = self.ocr_engine.process_frames_batch(benchmark_frames)
        total_time = time.time() - start_time

        # Get statistics
        stats = self.ocr_engine.get_processing_stats()

        self.assertEqual(len(results), len(benchmark_frames))
        self.assertGreater(stats['frames_processed'], 0)
        self.assertGreater(stats['average_fps'], 0)

        logger.info(f"✓ Performance benchmark results:")
        logger.info(f"  Frames processed: {stats['frames_processed']}")
        logger.info(f"  Total time: {stats['total_processing_time']:.3f}s")
        logger.info(f"  Average FPS: {stats['average_fps']:.2f}")
        logger.info(f"  Text segments found: {stats['text_segments_found']}")

        # Performance targets (adjust based on hardware)
        if self.ocr_engine.gpu_enabled:
            self.assertGreater(stats['average_fps'], 1.0, "GPU processing should be > 1 FPS")
        else:
            self.assertGreater(stats['average_fps'], 0.5, "CPU processing should be > 0.5 FPS")

    def test_result_serialization(self):
        """Test OCR result serialization/deserialization."""
        logger.info("Testing result serialization...")

        # Process a frame
        result = self.ocr_engine.process_frame(self.test_frames[0])

        # Convert to dict
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)

        # Check required fields
        required_fields = ['frame_number', 'timestamp', 'text_segments', 'processing_time']
        for field in required_fields:
            self.assertIn(field, result_dict)

        # Test JSON serialization
        import json
        json_str = json.dumps(result_dict)
        self.assertIsInstance(json_str, str)

        # Test deserialization
        loaded_dict = json.loads(json_str)
        self.assertEqual(loaded_dict['frame_number'], result.frame_number)

        logger.info("✓ Result serialization test passed")

    def test_save_and_load_results(self):
        """Test saving and loading OCR results."""
        logger.info("Testing save/load functionality...")

        # Process all frames
        results = self.ocr_engine.process_frames_batch(self.test_frames)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save results
            success = self.ocr_engine.save_results(results, temp_path)
            self.assertTrue(success, "Should save results successfully")

            # Check file exists and is valid JSON
            self.assertTrue(Path(temp_path).exists())

            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            self.assertIn('metadata', loaded_data)
            self.assertIn('results', loaded_data)
            self.assertEqual(len(loaded_data['results']), len(results))

            logger.info(f"✓ Results saved and loaded successfully: {len(results)} frames")

        finally:
            # Clean up
            if Path(temp_path).exists():
                Path(temp_path).unlink()


class TestOCRIntegration(unittest.TestCase):
    """Integration tests with video processor."""

    def setUp(self):
        """Set up integration tests."""
        self.ocr_engine = OCREngine(gpu_enabled=True, max_workers=2)

    def test_video_processor_integration(self):
        """Test integration with video processor frames."""
        logger.info("Testing video processor integration...")

        # This test would normally use real video frames
        # For now, we'll simulate the integration

        # Create mock video processor frame data
        mock_frame = np.random.randint(0, 255, (1440, 2560, 3), dtype=np.uint8)

        frame_data = FrameData(
            frame_number=100,
            timestamp=3.33,
            frame_array=mock_frame,
            frame_hash="integration_test_hash"
        )

        # Process with OCR engine
        result = self.ocr_engine.process_frame(frame_data)

        # Validate integration
        self.assertIsInstance(result, OCRResult)
        self.assertEqual(result.frame_number, frame_data.frame_number)
        self.assertEqual(result.timestamp, frame_data.timestamp)

        logger.info("✓ Video processor integration test passed")


def run_gpu_validation():
    """Validate GPU acceleration is working."""
    logger.info("=" * 60)
    logger.info("GPU Validation Test")
    logger.info("=" * 60)

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda

            logger.info(f"✓ CUDA Available: True")
            logger.info(f"✓ GPU Count: {gpu_count}")
            logger.info(f"✓ GPU Name: {gpu_name}")
            logger.info(f"✓ CUDA Version: {cuda_version}")

            # Test GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(f"✓ GPU Memory: {total_memory / 1024 ** 3:.1f} GB")

            return True
        else:
            logger.warning("⚠ CUDA not available - will use CPU processing")
            return False

    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False


def run_performance_test():
    """Run comprehensive performance test."""
    logger.info("=" * 60)
    logger.info("Performance Test")
    logger.info("=" * 60)

    # Create OCR engine
    ocr_engine = OCREngine(gpu_enabled=True, max_workers=4, batch_size=8)

    # Create test frames of different complexities
    test_frames = []

    for i in range(20):
        # Create frame with varying text complexity
        frame = np.ones((1440, 2560, 3), dtype=np.uint8) * 255

        # Add text with varying density
        text_count = (i % 5) + 1
        for j in range(text_count):
            cv2.putText(frame, f"Test text {i}-{j}",
                        (200 + j * 400, 900 + j * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        frame_data = FrameData(
            frame_number=i,
            timestamp=i * 0.033,
            frame_array=frame,
            frame_hash=f"perf_test_{i}"
        )
        test_frames.append(frame_data)

    # Run performance test
    logger.info(f"Processing {len(test_frames)} frames...")

    start_time = time.time()
    results = ocr_engine.process_frames_batch(test_frames)
    total_time = time.time() - start_time

    # Calculate metrics
    fps = len(results) / total_time
    total_segments = sum(len(r.text_segments) for r in results)
    avg_confidence = sum(r.total_confidence for r in results) / len(results)

    logger.info(f"✓ Performance Results:")
    logger.info(f"  Total frames: {len(results)}")
    logger.info(f"  Processing time: {total_time:.3f}s")
    logger.info(f"  Average FPS: {fps:.2f}")
    logger.info(f"  Total text segments: {total_segments}")
    logger.info(f"  Average confidence: {avg_confidence:.3f}")

    # Get detailed stats
    stats = ocr_engine.get_processing_stats()
    logger.info(f"  Engine stats: {stats}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("=" * 60)
    logger.info("SKYRIM DIALOGUE ANALYZER - OCR ENGINE TEST SUITE")
    logger.info("=" * 60)

    # Run GPU validation first
    gpu_available = run_gpu_validation()

    # Run performance test
    run_performance_test()

    # Run unit tests
    logger.info("=" * 60)
    logger.info("Unit Tests")
    logger.info("=" * 60)

    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    # Add test classes
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestOCREngine))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestOCRIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")

    if result.failures:
        logger.error("FAILURES:")
        for test, failure in result.failures:
            logger.error(f"  {test}: {failure}")

    if result.errors:
        logger.error("ERRORS:")
        for test, error in result.errors:
            logger.error(f"  {test}: {error}")

    if result.wasSuccessful():
        logger.info("✓ ALL TESTS PASSED!")
        exit_code = 0
    else:
        logger.error("✗ SOME TESTS FAILED!")
        exit_code = 1

    logger.info("=" * 60)
    sys.exit(exit_code)("Setting up OCR Engine test suite...")

    # Initialize OCR engine
    cls.ocr_engine = OCREngine(
        gpu_enabled=True,
        languages=['en'],
        confidence_threshold=0.5,
        max_workers=2,
        batch_size=4
    )

    # Create test frames
    cls.test_frames = cls._create_test_frames()

    logger.info("OCR Engine test suite setup complete")


@classmethod
def _create_test_frames(cls) -> list:
    """Create test frames with various text scenarios."""
    frames = []

    # Frame 1: Simple dialogue
    frame1 = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
    cv2.putText(frame1, "Hello, traveler! Welcome to Whiterun.",
                (220, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(frame1, "Guard",
                (220, 780), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

    frame_data1 = FrameData(
        frame_number=1,
        timestamp=0.033,
        frame_array=frame1,
        frame_hash="test_hash_001"
    )
    frames.append(frame_data1)

    # Frame 2: Character dialogue with longer text
    frame2 = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
    cv2.putText(frame2, "I used to be an adventurer like you,",
                (220, 880), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(frame2, "then I took an arrow to the knee.",
                (220, 920), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)
    cv2.putText(frame2, "Whiterun Guard",
                (220, 780), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)

    frame_data2 = FrameData(
        frame_number=2,
        timestamp=0.066,
        frame_array=frame2,
        frame_hash="test_hash_002"
    )
    frames.append(frame_data2)

    # Frame 3: Menu text (should not be detected as dialogue)
    frame3 = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
    cv2.putText(frame3, "CONTINUE",
                (1200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(frame3, "LOAD GAME",
                (1200, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(frame3, "NEW GAME",
                (1200, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    frame_data3 = FrameData(
        frame_number=3,
        timestamp=0.099,
        frame_array=frame3,
        frame_hash="test_hash_003"
    )
    frames.append(frame_data3)

    # Frame 4: Mixed content
    frame4 = np.ones((1440, 2560, 3), dtype=np.uint8) * 255
    cv2.putText(frame4, "What can I do for you?",
                (220, 900), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(frame4, "Lydia",
                (220, 780), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
    cv2.putText(frame4, "INVENTORY",
                (2200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    frame_data4 = FrameData(
        frame_number=4,
        timestamp=0.132,
        frame_array=frame4,
        frame_hash="test_hash_004"
    )
    frames.append(frame_data4)

    return frames


def test_ocr_engine_initialization(self):
    """Test OCR engine initialization."""
    logger.info("Testing OCR engine initialization...")

    self.assertIsNotNone(self.ocr_engine)
    self.assertIsNotNone(self.ocr_engine.reader)
    self.assertIsNotNone(self.ocr_engine.ui_detector)
    self.assertEqual(self.ocr_engine.languages, ['en'])
    self.assertEqual(self.ocr_engine.confidence_threshold, 0.5)

    logger.info("✓ OCR engine initialization test passed")


def test_ui_detector(self):
    """Test Skyrim UI detector functionality."""
    logger.info("Testing Skyrim UI detector...")

    ui_detector = SkyrimUIDetector()
    test_frame = self.test_frames[0].frame_array

    # Detect text regions
    regions = ui_detector.detect_text_regions(test_frame)

    self.assertIsInstance(regions, list)
    self.assertGreater(len(regions), 0, "Should detect at least one text region")

    for region in regions:
        self.assertIsInstance(region, BoundingBox)
        self.assertGreater(region.width, 0)
        self.assertGreater(region.height, 0)
        self.assertIn(region.region_type, [rt.value for rt in TextRegionType])

    logger.info(f"✓ UI detector found {len(regions)} regions")


def test_frame_preprocessing(self):
    """Test frame preprocessing functionality."""
    logger.info("Testing frame preprocessing...")

    original_frame = self.test_frames[0].frame_array
    processed_frame = self.ocr_engine.preprocess_frame(original_frame)

    self.assertIsNotNone(processed_frame)
    self.assertEqual(len(processed_frame.shape), 2, "Processed frame should be grayscale")
    self.assertEqual(processed_frame.dtype, np.uint8)

    # Check that preprocessing changed the frame
    original_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    self.assertFalse(np.array_equal(original_gray, processed_frame),
                     "Preprocessing should modify the frame")

    logger.info