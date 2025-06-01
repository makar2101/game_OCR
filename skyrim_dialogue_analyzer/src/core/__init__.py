#!/usr/bin/env python3
"""
Skyrim Dialogue Analyzer - Core Module
======================================

Core processing modules for video analysis, OCR, and AI translation.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Skyrim Dialogue Analyzer Team"

# Import core classes when available
try:
    from .video_processor import VideoProcessor, VideoFrame, VideoMetadata
    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSOR_AVAILABLE = False

try:
    from .ocr_engine import OCREngine
    OCR_ENGINE_AVAILABLE = True
except ImportError:
    OCR_ENGINE_AVAILABLE = False

try:
    from .ai_translator import AITranslator
    AI_TRANSLATOR_AVAILABLE = True
except ImportError:
    AI_TRANSLATOR_AVAILABLE = False

try:
    from .dialogue_detector import DialogueDetector
    DIALOGUE_DETECTOR_AVAILABLE = True
except ImportError:
    DIALOGUE_DETECTOR_AVAILABLE = False

try:
    from .scene_segmenter import SceneSegmenter
    SCENE_SEGMENTER_AVAILABLE = True
except ImportError:
    SCENE_SEGMENTER_AVAILABLE = False

try:
    from .learning_generator import LearningGenerator
    LEARNING_GENERATOR_AVAILABLE = True
except ImportError:
    LEARNING_GENERATOR_AVAILABLE = False

try:
    from .potplayer_manager import PotPlayerManager
    POTPLAYER_MANAGER_AVAILABLE = True
except ImportError:
    POTPLAYER_MANAGER_AVAILABLE = False

# Export available components
__all__ = []

if VIDEO_PROCESSOR_AVAILABLE:
    __all__.extend(['VideoProcessor', 'VideoFrame', 'VideoMetadata'])

if OCR_ENGINE_AVAILABLE:
    __all__.extend(['OCREngine'])

if AI_TRANSLATOR_AVAILABLE:
    __all__.extend(['AITranslator'])

if DIALOGUE_DETECTOR_AVAILABLE:
    __all__.extend(['DialogueDetector'])

if SCENE_SEGMENTER_AVAILABLE:
    __all__.extend(['SceneSegmenter'])

if LEARNING_GENERATOR_AVAILABLE:
    __all__.extend(['LearningGenerator'])

if POTPLAYER_MANAGER_AVAILABLE:
    __all__.extend(['PotPlayerManager'])

# Module status information
MODULE_STATUS = {
    'video_processor': VIDEO_PROCESSOR_AVAILABLE,
    'ocr_engine': OCR_ENGINE_AVAILABLE,
    'ai_translator': AI_TRANSLATOR_AVAILABLE,
    'dialogue_detector': DIALOGUE_DETECTOR_AVAILABLE,
    'scene_segmenter': SCENE_SEGMENTER_AVAILABLE,
    'learning_generator': LEARNING_GENERATOR_AVAILABLE,
    'potplayer_manager': POTPLAYER_MANAGER_AVAILABLE,
}

def get_module_status():
    """Get the availability status of all core modules."""
    return MODULE_STATUS.copy()

def get_available_modules():
    """Get list of available module names."""
    return [name for name, available in MODULE_STATUS.items() if available]

def get_missing_modules():
    """Get list of missing module names."""
    return [name for name, available in MODULE_STATUS.items() if not available]