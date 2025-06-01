#!/usr/bin/env python3
"""
Data Models for Skyrim Dialogue Analyzer
========================================

Comprehensive data structures for the application including:
- Video metadata and frame data
- OCR results and text segments
- Dialogue and translation data
- Learning progress and session data
- Database ORM models

Author: Skyrim Dialogue Analyzer Team
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TextRegionType(Enum):
    """Text region type enumeration."""
    DIALOGUE = "dialogue"
    CHARACTER_NAME = "character_name"
    SUBTITLE = "subtitle"
    MENU = "menu"
    UI_ELEMENT = "ui_element"
    UNKNOWN = "unknown"


class LearningDifficulty(Enum):
    """Learning difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Core Data Structures
@dataclass
class VideoMetadata:
    """Video file metadata."""
    filepath: str
    filename: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: str
    file_size: int
    creation_time: Optional[str] = None
    bitrate: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        return cls(**data)


@dataclass
class FrameData:
    """Individual video frame data."""
    frame_number: int
    timestamp: float
    frame_array: Any  # np.ndarray - not serializable
    frame_hash: str
    is_changed: bool = False
    text_regions: List[Dict] = field(default_factory=list)

    def to_dict_serializable(self) -> Dict[str, Any]:
        """Convert to serializable dictionary (excluding frame_array)."""
        return {
            'frame_number': self.frame_number,
            'timestamp': self.timestamp,
            'frame_hash': self.frame_hash,
            'is_changed': self.is_changed,
            'text_regions': self.text_regions
        }


@dataclass
class BoundingBox:
    """Bounding box for text regions."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    region_type: str = TextRegionType.UNKNOWN.value

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
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        return cls(**data)


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
    word_count: int = 0

    def __post_init__(self):
        if not self.cleaned_text:
            self.cleaned_text = self._clean_text(self.text)
        self.word_count = len(self.cleaned_text.split())

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        import re
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
            '|': 'I',
            '0': 'O',
        }

        for wrong, correct in replacements.items():
            cleaned = cleaned.replace(wrong, correct)

        return cleaned

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['bounding_box'] = self.bounding_box.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextSegment':
        bbox_data = data.pop('bounding_box')
        bbox = BoundingBox.from_dict(bbox_data)
        return cls(bounding_box=bbox, **data)


@dataclass
class OCRResult:
    """Complete OCR result for a frame."""
    frame_number: int
    timestamp: float
    frame_hash: str
    text_segments: List[TextSegment]
    processing_time: float
    total_confidence: float = 0.0
    dialogue_detected: bool = False
    character_speaking: Optional[str] = None

    def __post_init__(self):
        if self.text_segments:
            self.total_confidence = sum(seg.confidence for seg in self.text_segments) / len(self.text_segments)
            self.dialogue_detected = any(seg.is_dialogue for seg in self.text_segments)
            self.character_speaking = self._detect_character_name()
        else:
            self.total_confidence = 0.0

    def _detect_character_name(self) -> Optional[str]:
        """Detect character name from text segments."""
        for segment in self.text_segments:
            if segment.character_name:
                return segment.character_name

        # Look for character name patterns
        for segment in self.text_segments:
            text = segment.cleaned_text.strip()
            if (len(text.split()) <= 2 and
                text and text[0].isupper() and
                segment.bounding_box.region_type == TextRegionType.CHARACTER_NAME.value):
                return text

        return None

    def get_dialogue_text(self) -> str:
        """Get combined dialogue text."""
        dialogue_segments = [seg for seg in self.text_segments if seg.is_dialogue]
        return " ".join(seg.cleaned_text for seg in dialogue_segments)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['text_segments'] = [seg.to_dict() for seg in self.text_segments]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRResult':
        segments_data = data.pop('text_segments')
        segments = [TextSegment.from_dict(seg_data) for seg_data in segments_data]
        return cls(text_segments=segments, **data)


# Dialogue and Translation Models
@dataclass
class DialogueSegment:
    """A complete dialogue segment with timing."""
    segment_id: str
    start_timestamp: float
    end_timestamp: float
    duration: float
    character_name: Optional[str]
    dialogue_text: str
    confidence: float
    ocr_results: List[OCRResult]
    is_validated: bool = False
    notes: str = ""

    def __post_init__(self):
        if not self.segment_id:
            self.segment_id = str(uuid.uuid4())
        self.duration = self.end_timestamp - self.start_timestamp

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['ocr_results'] = [ocr.to_dict() for ocr in self.ocr_results]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueSegment':
        ocr_data = data.pop('ocr_results')
        ocr_results = [OCRResult.from_dict(ocr_dict) for ocr_dict in ocr_data]
        return cls(ocr_results=ocr_results, **data)


@dataclass
class Translation:
    """Translation of dialogue text."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    translation_confidence: float
    translation_method: str  # "ollama", "manual", etc.
    created_at: datetime
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Translation':
        created_at_str = data.pop('created_at')
        created_at = datetime.fromisoformat(created_at_str)
        return cls(created_at=created_at, **data)


@dataclass
class GrammarExplanation:
    """Grammar explanation for learning."""
    text_segment: str
    grammar_point: str
    explanation: str
    examples: List[str]
    difficulty_level: str
    related_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GrammarExplanation':
        return cls(**data)


@dataclass
class VocabularyItem:
    """Vocabulary item for learning."""
    word: str
    definition: str
    context: str
    difficulty_level: str
    frequency_score: int
    pronunciation: Optional[str] = None
    word_type: Optional[str] = None  # noun, verb, adjective, etc.
    related_words: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VocabularyItem':
        return cls(**data)


# Learning and Progress Models
@dataclass
class LearningSession:
    """Individual learning session data."""
    session_id: str
    project_id: str
    start_time: datetime
    end_time: Optional[datetime]
    dialogue_segments_studied: List[str]  # segment_ids
    vocabulary_learned: List[str]  # words
    grammar_points_covered: List[str]
    session_notes: str = ""

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningSession':
        start_time = datetime.fromisoformat(data.pop('start_time'))
        end_time_str = data.pop('end_time', None)
        end_time = datetime.fromisoformat(end_time_str) if end_time_str else None
        return cls(start_time=start_time, end_time=end_time, **data)


@dataclass
class UserProgress:
    """User learning progress tracking."""
    user_id: str
    project_id: str
    total_dialogue_segments: int
    completed_segments: int
    vocabulary_learned: int
    grammar_points_mastered: int
    total_study_time: float  # in seconds
    last_session_date: datetime
    skill_level: str = LearningDifficulty.BEGINNER.value

    @property
    def completion_percentage(self) -> float:
        if self.total_dialogue_segments == 0:
            return 0.0
        return (self.completed_segments / self.total_dialogue_segments) * 100

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['last_session_date'] = self.last_session_date.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProgress':
        last_session = datetime.fromisoformat(data.pop('last_session_date'))
        return cls(last_session_date=last_session, **data)


# Project and Session Models
@dataclass
class Project:
    """Main project containing video and analysis data."""
    project_id: str
    name: str
    description: str
    video_metadata: VideoMetadata
    created_at: datetime
    last_modified: datetime
    processing_status: str = ProcessingStatus.PENDING.value
    total_frames: int = 0
    processed_frames: int = 0
    dialogue_segments: List[DialogueSegment] = field(default_factory=list)

    def __post_init__(self):
        if not self.project_id:
            self.project_id = str(uuid.uuid4())

    @property
    def processing_progress(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return (self.processed_frames / self.total_frames) * 100

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['video_metadata'] = self.video_metadata.to_dict()
        result['created_at'] = self.created_at.isoformat()
        result['last_modified'] = self.last_modified.isoformat()
        result['dialogue_segments'] = [seg.to_dict() for seg in self.dialogue_segments]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        video_data = data.pop('video_metadata')
        video_metadata = VideoMetadata.from_dict(video_data)

        created_at = datetime.fromisoformat(data.pop('created_at'))
        last_modified = datetime.fromisoformat(data.pop('last_modified'))

        segments_data = data.pop('dialogue_segments', [])
        dialogue_segments = [DialogueSegment.from_dict(seg_data) for seg_data in segments_data]

        return cls(
            video_metadata=video_metadata,
            created_at=created_at,
            last_modified=last_modified,
            dialogue_segments=dialogue_segments,
            **data
        )


# Database Schema and ORM-like functionality
class DatabaseSchema:
    """Database schema definitions."""

    @staticmethod
    def create_tables(conn: sqlite3.Connection):
        """Create all necessary database tables."""
        cursor = conn.cursor()

        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                video_metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                processing_status TEXT DEFAULT 'pending',
                total_frames INTEGER DEFAULT 0,
                processed_frames INTEGER DEFAULT 0
            )
        ''')

        # Dialogue segments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dialogue_segments (
                segment_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                start_timestamp REAL NOT NULL,
                end_timestamp REAL NOT NULL,
                duration REAL NOT NULL,
                character_name TEXT,
                dialogue_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_validated BOOLEAN DEFAULT FALSE,
                notes TEXT DEFAULT '',
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')

        # OCR results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                frame_hash TEXT NOT NULL,
                processing_time REAL NOT NULL,
                total_confidence REAL NOT NULL,
                dialogue_detected BOOLEAN DEFAULT FALSE,
                character_speaking TEXT,
                text_segments TEXT NOT NULL,
                FOREIGN KEY (segment_id) REFERENCES dialogue_segments (segment_id)
            )
        ''')

        # Translations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                source_language TEXT NOT NULL,
                target_language TEXT NOT NULL,
                translation_confidence REAL NOT NULL,
                translation_method TEXT NOT NULL,
                created_at TEXT NOT NULL,
                notes TEXT DEFAULT '',
                FOREIGN KEY (segment_id) REFERENCES dialogue_segments (segment_id)
            )
        ''')

        # Vocabulary items table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocabulary_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                word TEXT NOT NULL,
                definition TEXT NOT NULL,
                context TEXT NOT NULL,
                difficulty_level TEXT NOT NULL,
                frequency_score INTEGER DEFAULT 1,
                pronunciation TEXT,
                word_type TEXT,
                related_words TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')

        # Grammar explanations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS grammar_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                text_segment TEXT NOT NULL,
                grammar_point TEXT NOT NULL,
                explanation TEXT NOT NULL,
                examples TEXT NOT NULL,
                difficulty_level TEXT NOT NULL,
                related_concepts TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')

        # Learning sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_sessions (
                session_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                dialogue_segments_studied TEXT,
                vocabulary_learned TEXT,
                grammar_points_covered TEXT,
                session_notes TEXT DEFAULT '',
                FOREIGN KEY (project_id) REFERENCES projects (project_id)
            )
        ''')

        # User progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                total_dialogue_segments INTEGER DEFAULT 0,
                completed_segments INTEGER DEFAULT 0,
                vocabulary_learned INTEGER DEFAULT 0,
                grammar_points_mastered INTEGER DEFAULT 0,
                total_study_time REAL DEFAULT 0,
                last_session_date TEXT NOT NULL,
                skill_level TEXT DEFAULT 'beginner',
                FOREIGN KEY (project_id) REFERENCES projects (project_id),
                UNIQUE(user_id, project_id)
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_project ON dialogue_segments(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ocr_segment ON ocr_results(segment_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_translations_segment ON translations(segment_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_vocabulary_project ON vocabulary_items(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_grammar_project ON grammar_explanations(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_project ON learning_sessions(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_progress_user ON user_progress(user_id)')

        conn.commit()
        logger.info("[DB] Database schema created successfully")


# Utility functions for data conversion
def serialize_to_json(obj: Any) -> str:
    """Serialize object to JSON string."""
    if hasattr(obj, 'to_dict'):
        return json.dumps(obj.to_dict(), ensure_ascii=False)
    elif isinstance(obj, list):
        return json.dumps([item.to_dict() if hasattr(item, 'to_dict') else item for item in obj], ensure_ascii=False)
    else:
        return json.dumps(obj, ensure_ascii=False)


def deserialize_from_json(json_str: str, obj_class: type) -> Any:
    """Deserialize JSON string to object."""
    data = json.loads(json_str)
    if hasattr(obj_class, 'from_dict'):
        return obj_class.from_dict(data)
    else:
        return obj_class(**data)


# Export all models
__all__ = [
    'ProcessingStatus', 'TextRegionType', 'LearningDifficulty',
    'VideoMetadata', 'FrameData', 'BoundingBox', 'TextSegment', 'OCRResult',
    'DialogueSegment', 'Translation', 'GrammarExplanation', 'VocabularyItem',
    'LearningSession', 'UserProgress', 'Project', 'DatabaseSchema',
    'serialize_to_json', 'deserialize_from_json'
]