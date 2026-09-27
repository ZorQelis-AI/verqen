"""Deduplicator providing exact SHA-256 and normalized near-duplicate detection."""

from __future__ import annotations

import hashlib
import re
from typing import Set


class Deduplicator:
    """Tracks seen states and normalized content to guarantee zero duplicates."""

    def __init__(self) -> None:
        self.exact_hashes: Set[str] = set()
        self.normalized_hashes: Set[str] = set()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Strip punctuation, lowercase, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_duplicate(self, state_text: str, question_text: str = "") -> bool:
        """Check whether state/question pair has been observed."""
        exact_h = hashlib.sha256(f"{state_text}::{question_text}".encode("utf-8")).hexdigest()
        if exact_h in self.exact_hashes:
            return True

        norm_content = f"{self._normalize_text(state_text)}::{self._normalize_text(question_text)}"
        norm_h = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
        if norm_h in self.normalized_hashes:
            return True

        return False

    def register(self, state_text: str, question_text: str = "") -> None:
        """Register state/question pair in deduplication registry."""
        exact_h = hashlib.sha256(f"{state_text}::{question_text}".encode("utf-8")).hexdigest()
        self.exact_hashes.add(exact_h)

        norm_content = f"{self._normalize_text(state_text)}::{self._normalize_text(question_text)}"
        norm_h = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()
        self.normalized_hashes.add(norm_h)

    def size(self) -> int:
        return len(self.exact_hashes)

    def clear(self) -> None:
        self.exact_hashes.clear()
        self.normalized_hashes.clear()
