"""Optional helper utilities for working with guideline text.

 They provide light search, section parsing, and
export capabilities on top of GuidelineRepository.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Iterable, List, Sequence

from guideline_and_clinical_trials import GuidelineRepository, GuidelineDocument, repository


def _get_text_blocks(text: str) -> List[str]:
    return [block.strip() for block in text.split("\n\n") if block.strip()]


class GuidelineToolkit:
    """Non-intrusive utilities layered over GuidelineRepository."""

    def __init__(self, repo: GuidelineRepository | None = None) -> None:
        self._repo = repo or repository

    def list_sections(self, source: str = "guideline") -> List[str]:
        doc = self._select_document(source)
        return _get_text_blocks(doc.body)

    def search(self, query: str, source: str = "guideline") -> List[str]:
        doc = self._select_document(source)
        needle = query.lower().strip()
        if not needle:
            return []
        return [block for block in _get_text_blocks(doc.body) if needle in block.lower()]

    def export_json(self) -> str:
        payload = {
            "guideline": asdict(self._repo.get_guideline()),
            "clinical_trials": asdict(self._repo.get_clinical_trials()),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _select_document(self, source: str) -> GuidelineDocument:
        normalized = source.lower()
        if normalized in ("guideline", "guidelines"):
            return self._repo.get_guideline()
        if normalized in ("trial", "trials", "clinical_trials"):
            return self._repo.get_clinical_trials()
        raise ValueError(f"Unknown source '{source}'")


def summarize_blocks(blocks: Sequence[str], limit: int = 5) -> List[str]:
    limit = max(0, limit)
    return list(blocks)[:limit]


def search_all(toolkit: GuidelineToolkit, query: str) -> List[str]:
    results: List[str] = []
    for source in ("guideline", "clinical_trials"):
        results.extend(toolkit.search(query, source=source))
    return results
