from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemeMatch:
    meme: str
    start: int
    end: int
    category: str
    meaning: str


class MemeLexicon:
    """Longest-match dictionary lookup for meme candidates."""

    def __init__(self, entries: dict[str, dict[str, str]]):
        self.entries = entries
        self._keys = sorted(entries, key=len, reverse=True)

    @classmethod
    def from_csv(cls, path: str | Path) -> "MemeLexicon":
        return cls(cls._read_entries(Path(path)))

    @classmethod
    def from_csvs(cls, paths: list[str | Path]) -> "MemeLexicon":
        entries: dict[str, dict[str, str]] = {}
        for path in paths:
            candidate = Path(path)
            if candidate.is_file():
                entries.update(cls._read_entries(candidate))
        return cls(entries)

    @staticmethod
    def _read_entries(path: Path) -> dict[str, dict[str, str]]:
        entries: dict[str, dict[str, str]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                meme = (row.get("meme") or "").strip()
                if meme:
                    entries[meme.lower()] = {
                        "display": meme,
                        "category": (row.get("category") or "").strip(),
                        "meaning": (row.get("meaning") or "").strip(),
                    }
        return entries

    def add(self, meme: str, category: str, meaning: str) -> None:
        normalized = meme.strip()
        if not normalized:
            raise ValueError("热梗词条不能为空。")
        self.entries[normalized.lower()] = {
            "display": normalized,
            "category": category.strip(),
            "meaning": meaning.strip(),
        }
        self._keys = sorted(self.entries, key=len, reverse=True)

    def find_all(self, text: str) -> list[MemeMatch]:
        """Return non-overlapping matches, preferring the longest term."""
        lowered = text.lower()
        found: list[MemeMatch] = []
        position = 0
        while position < len(lowered):
            match = next(
                (
                    key
                    for key in self._keys
                    if lowered.startswith(key, position)
                ),
                None,
            )
            if match is None:
                position += 1
                continue
            entry = self.entries[match]
            found.append(
                MemeMatch(
                    meme=entry["display"],
                    start=position,
                    end=position + len(match),
                    category=entry["category"],
                    meaning=entry["meaning"],
                )
            )
            position += len(match)
        return found

    def enrich(self, text: str) -> str:
        matches = self.find_all(text)
        hits = "|".join(match.meme for match in matches) or "NONE"
        return f"{text} [LEX] {hits}"

    def explain(self, text: str) -> list[dict[str, str]]:
        return [
            {
                "meme": match.meme,
                "category": match.category,
                "meaning": match.meaning,
            }
            for match in self.find_all(text)
        ]
