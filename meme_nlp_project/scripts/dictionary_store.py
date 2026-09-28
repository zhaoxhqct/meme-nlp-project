from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DICTIONARY_FIELDS = ("meme", "category", "meaning", "example")


class DuplicateDictionaryEntry(ValueError):
    pass


def normalize_dictionary_entry(payload: dict[str, Any]) -> dict[str, str]:
    values = {
        field: str(payload.get(field, "")).strip()
        for field in DICTIONARY_FIELDS
    }
    limits = {
        "meme": 40,
        "category": 24,
        "meaning": 100,
        "example": 300,
    }
    labels = {
        "meme": "热梗词条",
        "category": "类别",
        "meaning": "释义",
        "example": "示例",
    }

    for field, value in values.items():
        if not value:
            raise ValueError(f"{labels[field]}不能为空。")
        if len(value) > limits[field]:
            raise ValueError(
                f"{labels[field]}最多 {limits[field]} 个字符。"
            )
        if any(character in value for character in ("\r", "\n", "\t")):
            raise ValueError(f"{labels[field]}不能包含制表符或换行符。")
    return values


def append_dictionary_entry(
    path: str | Path,
    payload: dict[str, Any],
) -> dict[str, str]:
    dictionary_path = Path(path)
    entry = normalize_dictionary_entry(payload)
    rows: list[dict[str, str]] = []

    if dictionary_path.exists() and dictionary_path.stat().st_size:
        with dictionary_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != DICTIONARY_FIELDS:
                raise ValueError(
                    "用户词典表头必须是 meme,category,meaning,example。"
                )
            rows = [
                {
                    field: (row.get(field) or "").strip()
                    for field in DICTIONARY_FIELDS
                }
                for row in reader
            ]

    meme_key = entry["meme"].casefold()
    if any(row["meme"].casefold() == meme_key for row in rows):
        raise DuplicateDictionaryEntry("这个词条已经存在。")

    rows.append(entry)
    dictionary_path.parent.mkdir(parents=True, exist_ok=True)
    with dictionary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=DICTIONARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return entry
