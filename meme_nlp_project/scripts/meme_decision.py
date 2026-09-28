from __future__ import annotations

import re
from typing import Any


MODEL_YES_THRESHOLD = 0.55

_SEASON_PATTERN = re.compile(
    r"(?i)(?<![a-z0-9])s\d{1,2}(?![a-z0-9])"
)
_RANK_TERMS = (
    "王者",
    "国服",
    "巅峰",
    "第一",
    "段位",
    "上分",
    "赛季",
)
_RANK_PATTERNS = (
    re.compile(r"第[一二三四五六七八九十百\d]+个?王者"),
    re.compile(r"(?:国服|巅峰|最强)(?:第一|第[一二三四五六七八九十\d]+)"),
)
_LATIN_REPEAT_PATTERN = re.compile(r"(?i)([a-z]{2,12})不\1")


def unknown_meme_signals(text: str) -> list[str]:
    """Return conservative structural evidence for an unlisted meme."""
    signals: list[str] = []
    if _SEASON_PATTERN.search(text) and any(
        term in text for term in _RANK_TERMS
    ):
        signals.append("赛季或段位表达")
    if any(pattern.search(text) for pattern in _RANK_PATTERNS):
        signals.append("游戏排名表达")
    if _LATIN_REPEAT_PATTERN.search(text):
        signals.append("中英混搭问答")
    return signals


def decide_meme(
    text: str,
    matched_memes: list[dict[str, str]],
    model_label: str,
    model_yes_probability: float,
) -> dict[str, Any]:
    """Combine dictionary evidence, new-meme patterns, and model confidence."""
    if matched_memes:
        return {
            "has_meme": "yes",
            "decision_reason": "词典命中",
            "novel_signals": [],
            "confidence": 0.99,
        }

    signals = unknown_meme_signals(text)
    if signals:
        return {
            "has_meme": "yes",
            "decision_reason": "新梗模式识别",
            "novel_signals": signals,
            "confidence": 0.82,
        }

    if model_label == "yes" and model_yes_probability >= MODEL_YES_THRESHOLD:
        return {
            "has_meme": "yes",
            "decision_reason": "模型置信判断",
            "novel_signals": [],
            "confidence": round(model_yes_probability, 4),
        }

    return {
        "has_meme": "no",
        "decision_reason": "未发现热梗证据",
        "novel_signals": [],
        "confidence": round(1.0 - model_yes_probability, 4),
    }
