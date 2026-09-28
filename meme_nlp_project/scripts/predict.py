from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scripts.meme_decision import decide_meme
from scripts.meme_lexicon import MemeLexicon
from scripts.model_pipeline import (
    MODEL_CACHE_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)


def predict_tfidf(model_dir: Path, text: str) -> dict[str, Any]:
    import joblib

    has_model = joblib.load(model_dir / "has_meme.joblib")
    sentiment_model = joblib.load(model_dir / "sentiment.joblib")
    matched = MemeLexicon.from_csv(
        PROCESSED_DIR / "dictionary_clean.csv"
    ).find_all(text)
    has_value = "yes" if matched else "no"
    sentiment = str(sentiment_model.predict([text])[0])
    return {
        "has_meme": has_value,
        "sentiment": sentiment if has_value == "yes" else "",
    }


def predict_roberta(
    model_dir: Path,
    text: str,
    max_length: int,
    matched_memes: list[dict[str, str]],
) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_task(task_dir: Path) -> tuple[str, float]:
        tokenizer = AutoTokenizer.from_pretrained(task_dir)
        model = AutoModelForSequenceClassification.from_pretrained(task_dir)
        model.to(device)
        model.eval()
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]
            label_id = int(logits.argmax(dim=-1).item())
        label = str(model.config.id2label[label_id])
        yes_index = next(
            (
                int(index)
                for index, name in model.config.id2label.items()
                if str(name).lower() == "yes"
            ),
            label_id,
        )
        return label, float(probabilities[yes_index].item())

    has_value, yes_probability = predict_task(model_dir / "has_meme")
    sentiment = predict_task(model_dir / "sentiment")[0]
    decision = decide_meme(
        text,
        matched_memes,
        has_value,
        yes_probability,
    )
    return {
        "sentiment": (
            sentiment if decision["has_meme"] == "yes" else ""
        ),
        **decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one meme sentence.")
    parser.add_argument("--model", choices=("tfidf", "roberta"), required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    model_dir = args.model_dir or PROJECT_ROOT / "models" / args.model
    lexicon = MemeLexicon.from_csvs(
        [
            PROCESSED_DIR / "dictionary_clean.csv",
            PROJECT_ROOT
            / "data"
            / "manual"
            / "dictionary_overrides.csv",
        ]
    )
    for text in args.text:
        matched_memes = lexicon.explain(text)
        model_text = (
            lexicon.enrich(text)
            if args.model == "roberta"
            else text
        )
        prediction = (
            predict_tfidf(model_dir, model_text)
            if args.model == "tfidf"
            else predict_roberta(
                model_dir,
                model_text,
                args.max_length,
                matched_memes,
            )
        )
        prediction["text"] = text
        prediction["matched_memes"] = matched_memes
        print(json.dumps(prediction, ensure_ascii=False))


if __name__ == "__main__":
    main()
