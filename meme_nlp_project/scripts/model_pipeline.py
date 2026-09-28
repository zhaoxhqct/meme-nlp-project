from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from scripts.meme_lexicon import MemeLexicon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MODEL_CACHE_DIR = PROJECT_ROOT / ".hf_cache"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def is_hot_meme(row: dict[str, str]) -> bool:
    return row.get("has_meme", "").strip() == "yes"


def load_splits(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    return (
        read_csv(processed_dir / "train.csv"),
        read_csv(processed_dir / "dev.csv"),
        read_csv(processed_dir / "test.csv"),
    )


class LexiconHitFeatures(BaseEstimator, TransformerMixin):
    """Sparse indicators for dictionary terms found in each sentence."""

    def __init__(self, lexicon: MemeLexicon):
        self.lexicon = lexicon

    def fit(self, texts: Iterable[str], y: Any = None) -> "LexiconHitFeatures":
        self.n_features_ = len(self.lexicon.entries)
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        from scipy.sparse import csr_matrix

        texts = list(texts)
        keys = sorted(self.lexicon.entries)
        index = {meme: position for position, meme in enumerate(keys)}
        rows: list[int] = []
        columns: list[int] = []
        for row, text in enumerate(texts):
            for match in self.lexicon.find_all(text):
                column = index.get(match.meme.lower())
                if column is not None:
                    rows.append(row)
                    columns.append(column)
        values = np.ones(len(rows), dtype=float)
        return csr_matrix(
            (values, (rows, columns)),
            shape=(len(texts), len(keys)),
        )


def build_tfidf_pipeline(
    lexicon: MemeLexicon,
    regularization: float = 3.0,
    class_weight: str | None = None,
) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 4),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "lexicon",
                Pipeline(
                    [
                        ("enrich", EnrichText(lexicon)),
                        ("hits", LexiconHitFeatures(lexicon)),
                    ]
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=regularization,
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=20260928,
                ),
            ),
        ]
    )


class EnrichText(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon: MemeLexicon):
        self.lexicon = lexicon

    def fit(self, texts: Iterable[str], y: Any = None) -> "EnrichText":
        self.fitted_ = True
        return self

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self.lexicon.enrich(text) for text in texts]


def metric_payload(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_tfidf(args: argparse.Namespace) -> dict[str, Any]:
    import joblib

    lexicon = MemeLexicon.from_csv(PROCESSED_DIR / "dictionary_clean.csv")
    train_rows, dev_rows, test_rows = load_splits(args.processed_dir)

    train_texts = [row["text"] for row in train_rows]
    train_has = ["yes" if is_hot_meme(row) else "no" for row in train_rows]
    has_model = build_tfidf_pipeline(lexicon)
    has_model.fit(train_texts, train_has)

    hot_train = [row for row in train_rows if is_hot_meme(row)]
    sentiment_model = build_tfidf_pipeline(
        lexicon,
        class_weight="balanced",
    )
    sentiment_model.fit(
        [row["text"] for row in hot_train],
        [row["sentiment"] for row in hot_train],
    )

    test_texts = [row["text"] for row in test_rows]
    has_pred = [
        "yes" if lexicon.find_all(text) else "no"
        for text in test_texts
    ]
    sentiment_pred_all = sentiment_model.predict(test_texts).tolist()
    sentiment_mask = [is_hot_meme(row) for row in test_rows]
    sentiment_true = [
        row["sentiment"] for row in test_rows if is_hot_meme(row)
    ]
    sentiment_pred = [
        prediction
        for prediction, keep in zip(sentiment_pred_all, sentiment_mask)
        if keep
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(has_model, output_dir / "has_meme.joblib")
    joblib.dump(sentiment_model, output_dir / "sentiment.joblib")

    predictions = build_prediction_rows(
        test_rows,
        has_pred,
        sentiment_pred_all,
        lexicon,
    )
    write_predictions(output_dir / "test_predictions.csv", predictions)

    metrics = {
        "model": "tfidf_lexicon",
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "meme_recognition": metric_payload(
            ["yes" if is_hot_meme(row) else "no" for row in test_rows],
            has_pred,
        ),
        "sentiment": metric_payload(sentiment_true, sentiment_pred),
        "label_distribution": {
            "has_meme": dict(Counter(train_has)),
            "sentiment": dict(Counter(row["sentiment"] for row in hot_train)),
        },
        "output_dir": str(output_dir),
    }
    return metrics


def build_prediction_rows(
    rows: list[dict[str, str]],
    has_pred: list[str],
    sentiment_pred: list[str],
    lexicon: MemeLexicon,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for row, has_value, sentiment in zip(rows, has_pred, sentiment_pred):
        explanations = lexicon.explain(row["text"])
        results.append(
            {
                "record_id": row["record_id"],
                "text": row["text"],
                "gold_has_meme": "yes" if is_hot_meme(row) else "no",
                "pred_has_meme": has_value,
                "gold_sentiment": row["sentiment"] if is_hot_meme(row) else "",
                "pred_sentiment": sentiment if has_value == "yes" else "",
                "matched_memes": "|".join(item["meme"] for item in explanations),
                "meanings": "|".join(item["meaning"] for item in explanations),
            }
        )
    return results


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_roberta(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    class TextDataset(Dataset):
        def __init__(
            self,
            texts: list[str],
            labels: list[int],
            tokenizer: Any,
        ):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {
                key: value[index]
                for key, value in self.encodings.items()
            }
            item["labels"] = self.labels[index]
            return item

    def predict_model(
        model: Any,
        texts: list[str],
    ) -> list[int]:
        model.eval()
        tokenizer = model.tokenizer
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            inputs = {
                key: value.to(device)
                for key, value in encodings.items()
            }
            return model(**inputs).logits.argmax(dim=-1).cpu().tolist()

    def fit_task(
        texts: list[str],
        labels: list[str],
        task_dir: Path,
        use_class_weights: bool = False,
    ) -> tuple[Any, Any, list[str], dict[str, float]]:
        label_names = sorted(set(labels))
        label_to_id = {
            label: index for index, label in enumerate(label_names)
        }
        class_weight_map: dict[str, float] = {}
        class_weight_tensor = None
        if use_class_weights:
            label_counts = Counter(labels)
            weights = [
                len(labels) / (len(label_names) * label_counts[label])
                for label in label_names
            ]
            class_weight_map = {
                label: round(weight, 4)
                for label, weight in zip(label_names, weights)
            }
            class_weight_tensor = torch.tensor(
                weights,
                dtype=torch.float,
                device=device,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.pretrained_model,
            num_labels=len(label_names),
        ).to(device)
        model.config.id2label = {
            index: label for index, label in enumerate(label_names)
        }
        model.config.label2id = {
            label: index for index, label in enumerate(label_names)
        }
        dataset = TextDataset(
            texts,
            [label_to_id[label] for label in labels],
            tokenizer,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )
        model.train()
        for _ in range(args.epochs):
            for batch in loader:
                batch_labels = batch.pop("labels").to(device)
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }
                optimizer.zero_grad()
                output = model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    output.logits,
                    batch_labels,
                    weight=class_weight_tensor,
                )
                loss.backward()
                optimizer.step()
        model.tokenizer = tokenizer
        model.save_pretrained(task_dir)
        tokenizer.save_pretrained(task_dir)
        return model, tokenizer, label_names, class_weight_map

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    lexicon = MemeLexicon.from_csv(PROCESSED_DIR / "dictionary_clean.csv")
    train_rows, _dev_rows, test_rows = load_splits(args.processed_dir)

    has_train_texts = [lexicon.enrich(row["text"]) for row in train_rows]
    has_train_labels = [
        "yes" if is_hot_meme(row) else "no" for row in train_rows
    ]
    has_dir = args.output_dir / "has_meme"
    has_dir.mkdir(parents=True, exist_ok=True)
    has_model, _, has_labels, _ = fit_task(
        has_train_texts,
        has_train_labels,
        has_dir,
    )

    hot_train = [row for row in train_rows if is_hot_meme(row)]
    sentiment_model, _, sentiment_labels, sentiment_class_weights = fit_task(
        [lexicon.enrich(row["text"]) for row in hot_train],
        [row["sentiment"] for row in hot_train],
        args.output_dir / "sentiment",
        use_class_weights=True,
    )

    test_texts = [row["text"] for row in test_rows]
    test_inputs = [lexicon.enrich(text) for text in test_texts]
    has_ids = predict_model(has_model, test_inputs)
    sentiment_ids_all = predict_model(sentiment_model, test_inputs)
    has_pred = [has_labels[index] for index in has_ids]
    sentiment_pred_all = [
        sentiment_labels[index] for index in sentiment_ids_all
    ]
    sentiment_true = [
        row["sentiment"] for row in test_rows if is_hot_meme(row)
    ]
    sentiment_pred = [
        prediction
        for prediction, row in zip(sentiment_pred_all, test_rows)
        if is_hot_meme(row)
    ]
    predictions = build_prediction_rows(
        test_rows,
        has_pred,
        sentiment_pred_all,
        lexicon,
    )
    write_predictions(args.output_dir / "test_predictions.csv", predictions)

    metrics = {
        "model": "roberta_lexicon",
        "pretrained_model": args.pretrained_model,
        "device": str(device),
        "epochs": args.epochs,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "meme_recognition": metric_payload(
            ["yes" if is_hot_meme(row) else "no" for row in test_rows],
            has_pred,
        ),
        "sentiment": metric_payload(sentiment_true, sentiment_pred),
        "label_distribution": {
            "has_meme": dict(Counter(has_train_labels)),
            "sentiment": dict(
                Counter(row["sentiment"] for row in hot_train)
            ),
            "sentiment_class_weights": sentiment_class_weights,
        },
        "output_dir": str(args.output_dir),
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Chinese meme recognition models."
    )
    parser.add_argument(
        "--model",
        choices=("tfidf", "roberta"),
        default="tfidf",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pretrained-model",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "models" / args.model
    metrics = (
        train_tfidf(args)
        if args.model == "tfidf"
        else train_roberta(args)
    )
    metrics_path = PROJECT_ROOT / "reports" / f"{args.model}_metrics.json"
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nMetrics: {metrics_path}")


if __name__ == "__main__":
    main()
