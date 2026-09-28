from __future__ import annotations

import argparse
import csv
import io
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path


ENCODINGS = ("utf-8-sig", "gb18030")
SPLIT_RATIOS = OrderedDict((("train", 0.70), ("dev", 0.15), ("test", 0.15)))
SPLIT_NAMES = tuple(SPLIT_RATIOS)
DEFAULT_SPLIT_SEED = 2212
CLEAN_FIELDS = (
    "record_id",
    "text",
    "meme",
    "sentiment",
    "has_meme",
    "meme_form",
    "category",
    "meaning",
    "example",
    "note",
    "source",
    "split_group",
    "split",
)


def decode_text(data: bytes) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", data, 0, len(data), "unsupported encoding")


def read_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    text, encoding = decode_text(path.read_bytes())
    rows = [
        {key: (value or "").strip() for key, value in row.items()}
        for row in csv.DictReader(io.StringIO(text))
    ]
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows, encoding


def read_optional_csv(path: Path) -> tuple[list[dict[str, str]], str]:
    if not path.is_file():
        return [], "missing"
    return read_csv(path)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_sentiment(value: str) -> str:
    normalized = value.strip()
    if normalized == "中立":
        return "中性"
    return normalized


def merge_dictionary(
    dictionary: list[dict[str, str]],
    overrides: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in [*dictionary, *overrides]:
        meme = row["meme"].strip()
        if not meme or meme in seen:
            continue
        merged.append(
            {
                "meme": meme,
                "category": row["category"].strip(),
                "meaning": row["meaning"].strip(),
                "example": row["example"].strip(),
            }
        )
        seen.add(meme)
    return merged


def build_clean_records(
    hot_rows: list[dict[str, str]],
    negative_rows: list[dict[str, str]],
    dictionary: list[dict[str, str]] | dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    if isinstance(dictionary, dict):
        dictionary_by_meme = dictionary
    else:
        dictionary_by_meme = {row["meme"]: row for row in dictionary}

    records: list[dict[str, str]] = []
    for source, rows, prefix in (
        ("hot_meme", hot_rows, "hot"),
        ("normal", negative_rows, "neg"),
    ):
        for row in rows:
            text = row["text"].strip()
            meme = row["meme"].strip()
            has_meme = "yes" if meme and meme != "无" else "no"
            dictionary_row = dictionary_by_meme.get(meme, {})

            if has_meme == "yes":
                meme_form = "literal" if meme in text else "variant"
                split_group = f"MEME::{meme}"
            else:
                meme_form = "none"
                split_group = f"NEG::{text}"

            records.append(
                {
                    "record_id": f"{prefix}-{row['id'].strip()}",
                    "text": text,
                    "meme": meme,
                    "sentiment": normalize_sentiment(row["sentiment"]),
                    "has_meme": has_meme,
                    "meme_form": meme_form,
                    "category": dictionary_row.get("category", ""),
                    "meaning": dictionary_row.get("meaning", ""),
                    "example": dictionary_row.get("example", ""),
                    "note": row.get("note", "").strip(),
                    "source": source,
                    "split_group": split_group,
                    "split": "",
                }
            )
    return records


def assign_splits(
    records: list[dict[str, str]],
    ratios: dict[str, float] | None = None,
    seed: int = DEFAULT_SPLIT_SEED,
    _stratify_source: bool = True,
) -> list[dict[str, str]]:
    ratios = ratios or dict(SPLIT_RATIOS)
    ratio_total = sum(ratios.values())
    if abs(ratio_total - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")

    if _stratify_source:
        records_by_source: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        for record in records:
            records_by_source.setdefault(record["has_meme"], []).append(record)
        if len(records_by_source) > 1:
            split_records: list[dict[str, str]] = []
            for offset, source_records in enumerate(records_by_source.values()):
                split_records.extend(
                    assign_splits(
                        source_records,
                        ratios,
                        seed + offset,
                        _stratify_source=False,
                    )
                )
            return split_records

    groups: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for record in records:
        groups.setdefault(record["split_group"], []).append(dict(record))

    group_items = []
    for group, group_records in groups.items():
        label_counts = Counter(
            (row["has_meme"], row["sentiment"]) for row in group_records
        )
        group_items.append(
            {
                "group": group,
                "label_counts": label_counts,
                "records": group_records,
            }
        )

    random.Random(seed).shuffle(group_items)

    stratum_totals = Counter()
    for item in group_items:
        stratum_totals.update(item["label_counts"])
    current = {
        key: {split: 0 for split in ratios}
        for key in stratum_totals
    }
    totals_by_split = {split: 0 for split in ratios}
    overall_target = {
        split: len(records) * ratio
        for split, ratio in ratios.items()
    }

    for item in group_items:
        label_counts = item["label_counts"]
        group_size = len(item["records"])

        def split_score(split: str) -> tuple[float, float, int]:
            projected_size = totals_by_split[split] + group_size
            size_error = projected_size / max(1.0, overall_target[split])
            label_error = sum(
                (
                    (
                        current[key][split] + count
                        - stratum_totals[key] * ratios[split]
                    )
                    / max(1.0, stratum_totals[key])
                )
                ** 2
                for key, count in label_counts.items()
            ) / max(1, len(label_counts))
            split_share = totals_by_split[split] / max(1, len(records))
            return size_error + 0.05 * label_error, split_share, SPLIT_NAMES.index(split)

        selected_split = min(ratios, key=split_score)
        for record in item["records"]:
            record["split"] = selected_split
        for key, count in label_counts.items():
            current[key][selected_split] += count
        totals_by_split[selected_split] += group_size

    return [record for item in group_items for record in item["records"]]


def summarize(records: list[dict[str, str]]) -> dict[str, object]:
    split_groups = {
        split: {
            record["split_group"]
            for record in records
            if record["split"] == split
        }
        for split in SPLIT_NAMES
    }
    summary: dict[str, object] = {
        "total": len(records),
        "splits": {},
        "sentiment": dict(Counter(row["sentiment"] for row in records)),
        "sources": dict(Counter(row["source"] for row in records)),
        "has_meme": dict(Counter(row["has_meme"] for row in records)),
        "meme_form": dict(Counter(row["meme_form"] for row in records)),
    }
    for split in SPLIT_NAMES:
        split_records = [row for row in records if row["split"] == split]
        summary["splits"][split] = {
            "records": len(split_records),
            "split_groups": len(split_groups[split]),
            "hot_meme": sum(row["source"] == "hot_meme" for row in split_records),
            "normal": sum(row["source"] == "normal" for row in split_records),
            "sentiment": dict(Counter(row["sentiment"] for row in split_records)),
        }
    return summary


def write_quality_report(
    path: Path,
    summary: dict[str, object],
    dictionary_rows: list[dict[str, str]],
    raw_encodings: dict[str, str],
) -> None:
    lines = [
        "# 数据质量与划分报告",
        "",
        "## 原始文件编码",
        "",
    ]
    lines.extend(f"- `{name}`：{encoding}" for name, encoding in raw_encodings.items())
    lines.extend(
        [
            "",
            "## 清洗规则",
            "",
            "- 情绪标签 `中立` 统一为 `中性`。",
            "- 原句包含热梗原词时标记为 `literal`，否则标记为 `variant`。",
            "- 正常样本的 `meme` 统一标记为 `无`。",
            "- 同一热梗的所有样本只进入一个数据划分，避免分组泄漏。",
            "",
            "## 全量统计",
            "",
            f"- 总样本数：{summary['total']}",
            f"- 热梗词表数量：{len(dictionary_rows)}",
            f"- 来源分布：{summary['sources']}",
            f"- 是否热梗：{summary['has_meme']}",
            f"- 热梗形式：{summary['meme_form']}",
            f"- 情绪分布：{summary['sentiment']}",
            "",
            "## 数据划分",
            "",
        ]
    )
    for split, split_summary in summary["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- 样本数：{split_summary['records']}",
                f"- 独立热梗/文本组：{split_summary['split_groups']}",
                f"- 热梗样本：{split_summary['hot_meme']}",
                f"- 普通样本：{split_summary['normal']}",
                f"- 情绪分布：{split_summary['sentiment']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_required_fields(
    rows: list[dict[str, str]],
    required: tuple[str, ...],
    filename: str,
) -> None:
    if not rows:
        raise ValueError(f"{filename} contains no rows")
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise ValueError(f"{filename} missing fields: {', '.join(missing)}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Clean and split the meme dataset.")
    parser.add_argument("--raw-dir", type=Path, default=project_root.parent)
    parser.add_argument(
        "--override-file",
        type=Path,
        default=project_root / "data" / "manual" / "dictionary_overrides.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "processed",
    )
    parser.add_argument(
        "--expansion-dir",
        type=Path,
        default=project_root / "data" / "expansion",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "reports" / "data_quality_report.md",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = parser.parse_args()

    hot_rows, hot_encoding = read_csv(args.raw_dir / "hot_meme_dataset.csv")
    negative_rows, negative_encoding = read_csv(
        args.raw_dir / "normal_negative_samples.csv"
    )
    dictionary_rows, dictionary_encoding = read_csv(
        args.raw_dir / "meme_dictionary.csv"
    )
    extra_hot_rows, extra_hot_encoding = read_optional_csv(
        args.expansion_dir / "hot_meme_dataset.csv"
    )
    extra_negative_rows, extra_negative_encoding = read_optional_csv(
        args.expansion_dir / "normal_samples.csv"
    )
    extra_dictionary_rows, extra_dictionary_encoding = read_optional_csv(
        args.expansion_dir / "meme_dictionary.csv"
    )
    override_rows, override_encoding = read_csv(args.override_file)

    validate_required_fields(
        hot_rows, ("id", "text", "meme", "sentiment", "note"), "hot_meme_dataset.csv"
    )
    validate_required_fields(
        negative_rows,
        ("id", "text", "meme", "sentiment", "note"),
        "normal_negative_samples.csv",
    )
    validate_required_fields(
        dictionary_rows,
        ("meme", "category", "meaning", "example"),
        "meme_dictionary.csv",
    )
    for rows, required, filename in (
        (
            extra_hot_rows,
            ("id", "text", "meme", "sentiment", "note"),
            "expansion/hot_meme_dataset.csv",
        ),
        (
            extra_negative_rows,
            ("id", "text", "meme", "sentiment", "note"),
            "expansion/normal_samples.csv",
        ),
        (
            extra_dictionary_rows,
            ("meme", "category", "meaning", "example"),
            "expansion/meme_dictionary.csv",
        ),
    ):
        if rows:
            validate_required_fields(rows, required, filename)
    validate_required_fields(
        override_rows,
        ("meme", "category", "meaning", "example"),
        args.override_file.name,
    )

    hot_rows.extend(extra_hot_rows)
    negative_rows.extend(extra_negative_rows)
    dictionary_rows.extend(extra_dictionary_rows)
    merged_dictionary = merge_dictionary(dictionary_rows, override_rows)
    clean_records = build_clean_records(
        hot_rows, negative_rows, merged_dictionary
    )
    split_records = assign_splits(clean_records, seed=args.seed)
    summary = summarize(split_records)

    group_to_split: dict[str, str] = {}
    for record in split_records:
        previous = group_to_split.setdefault(record["split_group"], record["split"])
        if previous != record["split"]:
            raise ValueError(f"split leakage detected: {record['split_group']}")

    write_csv(
        args.output_dir / "all_clean.csv",
        split_records,
        CLEAN_FIELDS,
    )
    for split in SPLIT_NAMES:
        write_csv(
            args.output_dir / f"{split}.csv",
            [record for record in split_records if record["split"] == split],
            CLEAN_FIELDS,
        )
    write_csv(
        args.output_dir / "dictionary_clean.csv",
        merged_dictionary,
        ("meme", "category", "meaning", "example"),
    )

    summary["seed"] = args.seed
    summary["raw_encodings"] = {
        "hot_meme_dataset.csv": hot_encoding,
        "expansion/hot_meme_dataset.csv": extra_hot_encoding,
        "normal_negative_samples.csv": negative_encoding,
        "expansion/normal_samples.csv": extra_negative_encoding,
        "meme_dictionary.csv": dictionary_encoding,
        "expansion/meme_dictionary.csv": extra_dictionary_encoding,
        args.override_file.name: override_encoding,
    }
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_quality_report(
        args.report,
        summary,
        merged_dictionary,
        {
            "hot_meme_dataset.csv": hot_encoding,
            "expansion/hot_meme_dataset.csv": extra_hot_encoding,
            "normal_negative_samples.csv": negative_encoding,
            "expansion/normal_samples.csv": extra_negative_encoding,
            "meme_dictionary.csv": dictionary_encoding,
            "expansion/meme_dictionary.csv": extra_dictionary_encoding,
            args.override_file.name: override_encoding,
        },
    )

    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))
    print(f"dictionary: {len(merged_dictionary)}")


if __name__ == "__main__":
    main()
