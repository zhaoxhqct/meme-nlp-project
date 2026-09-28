import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import prepare_data


class PrepareDataTests(unittest.TestCase):
    def test_decode_bytes_accepts_gb18030(self):
        text, encoding = prepare_data.decode_text("热梗：yyds".encode("gb18030"))

        self.assertEqual(text, "热梗：yyds")
        self.assertEqual(encoding, "gb18030")

    def test_decode_bytes_prefers_utf8(self):
        text, encoding = prepare_data.decode_text("热梗：yyds".encode("utf-8-sig"))

        self.assertEqual(text, "热梗：yyds")
        self.assertEqual(encoding, "utf-8-sig")

    def test_normalize_sentiment_maps_zhongli_to_neutral(self):
        self.assertEqual(prepare_data.normalize_sentiment("中立"), "中性")
        self.assertEqual(prepare_data.normalize_sentiment(" 正向 "), "正向")

    def test_merge_dictionary_adds_confirmed_override(self):
        dictionary = [
            {
                "meme": "yyds",
                "category": "缩写类",
                "meaning": "永远的神",
                "example": "yyds！",
            }
        ]
        overrides = [
            {
                "meme": "zqsg",
                "category": "缩写类",
                "meaning": "真情实感",
                "example": "追这个综艺追得zqsg。",
            }
        ]

        merged = prepare_data.merge_dictionary(dictionary, overrides)

        self.assertEqual([row["meme"] for row in merged], ["yyds", "zqsg"])

    def test_build_clean_records_marks_literal_and_variant_memes(self):
        dictionary = {
            "yyds": {
                "meme": "yyds",
                "category": "缩写类",
                "meaning": "永远的神",
                "example": "yyds！",
            },
            "挖坟": {
                "meme": "挖坟",
                "category": "社会现象类",
                "meaning": "翻出旧帖",
                "example": "又开始挖坟了。",
            },
        }
        hot_rows = [
            {
                "id": "1",
                "text": "这个视频水平yyds。",
                "meme": "yyds",
                "sentiment": "中立",
                "note": "示例",
            },
            {
                "id": "2",
                "text": "谁又把老帖挖出来了。",
                "meme": "挖坟",
                "sentiment": "中性",
                "note": "示例",
            },
        ]
        negative_rows = [
            {
                "id": "1",
                "text": "今天天气挺好。",
                "meme": "无",
                "sentiment": "中性",
                "note": "普通语句",
            }
        ]

        records = prepare_data.build_clean_records(hot_rows, negative_rows, dictionary)

        self.assertEqual(records[0]["sentiment"], "中性")
        self.assertEqual(records[0]["meme_form"], "literal")
        self.assertEqual(records[1]["meme_form"], "variant")
        self.assertEqual(records[2]["has_meme"], "no")
        self.assertEqual(records[2]["source"], "normal")

    def test_assign_splits_keeps_a_meme_group_in_one_split(self):
        records = []
        for group_index in range(10):
            for row_index in range(2):
                records.append(
                    {
                        "record_id": f"g{group_index}-r{row_index}",
                        "split_group": f"g{group_index}",
                        "has_meme": "yes",
                        "sentiment": "中性",
                    }
                )

        split_records = prepare_data.assign_splits(
            records,
            ratios={"train": 0.7, "dev": 0.15, "test": 0.15},
            seed=42,
        )

        self.assertEqual(len(split_records), len(records))
        groups_to_splits = {}
        for record in split_records:
            group = record["split_group"]
            groups_to_splits.setdefault(group, set()).add(record["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups_to_splits.values()))
        self.assertEqual(set(groups_to_splits), {f"g{i}" for i in range(10)})

    def test_assign_splits_allows_one_group_with_multiple_sentiments(self):
        records = [
            {
                "record_id": "m1-positive",
                "split_group": "M1",
                "has_meme": "yes",
                "sentiment": "正向",
            },
            {
                "record_id": "m1-negative",
                "split_group": "M1",
                "has_meme": "yes",
                "sentiment": "负向",
            },
            {
                "record_id": "m2-neutral",
                "split_group": "M2",
                "has_meme": "yes",
                "sentiment": "中性",
            },
        ]

        split_records = prepare_data.assign_splits(records, seed=42)

        m1_splits = {
            record["split"] for record in split_records if record["split_group"] == "M1"
        }
        self.assertEqual(len(m1_splits), 1)

    def test_assign_splits_uses_all_splits_for_unique_strata(self):
        sentiments = ("正向", "负向", "中性")
        records = [
            {
                "record_id": f"row-{index}",
                "split_group": f"group-{index}",
                "has_meme": "yes",
                "sentiment": sentiments[index % len(sentiments)],
            }
            for index in range(30)
        ]

        split_records = prepare_data.assign_splits(records, seed=42)
        split_counts = {
            split: sum(record["split"] == split for record in split_records)
            for split in prepare_data.SPLIT_NAMES
        }

        self.assertTrue(all(count > 0 for count in split_counts.values()))
        self.assertGreater(split_counts["train"], split_counts["dev"])
        self.assertGreater(split_counts["train"], split_counts["test"])

    def test_assign_splits_respects_global_ratios(self):
        records = [
            {
                "record_id": f"row-{index}",
                "split_group": f"group-{index}",
                "has_meme": "yes",
                "sentiment": "中性",
            }
            for index in range(100)
        ]

        split_records = prepare_data.assign_splits(records, seed=42)
        split_counts = {
            split: sum(record["split"] == split for record in split_records)
            for split in prepare_data.SPLIT_NAMES
        }

        self.assertEqual(split_counts, {"train": 70, "dev": 15, "test": 15})

    def test_assign_splits_balances_hot_and_normal_sources(self):
        records = []
        for index in range(100):
            records.append(
                {
                    "record_id": f"hot-{index}",
                    "split_group": f"HOT::{index}",
                    "has_meme": "yes",
                    "sentiment": "中性",
                }
            )
        for index in range(50):
            records.append(
                {
                    "record_id": f"normal-{index}",
                    "split_group": f"NEG::{index}",
                    "has_meme": "no",
                    "sentiment": "中性",
                }
            )

        split_records = prepare_data.assign_splits(records, seed=42)
        source_counts = {
            split: {
                source: sum(
                    record["split"] == split and record["has_meme"] == source
                    for record in split_records
                )
                for source in ("yes", "no")
            }
            for split in prepare_data.SPLIT_NAMES
        }

        self.assertEqual(source_counts["train"]["yes"], 70)
        self.assertIn(source_counts["train"]["no"], (34, 35, 36))
        self.assertTrue(all(count > 0 for count in source_counts["dev"].values()))
        self.assertTrue(all(count > 0 for count in source_counts["test"].values()))


if __name__ == "__main__":
    unittest.main()
