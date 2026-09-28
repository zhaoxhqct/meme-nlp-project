import tempfile
import unittest
from pathlib import Path

from scripts.dictionary_store import (
    DuplicateDictionaryEntry,
    append_dictionary_entry,
)
from scripts.meme_decision import decide_meme, unknown_meme_signals
from scripts.meme_lexicon import MemeLexicon
from scripts.model_pipeline import metric_payload


class MemeLexiconTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "dictionary.csv"
        path.write_text(
            "meme,category,meaning,example\n"
            "yyds,缩写类,永远的神,这个视频yyds\n"
            "emo,情绪状态类,情绪低落,今天emo了\n",
            encoding="utf-8-sig",
        )
        self.lexicon = MemeLexicon.from_csv(path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_find_literal_matches_and_explanations(self) -> None:
        matches = self.lexicon.find_all("这部电影yyds，但我有点emo。")
        self.assertEqual(["yyds", "emo"], [item.meme for item in matches])
        self.assertEqual(
            ["永远的神", "情绪低落"],
            [item["meaning"] for item in self.lexicon.explain("yyds emo")],
        )

    def test_enrich_text_handles_no_match(self) -> None:
        self.assertTrue(self.lexicon.enrich("普通句子").endswith("[LEX] NONE"))


class MetricTests(unittest.TestCase):
    def test_metric_payload_contains_required_scores(self) -> None:
        result = metric_payload(["yes", "no"], ["yes", "yes"])
        self.assertEqual(0.5, result["accuracy"])
        self.assertIn("per_class", result)


class MemeDecisionTests(unittest.TestCase):
    def test_common_greeting_is_not_a_meme(self) -> None:
        result = decide_meme("你好", [], "no", 0.002)
        self.assertEqual("no", result["has_meme"])

    def test_unlisted_season_rank_phrase_uses_new_meme_pattern(self) -> None:
        result = decide_meme("s6第一个王者", [], "no", 0.015)
        self.assertEqual("yes", result["has_meme"])
        self.assertEqual("新梗模式识别", result["decision_reason"])
        self.assertIn("赛季或段位表达", result["novel_signals"])

    def test_ordinary_game_sentence_is_not_a_meme(self) -> None:
        result = decide_meme("我的王者段位是钻石", [], "no", 0.009)
        self.assertEqual("no", result["has_meme"])

    def test_dictionary_match_has_priority(self) -> None:
        result = decide_meme(
            "这部电影yyds",
            [{"meme": "yyds"}],
            "no",
            0.1,
        )
        self.assertEqual("yes", result["has_meme"])
        self.assertEqual("词典命中", result["decision_reason"])

    def test_latin_repeat_pattern_is_generalized(self) -> None:
        self.assertIn(
            "中英混搭问答",
            unknown_meme_signals("今天city不city"),
        )


class DictionaryStoreTests(unittest.TestCase):
    def test_dictionary_entry_is_written_in_required_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dictionary_overrides.csv"
            entry = append_dictionary_entry(
                path,
                {
                    "meme": "s6第一个王者",
                    "category": "游戏类",
                    "meaning": "S6赛季首个王者段位",
                    "example": "s6第一个王者",
                },
            )
            self.assertEqual("s6第一个王者", entry["meme"])
            self.assertIn(
                "meme,category,meaning,example",
                path.read_text(encoding="utf-8-sig"),
            )

    def test_duplicate_dictionary_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "dictionary_overrides.csv"
            payload = {
                "meme": "s6第一个王者",
                "category": "游戏类",
                "meaning": "S6赛季首个王者段位",
                "example": "s6第一个王者",
            }
            append_dictionary_entry(path, payload)
            with self.assertRaises(DuplicateDictionaryEntry):
                append_dictionary_entry(path, payload)


if __name__ == "__main__":
    unittest.main()
