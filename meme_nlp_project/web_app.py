from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scripts.dictionary_store import (
    DuplicateDictionaryEntry,
    append_dictionary_entry,
)
from scripts.meme_decision import decide_meme
from scripts.meme_lexicon import MemeLexicon
from scripts.model_pipeline import MODEL_CACHE_DIR, PROCESSED_DIR, PROJECT_ROOT


WEB_DIR = PROJECT_ROOT / "web"
MODEL_DIR = PROJECT_ROOT / "models" / "roberta"
DICTIONARY_OVERRIDE_PATH = Path(
    os.environ.get(
        "MEME_DICTIONARY_PATH",
        str(
            PROJECT_ROOT
            / "data"
            / "manual"
            / "dictionary_overrides.csv"
        ),
    )
)
MAX_TEXT_LENGTH = 300
MAX_BODY_SIZE = 10_000


class Predictor:
    def __init__(self) -> None:
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._tokenizer_class = AutoTokenizer
        self._model_class = AutoModelForSequenceClassification
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: dict[str, tuple[Any, Any]] = {}
        self._lexicon = MemeLexicon.from_csvs(
            [
                PROCESSED_DIR / "dictionary_clean.csv",
                DICTIONARY_OVERRIDE_PATH,
            ]
        )
        self._dictionary_lock = threading.Lock()

    def _load_task(self, task: str) -> tuple[Any, Any]:
        if task not in self._models:
            task_dir = MODEL_DIR / task
            tokenizer = self._tokenizer_class.from_pretrained(task_dir)
            model = self._model_class.from_pretrained(task_dir)
            model.to(self._device)
            model.eval()
            self._models[task] = (tokenizer, model)
        return self._models[task]

    def _predict_task(
        self,
        task: str,
        text: str,
    ) -> tuple[str, float]:
        tokenizer, model = self._load_task(task)
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }
        with self._torch.no_grad():
            logits = model(**inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)[0]
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

    def predict(self, text: str) -> dict[str, Any]:
        model_text = self._lexicon.enrich(text)
        model_label, yes_probability = self._predict_task(
            "has_meme",
            model_text,
        )
        matched_memes = self._lexicon.explain(text)
        decision = decide_meme(
            text,
            matched_memes,
            model_label,
            yes_probability,
        )
        has_meme = str(decision["has_meme"])
        sentiment = (
            self._predict_task("sentiment", model_text)[0]
            if has_meme == "yes"
            else ""
        )
        return {
            "text": text,
            "has_meme": has_meme,
            "sentiment": sentiment,
            "matched_memes": matched_memes,
            **decision,
        }

    def add_dictionary_entry(
        self,
        payload: dict[str, Any],
    ) -> dict[str, str]:
        meme = str(payload.get("meme", "")).strip()
        if meme.casefold() in self._lexicon.entries:
            raise DuplicateDictionaryEntry("这个词条已经存在。")

        with self._dictionary_lock:
            entry = append_dictionary_entry(
                DICTIONARY_OVERRIDE_PATH,
                payload,
            )
            self._lexicon.add(
                entry["meme"],
                entry["category"],
                entry["meaning"],
            )
        return entry


class AppHandler(BaseHTTPRequestHandler):
    predictor = Predictor()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"

        relative_path = path.lstrip("/")
        candidate = (WEB_DIR / relative_path).resolve()
        web_root = WEB_DIR.resolve()
        if candidate != web_root and web_root not in candidate.parents:
            self._send_json(
                {"error": "资源不存在。"},
                HTTPStatus.NOT_FOUND,
            )
            return
        if not candidate.is_file():
            self._send_json(
                {"error": "资源不存在。"},
                HTTPStatus.NOT_FOUND,
            )
            return

        content = candidate.read_bytes()
        content_type = (
            mimetypes.guess_type(candidate.name)[0]
            or "application/octet-stream"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/predict", "/api/dictionary"):
            self._send_json(
                {"error": "接口不存在。"},
                HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > MAX_BODY_SIZE:
                raise ValueError("请求内容长度不正确。")
            payload = json.loads(
                self.rfile.read(content_length).decode("utf-8")
            )
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("请输入需要分析的中文文本。")
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError(
                    f"文本最多 {MAX_TEXT_LENGTH} 个字符。"
                )
            if path == "/api/dictionary":
                entry = self.predictor.add_dictionary_entry(
                    {**payload, "example": text}
                )
                result = {
                    "message": "新梗已加入用户词典。",
                    "entry": entry,
                }
            else:
                result = self.predictor.predict(text)
        except DuplicateDictionaryEntry as error:
            self._send_json(
                {"error": str(error)},
                HTTPStatus.CONFLICT,
            )
            return
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self._send_json(
                {"error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
            return
        except Exception:
            traceback.print_exc()
            self._send_json(
                {"error": "模型分析失败，请查看启动窗口中的错误信息。"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(result, HTTPStatus.OK)

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus,
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[网页] {self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the local meme web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def create_server(host: str, preferred_port: int) -> ThreadingHTTPServer:
    last_error: OSError | None = None
    for port in range(preferred_port, preferred_port + 11):
        try:
            return ThreadingHTTPServer((host, port), AppHandler)
        except OSError as error:
            last_error = error
    raise RuntimeError("没有找到可用的本地端口。") from last_error


def main() -> None:
    args = parse_args()
    server = create_server(args.host, args.port)
    url = f"http://{args.host}:{server.server_port}/"
    print("中文网络热梗分析页面已启动：")
    print(url)
    print("关闭此窗口即可停止服务。")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
