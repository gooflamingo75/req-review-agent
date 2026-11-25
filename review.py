#!/usr/bin/env python3
"""Запуск проверки документа требований одной командой.

    python review.py <файл требований> <файл отчёта>

Пример:
    python review.py docs/requirements.md reports/report.md
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Windows-консоль: включаем UTF-8, чтобы русский текст не превращался в «кракозябры»
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def agent_version() -> str:
    from reqreview import __version__
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--always", "--dirty"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = ""
    return f"v{__version__}" + (f" (коммит {commit})" if commit else " (коммит неизвестен: запуск не из git-репозитория)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ИИ-агент статической проверки документа требований (ISO/IEC/IEEE 29148, IEEE 830).")
    p.add_argument("input", help="путь к документу требований (.pdf, .docx, .md, .txt)")
    p.add_argument("output", help="путь к файлу отчёта (.md); рядом будет создан .json")
    p.add_argument("--model", help="модель Gemini (по умолчанию из .env или gemini-3.8-flash)")
    p.add_argument("--batch-size", type=int, default=None, help="сколько требований проверять за один запрос (по умолчанию 8)")
    p.add_argument("--mock", action="store_true", help="демо-режим без ключа и без LLM (только правила) — для проверки установки")
    a = p.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        load_dotenv()  # .env в текущей папке, если есть
    except ImportError:
        pass

    from reqreview.agent import run
    from reqreview.llm import GeminiLLM, LLMError, MockLLM, OpenAICompatLLM
    from reqreview.loader import load_document
    from reqreview.report import write

    try:
        doc = load_document(a.input)
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА чтения документа: {e}", file=sys.stderr)
        return 2
    print(f"Документ: «{doc.title}» ({doc.fmt}, {len(doc.nonempty())} непустых строк)", file=sys.stderr)

    mock = a.mock or os.getenv("LLM_PROVIDER", "").lower() == "mock"
    try:
        provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        if mock:
            llm = MockLLM()
        elif provider == "openai":
            llm = OpenAICompatLLM(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                  os.getenv("OPENAI_API_KEY", ""), a.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                  "OpenAI-совместимый API")
        elif provider == "ollama":
            llm = OpenAICompatLLM(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1", "",
                                  a.model or os.getenv("OLLAMA_MODEL", ""), "Ollama (локально)")
        else:
            llm = GeminiLLM(
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
                model=a.model or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash",
                min_interval=float(os.getenv("REQUEST_INTERVAL_SEC", "6")),
                max_retries=int(os.getenv("MAX_RETRIES", "6")),
            )
        print(f"Модель: {llm.name}", file=sys.stderr)
        batch = a.batch_size or int(os.getenv("BATCH_SIZE", "8"))
        result = run(doc, llm, batch_size=max(1, batch))
    except LLMError as e:
        print(f"\nОШИБКА модели: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        return 130

    try:
        shown = os.path.relpath(Path(a.input).resolve(), Path.cwd())
    except ValueError:
        shown = Path(a.input).name
    files = write(result, a.output, agent_version(), shown)
    s = result.stats
    print("\nГотово!", file=sys.stderr)
    print(f"  Требований: {s['requirements']}, нарушений: {s['findings']} "
          f"(критичных: {s['by_severity']['critical']})", file=sys.stderr)
    print(f"  Отчёт: {files[0].resolve()}", file=sys.stderr)
    print(f"  Данные (JSON): {files[1].resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
