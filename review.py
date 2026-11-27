#!/usr/bin/env python3
"""Запуск проверки документа требований одной командой.

    python review.py <файл требований> <файл отчёта>

Создаются три файла: отчёт PDF (основной), тот же отчёт в Markdown и данные JSON.
Код завершения: 0 — проверка завершена полностью; 10 — частичная проверка (отчёт создан, ограничения указаны
в панели); 11 — в документе нет оцениваемых требований; 2 — документ нельзя прочитать (создан диагностический
отчёт); 3 — модель недоступна (создан диагностический отчёт); 4 — не удалось создать PDF (Markdown и JSON есть).

Пример:
    python review.py docs/requirements.md reports/report.md
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Windows-консоль: включаем UTF-8, чтобы русский текст не превращался в «кракозябры»
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _git_commit() -> str:
    """Читает текущий коммит прямо из .git (без вызова git — на macOS без Xcode
    вызов git открывает окно установки)."""
    g = ROOT / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head[:7]
        ref = head.split(":", 1)[1].strip()
        f = g / ref
        if f.exists():
            return f.read_text().strip()[:7]
        packed = g / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0][:7]
    except OSError:
        pass
    return ""


def agent_version() -> str:
    from reqreview import __version__
    commit = _git_commit()
    return f"v{__version__}" + (f" (коммит {commit})" if commit else " (коммит неизвестен: запуск не из git-репозитория)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="ИИ-агент статической проверки документа требований (ISO/IEC/IEEE 29148, IEEE 830).")
    p.add_argument("input", nargs="?", help="путь к документу требований (.pdf, .docx, .md, .txt)")
    p.add_argument("output", nargs="?", help="путь к отчёту (.md или .pdf); рядом создаются .md, .pdf и .json")
    p.add_argument("--input", "-i", dest="input_opt", help="то же, что позиционный аргумент input")
    p.add_argument("--output", "-o", dest="output_opt", help="то же, что позиционный аргумент output")
    p.add_argument("--model", help="модель Gemini (по умолчанию из .env или gemini-3.8-flash)")
    p.add_argument("--batch-size", type=int, default=None, help="сколько требований проверять за один запрос (по умолчанию 8)")
    p.add_argument("--mock", action="store_true", help="демо-режим без ключа и без LLM (только правила) — для проверки установки")
    p.add_argument("--from-json", dest="from_json", help="перестроить отчёт (PDF/MD) из сохранённого report.json без "
                   "обращения к модели; input — тот же документ (для приложения «Исходный текст»)")
    p.add_argument("--compare", dest="compare", help="сравнить с предыдущим прогоном (путь к его .json): "
                   "в отчёте появится блок «Что изменилось» — закрытые, новые и оставшиеся дефекты")
    a = p.parse_args(argv)
    a.input = a.input_opt or a.input
    a.output = a.output_opt or a.output or "reports/report.md"
    if not a.input:
        p.error("укажите путь к документу требований, например: python review.py docs/requirements.md reports/report.md")

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        load_dotenv()  # .env в текущей папке, если есть
    except ImportError:
        pass

    from reqreview.agent import run
    from reqreview.ingest import InputError
    from reqreview.llm import GeminiLLM, LLMError, MockLLM, OpenAICompatLLM
    from reqreview.loader import load_document
    from reqreview.report import write, write_diagnostic

    try:
        shown = os.path.relpath(Path(a.input).resolve(), Path.cwd())
    except ValueError:
        shown = Path(a.input).name
    version = agent_version()

    def report_files(files: dict, ok_word: str) -> None:
        print(f"\n{ok_word}", file=sys.stderr)
        if files.get("pdf"):
            print(f"  Отчёт (PDF): {Path(files['pdf']).resolve()}", file=sys.stderr)
        else:
            print(f"  ⚠ PDF не создан: {files.get('pdf_error')}", file=sys.stderr)
        print(f"  Отчёт (Markdown): {files['md'].resolve()}", file=sys.stderr)
        print(f"  Данные (JSON): {files['json'].resolve()}", file=sys.stderr)

    if a.from_json:
        import json as _json
        from reqreview.report import write_data
        data = _json.loads(Path(a.from_json).read_text(encoding="utf-8"))
        doc = load_document(a.input)
        if (data.get("document") or {}).get("sha256") not in (None, doc.sha256):
            print("ОШИБКА: документ отличается от того, по которому создан JSON (SHA-256 не совпадает).", file=sys.stderr)
            return 2
        data["render"] = {"rerendered_by": version}
        if a.compare:
            from reqreview.compare import compare
            data["comparison"] = compare(_json.loads(Path(a.compare).read_text(encoding="utf-8")), data)
        files = write_data(data, doc, a.output, version)
        report_files(files, f"Отчёт перестроен из {a.from_json} (модель не вызывалась):")
        return 0 if files.get("pdf") else 4

    try:
        doc = load_document(a.input)
    except InputError as e:
        print(f"ОШИБКА: документ нельзя проверить — {e}", file=sys.stderr)
        files = write_diagnostic(e.ingestion, "failed", [f"{e}."], a.output, version, shown)
        report_files(files, "Создан диагностический отчёт (проверка по существу не выполнялась):")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"ОШИБКА чтения документа: {e}", file=sys.stderr)
        return 2
    ing = doc.ingest
    print(f"Документ: «{doc.title}» ({doc.fmt}, {ing.size_human()}, {len(doc.nonempty())} непустых строк; "
          f"приём: {ing.status}, OCR: {ing.ocr_state})", file=sys.stderr)
    for i in ing.issues:
        print(f"  {i.id}: {i.location} — {i.observation}", file=sys.stderr)

    mock = a.mock or os.getenv("LLM_PROVIDER", "").lower() == "mock"
    llm = None
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
                fallbacks=[m for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()] or None,
            )
        print(f"Модель: {llm.name}", file=sys.stderr)
        batch = a.batch_size or int(os.getenv("BATCH_SIZE", "8"))
        result = run(doc, llm, batch_size=max(1, batch))
    except LLMError as e:
        print(f"\nОШИБКА модели: {e}", file=sys.stderr)
        files = write_diagnostic(ing, "failed", [f"Языковая модель недоступна или вернула ошибку: {e}"], a.output,
                                 version, shown, llm.name if llm else "—", doc)
        report_files(files, "Создан диагностический отчёт (документ прочитан, проверка моделью не выполнена):")
        return 3
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        return 130

    prev = None
    if a.compare:
        import json as _json
        try:
            prev = _json.loads(Path(a.compare).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"  ⚠ сравнение пропущено: не удалось прочитать {a.compare}: {e}", file=sys.stderr)
    files = write(result, a.output, version, shown, compare_with=prev)
    s = result.stats
    from reqreview.coverage import STATUS_RU
    report_files(files, f"Готово! Статус проверки: {STATUS_RU[result.status]}"
                 + (f" ({' '.join(result.status_reasons)})" if result.status_reasons else ""))
    print(f"  Требований: {s['requirements']}, дефектов: {s['findings']} "
          f"(критичных: {s['by_severity']['critical']})", file=sys.stderr)
    if not files.get("pdf"):
        return 4
    return {"completed": 0, "partial": 10, "na": 11, "failed": 12}.get(result.status, 0)


if __name__ == "__main__":
    sys.exit(main())
