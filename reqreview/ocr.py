"""Вызов OCR-воркера (ocr_worker.py) в отдельном процессе с ограничением времени.

Агент не устанавливает OCR сам: он проверяет, установлен ли Tesseract с нужными языками (health check),
и вызывает воркер только когда проверка качества извлечения текста показала, что это нужно.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parent.parent / "ocr_worker.py"

ERROR_RU = {
    "OCR_PYTHON_PACKAGES_MISSING": "не установлены пакеты pytesseract/Pillow (requirements-ocr.txt)",
    "OCR_ENGINE_UNAVAILABLE": "не установлена программа Tesseract",
    "OCR_LANGUAGES_UNAVAILABLE": "у Tesseract нет нужных языков",
    "OCR_FIXTURE_FAILED": "контрольное распознавание не прошло",
    "OCR_TIMEOUT": "превышено время распознавания",
    "OCR_RENDER_FAILED": "страницу не удалось отрисовать",
    "OCR_RENDER_LIMIT_EXCEEDED": "превышен лимит размера изображения",
    "OCR_RECOGNITION_FAILED": "ошибка распознавания",
    "OCR_IMAGE_UNREADABLE": "изображение не читается",
    "OCR_ENCRYPTED": "документ защищён паролем",
    "OCR_WORKER_CRASHED": "OCR-процесс завершился с ошибкой",
    "OCR_DISABLED": "OCR отключён настройкой OCR_ENABLED=off",
}


class OcrConfig:
    def __init__(self):
        self.enabled = os.getenv("OCR_ENABLED", "auto").strip().lower() not in ("0", "off", "false", "no")
        self.languages = os.getenv("OCR_LANGUAGES", "rus+eng").strip() or "rus+eng"
        self.dpi = int(os.getenv("OCR_DPI", "300"))
        self.max_pixels = int(os.getenv("OCR_MAX_PIXELS", "25000000"))
        self.page_timeout = float(os.getenv("OCR_PAGE_TIMEOUT_SEC", "60"))
        self.retries = int(os.getenv("OCR_RETRIES_PER_PAGE", "1"))


class Ocr:
    """Один экземпляр на запуск: health check выполняется не более одного раза."""

    def __init__(self, cfg: OcrConfig | None = None):
        self.cfg = cfg or OcrConfig()
        self._health: dict | None = None
        self.calls = 0

    def health(self) -> dict:
        if self._health is None:
            if not self.cfg.enabled:
                self._health = {"ok": False, "code": "OCR_DISABLED", "detail": ""}
            else:
                self._health = self._run(["--check"], timeout=120)
        return self._health

    @property
    def available(self) -> bool:
        return bool(self.health().get("ok"))

    def describe(self) -> str:
        h = self.health()
        if h.get("ok"):
            return f"{h.get('version', 'tesseract')}, языки {'+'.join(h.get('languages', []))}"
        return ERROR_RU.get(h.get("code", ""), h.get("code", "")) + (f" ({h['detail']})" if h.get("detail") else "")

    def pdf_page(self, path: Path, page: int) -> dict:
        self.calls += 1
        args = ["--pdf", str(path), "--page", str(page), "--dpi", str(self.cfg.dpi),
                "--max-pixels", str(self.cfg.max_pixels), "--timeout", str(self.cfg.page_timeout)]
        res = self._run(args, timeout=self.cfg.page_timeout + 30)
        tries = 0
        while not res.get("ok") and res.get("code") in ("OCR_WORKER_CRASHED",) and tries < self.cfg.retries:
            tries += 1
            res = self._run(args, timeout=self.cfg.page_timeout + 30)
        return res

    def image(self, path: Path) -> dict:
        self.calls += 1
        return self._run(["--image", str(path), "--max-pixels", str(self.cfg.max_pixels),
                          "--timeout", str(self.cfg.page_timeout)], timeout=self.cfg.page_timeout + 30)

    def _run(self, args: list[str], timeout: float) -> dict:
        cmd = [sys.executable, str(WORKER), "--languages", self.cfg.languages, *args]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
                               shell=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "code": "OCR_TIMEOUT", "detail": f"процесс остановлен через {int(timeout)} с"}
        except OSError as e:
            return {"ok": False, "code": "OCR_WORKER_CRASHED", "detail": str(e)[:300]}
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "code": "OCR_WORKER_CRASHED", "detail": (p.stderr or "")[-300:]}
