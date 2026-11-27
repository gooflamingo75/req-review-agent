"""Приём документа: снимок файла, проверка формата и читаемости, контроль качества извлечения текста,
восстановление через OCR, журнал страниц и инцидентов целостности.

Три понятия держатся раздельно:
  * целостность входных данных (что прочитано, повреждено, отсутствует или под сомнением) — здесь;
  * покрытие проверки (какой извлечённый текст реально проверен моделью) — agent.coverage;
  * качество спецификации (дефекты требований) — agent / report.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


SUPPORTED_EXT = {".pdf": "PDF", ".docx": "DOCX", ".md": "Markdown", ".markdown": "Markdown",
                 ".txt": "TXT", ".text": "TXT"}

STATUS_RU = {
    "ok": "прочитан",
    "partial": "прочитан частично",
    "empty_file": "пустой файл (0 байт)",
    "empty_content": "файл прочитан, но текста нет",
    "unreadable": "не читается",
    "encrypted": "защищён паролем",
    "unsupported": "формат не поддерживается",
    "too_large": "превышен лимит размера",
    "missing": "файл не найден",
}
PAGE_RU = {"readable": "прочитана", "partial": "частично", "unreadable": "не прочитана", "blank": "пустая (проверено)",
           "uncertain": "под сомнением", "skipped": "не обработана (лимит)"}
METHOD_RU = {"native": "текстовый слой", "ocr": "OCR", "mixed": "текстовый слой + OCR", "—": "—"}


@dataclass
class Issue:
    """Инцидент целостности входных данных (не дефект спецификации, своя нумерация ING-…)."""
    category: str            # processing | source | suspected | limitation
    certainty: str           # подтверждено | подозрение | ограничение инструмента
    location: str
    observation: str
    recovery: str = "—"
    result: str = "—"
    effect: str = ""
    next_action: str = ""
    limits_review: bool = True
    id: str = ""


@dataclass
class PageRec:
    page: int
    status: str = "readable"
    method: str = "native"
    native_chars: int = 0
    ocr_chars: int = 0
    ocr_conf: float | None = None
    note: str = ""
    native_candidate: str = ""          # исходный кандидат, если выбран другой (для приложения)


@dataclass
class Ingestion:
    filename: str
    size: int = 0
    sha256: str = ""
    ext: str = ""
    detected: str = ""
    fmt: str = ""
    encoding: str = ""
    status: str = "ok"
    parser: str = ""
    pages: list[PageRec] = field(default_factory=list)
    expected_pages: int | None = None
    issues: list[Issue] = field(default_factory=list)
    ocr_state: str = "не требовался"
    ocr_engine: str = ""
    native_chars: int = 0
    ocr_chars: int = 0
    images: int = 0
    notes: list[str] = field(default_factory=list)
    limits: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def fatal(self) -> bool:
        return self.status not in ("ok", "partial")

    def add(self, issue: Issue) -> Issue:
        issue.id = f"ING-{len(self.issues) + 1:03d}"
        self.issues.append(issue)
        return issue

    def size_human(self) -> str:
        return human_size(self.size)

    def page_counts(self) -> dict:
        c = {k: 0 for k in PAGE_RU}
        for p in self.pages:
            c[p.status] = c.get(p.status, 0) + 1
        return c

    def methods(self) -> list[str]:
        ms = sorted({p.method for p in self.pages if p.method != "—"})
        return [METHOD_RU[m] for m in ms] or ([f"{self.parser}"] if self.parser else [])


def human_size(n: int) -> str:
    if n < 1024:
        return f"{n} Б"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} КиБ".replace(".", ",")
    return f"{n / 1048576:.2f} МиБ".replace(".", ",")


class InputError(Exception):
    """Документ нельзя проверить по существу; ingestion содержит диагностику."""

    def __init__(self, ingestion: Ingestion, message: str):
        super().__init__(message)
        self.ingestion = ingestion


# ------------------------------------------------------------------ снимок и формат
def capture(path: Path, max_bytes: int) -> tuple[bytes, Ingestion]:
    ing = Ingestion(filename=path.name, ext=path.suffix.lower())
    ing.limits = {"max_input_bytes": max_bytes, "max_pages": int(os.getenv("MAX_PAGES", "200"))}
    if not path.exists() or not path.is_file():
        ing.status = "missing"
        ing.add(Issue("processing", "подтверждено", str(path), "Файл не найден или это не файл.",
                      next_action="Проверьте путь к документу."))
        raise InputError(ing, f"Файл не найден: {path}")
    st1 = path.stat()
    if st1.st_size > max_bytes:
        ing.size = st1.st_size
        ing.status = "too_large"
        ing.add(Issue("limitation", "подтверждено", "весь файл",
                      f"Размер {human_size(st1.st_size)} больше лимита {human_size(max_bytes)} (MAX_INPUT_MB).",
                      next_action="Увеличьте MAX_INPUT_MB в .env или разделите документ."))
        raise InputError(ing, "Файл больше допустимого размера")
    data = path.read_bytes()
    st2 = path.stat()
    if st2.st_size != len(data) or st2.st_mtime_ns != st1.st_mtime_ns:
        # файл меняется во время чтения (ещё копируется/сохраняется) — повторяем один раз
        time.sleep(1.0)
        data = path.read_bytes()
        if path.stat().st_size != len(data):
            ing.add(Issue("processing", "подозрение", "весь файл",
                          "Файл изменялся во время чтения; проверен снимок, прочитанный последним.",
                          effect="Отчёт может не соответствовать финальной версии файла.",
                          next_action="Дождитесь окончания сохранения файла и запустите проверку снова."))
    ing.size = len(data)
    ing.sha256 = hashlib.sha256(data).hexdigest()
    ing.detected = detect_format(data, ing.ext)
    return data, ing


def detect_format(data: bytes, ext: str) -> str:
    if not data:
        return "пустой"
    if data[:5] == b"%PDF-" or b"%PDF-" in data[:1024]:
        return "PDF"
    if data[:4] == b"PK\x03\x04":
        return "DOCX" if b"word/" in data[:200000] or ext == ".docx" else "ZIP"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "DOC/OLE (старый формат Word)"
    for sig, name in ((b"\x89PNG", "PNG"), (b"\xff\xd8\xff", "JPEG"), (b"GIF8", "GIF"), (b"II*\x00", "TIFF"),
                      (b"MM\x00*", "TIFF")):
        if data.startswith(sig):
            return name
    if b"\x00" in data[:4096] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "двоичный"
    return SUPPORTED_EXT.get(ext, "текст") if ext in (".md", ".markdown", ".txt", ".text") else "текст"


def decode_text(data: bytes, ing: Ingestion) -> str:
    """Строгое декодирование: UTF-8 (BOM удаляется) → UTF-16 по BOM → CP1251. Без молчаливой потери байтов."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        ing.encoding = "UTF-16"
        return data.decode("utf-16")
    try:
        text = data.decode("utf-8-sig")
        ing.encoding = "UTF-8" + (" (с BOM, BOM не входит в текст)" if data.startswith(b"\xef\xbb\xbf") else "")
        return text
    except UnicodeDecodeError as e:
        first_err = e
    try:
        text = data.decode("cp1251")
        ing.encoding = "Windows-1251"
        ing.add(Issue("processing", "подозрение", "весь файл",
                      f"Файл не в UTF-8 (первая ошибка на байте {first_err.start}); прочитан как Windows-1251.",
                      recovery="Строгое декодирование Windows-1251", result="успешно",
                      effect="Если кодировка определена неверно, часть букв может быть искажена.",
                      next_action="Сохраните документ в UTF-8.", limits_review=False))
        return text
    except UnicodeDecodeError:
        pass
    text = data.decode("utf-8", errors="replace")
    bad = text.count("�")
    ing.encoding = "UTF-8 (с ошибками)"
    ing.add(Issue("source", "подтверждено", "весь файл",
                  f"{bad} последовательностей байтов не декодируются ни в UTF-8, ни в Windows-1251; "
                  "они показаны символом «�».",
                  recovery="UTF-8 с заменой нечитаемых байтов", result="частично",
                  effect="Требования с символами «�» оцениваются с оговоркой.",
                  next_action="Пересохраните документ в UTF-8 и запустите проверку снова."))
    return text


# ------------------------------------------------------------------ сигналы качества
_PUA = re.compile(r"[-]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def garble_score(text: str) -> float:
    """Доля «подозрительных» символов: замены, управляющие, частная область Unicode, необычные категории."""
    t = re.sub(r"\s", "", text)
    if not t:
        return 0.0
    bad = t.count("�") + len(_PUA.findall(t)) + len(_CTRL.findall(t))
    odd = sum(1 for ch in t if unicodedata.category(ch) in ("Co", "Cn", "Cs"))
    return (bad + odd) / len(t)


def _nonblank_fraction(page) -> float:
    import pymupdf
    pix = page.get_pixmap(dpi=24, colorspace=pymupdf.csGRAY, alpha=False)
    s = pix.samples
    if not s:
        return 0.0
    dark = sum(1 for b in s if b < 200)
    return dark / len(s)


def _image_cover(page) -> float:
    area = abs(page.rect) or 1.0
    tot = 0.0
    try:
        for info in page.get_image_info():
            bb = info.get("bbox")
            if bb:
                import pymupdf
                r = pymupdf.Rect(bb) & page.rect
                tot += abs(r)
    except Exception:  # noqa: BLE001
        return 0.0
    return min(tot / area, 1.0)
