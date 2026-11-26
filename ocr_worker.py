#!/usr/bin/env python3
"""OCR-воркер: распознаёт текст страницы PDF или изображения (Tesseract) и печатает JSON.

Запускается агентом в ОТДЕЛЬНОМ процессе с ограничением времени, поэтому зависание или падение
распознавания не роняет проверку остального документа.

    python ocr_worker.py --check                       # проверка установки OCR (движок, языки, тестовая картинка)
    python ocr_worker.py --pdf file.pdf --page 3       # распознать страницу 3
    python ocr_worker.py --image picture.png           # распознать изображение

Нужны: Tesseract (программа) с языками rus и eng + пакеты из requirements-ocr.txt (pytesseract, Pillow).
Путь к программе можно задать переменной окружения TESSERACT_CMD.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


class OcrError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


def configure_ocr(languages: tuple[str, ...]) -> dict:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        raise OcrError("OCR_PYTHON_PACKAGES_MISSING", str(e)) from e
    import pytesseract

    candidate = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not candidate or not Path(candidate).is_file():
        raise OcrError("OCR_ENGINE_UNAVAILABLE", "программа tesseract не найдена (задайте TESSERACT_CMD)")
    exe = str(Path(candidate).resolve())
    try:
        ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15, check=True, shell=False)
        langs = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=15, check=True, shell=False)
    except (subprocess.SubprocessError, OSError) as e:
        raise OcrError("OCR_ENGINE_UNAVAILABLE", str(e)) from e
    available = {x.strip() for x in langs.stdout.splitlines()[1:] if x.strip()}
    missing = set(languages) - available
    if missing:
        raise OcrError("OCR_LANGUAGES_UNAVAILABLE", ", ".join(sorted(missing)))
    pytesseract.pytesseract.tesseract_cmd = exe
    first = (ver.stdout or ver.stderr).splitlines()
    return {"engine": exe, "version": first[0] if first else "?", "languages": list(languages)}


def recognize_image(image, languages: tuple[str, ...], timeout: float) -> dict:
    import pytesseract
    from pytesseract import Output

    try:
        data = pytesseract.image_to_data(image, lang="+".join(languages), output_type=Output.DICT, timeout=timeout)
    except RuntimeError as e:  # pytesseract сообщает о тайм-ауте через RuntimeError
        raise OcrError("OCR_TIMEOUT" if "timeout" in str(e).lower() else "OCR_RECOGNITION_FAILED", str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise OcrError("OCR_RECOGNITION_FAILED", str(e)) from e
    lines: dict[tuple, dict] = {}
    words = 0
    for i, raw in enumerate(data["text"]):
        t = (raw or "").strip()
        if not t:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        conf = float(data["conf"][i])
        ln = lines.setdefault(key, {"words": [], "confs": [], "box": [10**9, 10**9, 0, 0]})
        ln["words"].append(t)
        if conf >= 0:                       # -1 — служебное значение, не уверенность
            ln["confs"].append(conf)
        x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
        b = ln["box"]
        b[0], b[1], b[2], b[3] = min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)
        words += 1
    out = []
    for key in sorted(lines, key=lambda k: (lines[k]["box"][1], k)):
        ln = lines[key]
        out.append({"text": " ".join(ln["words"]),
                    "conf": round(sum(ln["confs"]) / len(ln["confs"]), 1) if ln["confs"] else None,
                    "box_px": ln["box"]})
    confs = [c for ln in lines.values() for c in ln["confs"]]
    return {"lines": out, "words": words, "mean_conf": round(sum(confs) / len(confs), 1) if confs else None,
            "image_size_px": list(image.size), "layout_verified": False}


def _pix_to_pil(pix):
    from PIL import Image
    return Image.open(io.BytesIO(pix.tobytes("png")))


def recognize_pdf_page(path: Path, page_no: int, languages, dpi: int, max_pixels: int, timeout: float) -> dict:
    import pymupdf

    if page_no < 1 or dpi <= 0 or max_pixels <= 0:
        raise OcrError("OCR_BAD_REQUEST", "неверные параметры страницы/рендера")
    try:
        doc = pymupdf.open(str(path))
    except Exception as e:  # noqa: BLE001
        raise OcrError("OCR_RENDER_FAILED", str(e)) from e
    with doc:
        if doc.needs_pass:
            pw = os.environ.get("PDF_PASSWORD", "")
            if not pw or not doc.authenticate(pw):
                raise OcrError("OCR_ENCRYPTED", "документ защищён паролем")
        if page_no > doc.page_count:
            raise OcrError("OCR_BAD_REQUEST", "страницы нет в документе")
        page = doc[page_no - 1]
        r = page.rect
        scale = dpi / 72.0
        if math.ceil(r.width * scale) * math.ceil(r.height * scale) > max_pixels:
            raise OcrError("OCR_RENDER_LIMIT_EXCEEDED", f"страница больше {max_pixels} пикселей при {dpi} dpi")
        try:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
        except Exception as e:  # noqa: BLE001
            raise OcrError("OCR_RENDER_FAILED", str(e)) from e
        with _pix_to_pil(pix) as img:
            res = recognize_image(img, languages, timeout)
    res.update({"source_page": page_no, "method": "ocr", "dpi": dpi})
    return res


def recognize_image_file(path: Path, languages, max_pixels: int, timeout: float) -> dict:
    from PIL import Image

    try:
        img = Image.open(str(path))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise OcrError("OCR_IMAGE_UNREADABLE", str(e)) from e
    with img:
        if getattr(img, "n_frames", 1) > 20:
            raise OcrError("OCR_RENDER_LIMIT_EXCEEDED", "слишком много кадров в изображении")
        if img.width * img.height > max_pixels:
            raise OcrError("OCR_RENDER_LIMIT_EXCEEDED", f"изображение больше {max_pixels} пикселей")
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        res = recognize_image(img, languages, timeout)
    res.update({"method": "ocr", "frame": 0})
    return res


def health_check(languages) -> dict:
    """Движок + языки + распознавание контрольной картинки, сгенерированной самим проектом."""
    info = configure_ocr(languages)
    import pymupdf

    html = "<p style='font-size:28px'>Проверка распознавания 12345</p>"
    story = pymupdf.Story(html=html)
    buf = io.BytesIO()
    w = pymupdf.DocumentWriter(buf)
    rect = pymupdf.Rect(0, 0, 520, 90)
    dev = w.begin_page(rect)
    story.place(rect + (10, 10, -10, -10))
    story.draw(dev)
    w.end_page()
    w.close()
    with pymupdf.open(stream=buf.getvalue(), filetype="pdf") as d:
        pix = d[0].get_pixmap(dpi=300, colorspace=pymupdf.csGRAY)
    with _pix_to_pil(pix) as img:
        res = recognize_image(img, languages, 60)
    text = " ".join(ln["text"] for ln in res["lines"]).lower()
    if "12345" not in text or "провер" not in text:
        raise OcrError("OCR_FIXTURE_FAILED", f"контрольный текст распознан неверно: «{text}»")
    info["fixture"] = "ok"
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OCR-воркер агента проверки требований")
    p.add_argument("--check", action="store_true")
    p.add_argument("--pdf")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--image")
    p.add_argument("--languages", default=os.environ.get("OCR_LANGUAGES", "rus+eng"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--max-pixels", type=int, default=25_000_000)
    p.add_argument("--timeout", type=float, default=60)
    a = p.parse_args(argv)
    langs = tuple(x for x in a.languages.split("+") if x)
    try:
        if a.check:
            out = health_check(langs)
        else:
            info = configure_ocr(langs)
            if a.pdf:
                out = recognize_pdf_page(Path(a.pdf), a.page, langs, a.dpi, a.max_pixels, a.timeout)
            elif a.image:
                out = recognize_image_file(Path(a.image), langs, a.max_pixels, a.timeout)
            else:
                raise OcrError("OCR_BAD_REQUEST", "укажите --check, --pdf или --image")
            out.update({"engine_version": info["version"], "languages": info["languages"]})
        print(json.dumps({"ok": True, **out}, ensure_ascii=False))
        return 0
    except OcrError as e:
        print(json.dumps({"ok": False, "code": e.code, "detail": e.detail[:500]}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
