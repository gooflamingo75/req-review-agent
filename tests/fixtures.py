"""Генерация тестовых входных файлов (фикстуры, не реальные документы заказчика)."""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf

BODY = ("<h2>2. Функциональные требования</h2>"
        "<p><b>Ф-1.</b> Система должна отправлять push-уведомление при превышении лимита 50000 тенге.</p>"
        "<p><b>Ф-2.</b> Уведомление отправляется не чаще одного раза в сутки.</p>")


def _story_pdf(html: str) -> bytes:
    buf = io.BytesIO()
    w = pymupdf.DocumentWriter(buf)
    story = pymupdf.Story(html=html, user_css="body{font-size:14px}")
    rect = pymupdf.paper_rect("a4")
    more = 1
    while more:
        dev = w.begin_page(rect)
        more, _ = story.place(rect + (50, 50, -50, -50))
        story.draw(dev)
        w.end_page()
    w.close()
    return buf.getvalue()


def _image_of(html: str, dpi: int = 200) -> bytes:
    with pymupdf.open(stream=_story_pdf(html), filetype="pdf") as d:
        return d[0].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY).tobytes("png")


def native_pdf(path: Path) -> Path:
    path.write_bytes(_story_pdf("<h1>Тестовое ТЗ</h1>" + BODY))
    return path


def scanned_pdf(path: Path) -> Path:
    """Только изображение страницы, без текстового слоя (скан)."""
    img = _image_of("<h1>Тестовое ТЗ</h1>" + BODY)
    d = pymupdf.open()
    p = d.new_page(width=595, height=842)
    p.insert_image(p.rect, stream=img)
    d.save(str(path))
    return path


def mixed_pdf(path: Path) -> Path:
    """Страница 1: текстовый заголовок + отсканированное тело; страница 2: пустая."""
    d = pymupdf.open(stream=_story_pdf("<h1>Тестовое ТЗ (смешанное)</h1>"), filetype="pdf")
    img = _image_of(BODY)
    p = d[0]
    p.insert_image(pymupdf.Rect(40, 140, 555, 800), stream=img)
    d.new_page(width=595, height=842)          # пустая страница
    d.save(str(path))
    return path


def truncated_pdf(path: Path) -> Path:
    data = _story_pdf("<h1>Тестовое ТЗ</h1>" + BODY)
    path.write_bytes(data[: len(data) // 3])
    return path


def encrypted_pdf(path: Path) -> Path:
    d = pymupdf.open(stream=_story_pdf("<h1>Тестовое ТЗ</h1>" + BODY), filetype="pdf")
    d.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    return path


def fake_pdf(path: Path) -> Path:
    """PNG, переименованный в .pdf."""
    path.write_bytes(_image_of("<p>картинка</p>", 50))
    return path
