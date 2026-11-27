"""PDF-версия отчёта: тот же Markdown-отчёт (одна модель данных) → HTML → PDF через PyMuPDF Story.

* шрифты с кириллицей и спецсимволами (₸, ≤, →) встроены MuPDF (Noto/Charis), текст выделяется и ищется;
* внутренние ссылки «место → строка исходного текста», закладки по разделам, номера страниц;
* приложение «Исходный текст» строится программой из канонического транскрипта, каждая строка — якорь L0001…;
* текст документа экранируется: разметка и инструкции внутри документа не исполняются.
"""
from __future__ import annotations

import html
import io
import re
from pathlib import Path

CSS = """
body { font-family: sans-serif; font-size: 9pt; line-height: 1.35; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 6pt 0; color: #0b3a5b; }
h2 { font-size: 13pt; margin: 14pt 0 5pt 0; color: #0b3a5b; border-bottom: 1px solid #9bb3c7; }
h3 { font-size: 11pt; margin: 10pt 0 4pt 0; color: #123; }
h4 { font-size: 10pt; margin: 8pt 0 3pt 0; }
p { margin: 3pt 0; }
ul, ol { margin: 2pt 0 2pt 0; padding-left: 14pt; }
li { margin: 1pt 0; }
table { width: 100%; margin: 4pt 0 6pt 0; }
th { background-color: #e6eef5; font-weight: bold; text-align: left; }
th, td { border: 0.5px solid #9aa7b3; padding: 2pt 3pt; font-size: 8pt; vertical-align: top; }
code { font-family: monospace; font-size: 8pt; background-color: #f1f3f5; }
pre { font-family: monospace; font-size: 7.5pt; background-color: #f6f8fa; padding: 4pt; white-space: pre-wrap; }
blockquote { margin: 4pt 0; padding: 2pt 6pt; border-left: 3px solid #e39b2d; }
a { color: #0b5cad; text-decoration: none; }
i { font-style: normal; color: #2f3b47; }
hr { border: 0; border-top: 1px solid #c8d1da; margin: 8pt 0; }
.src { font-family: monospace; font-size: 7.5pt; margin: 0; white-space: pre-wrap; }
.ln { color: #6b7785; }
.ocr { background-color: #fff7e6; }
.pg { font-family: sans-serif; font-size: 8pt; font-weight: bold; color: #0b3a5b; margin: 6pt 0 2pt 0;
      border-bottom: 0.5px solid #c8d1da; }
.gap { font-family: sans-serif; font-size: 8pt; color: #8a1c1c; background-color: #fdecec; padding: 2pt; margin: 2pt 0; }
"""

_TABLE_SEP = re.compile(r"^\|\s*:?-{2,}")
_LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(text: str, used: dict) -> str:
    base = re.sub(r"[^\w]+", "-", text.lower()).strip("-")[:40] or "s"
    n = used.get(base, 0)
    used[base] = n + 1
    return f"h-{base}" + (f"-{n}" if n else "")


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    codes: list[str] = []

    def keep(m):
        codes.append(f"<code>{m.group(1)}</code>")
        return f"\x00{len(codes) - 1}\x00"
    t = re.sub(r"`([^`]+)`", keep, t)
    t = re.sub(r"\[([^\]]+)\]\((#[\w-]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", t)
    t = t.replace("\\|", "|")
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], t)


def _cells(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [c.strip() for c in re.split(r"(?<!\\)\|", row)]


def md_to_html(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    used: dict = {}
    i = 0
    para: list[str] = []
    stack: list[tuple[int, str]] = []        # открытые списки: (отступ, тег)

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    def close_lists(level: int = -1):
        while stack and stack[-1][0] > level:
            out.append("</li></" + stack.pop()[1] + ">")

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            flush_para()
            close_lists()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append("<pre>" + html.escape("\n".join(buf), quote=False) + "</pre>")
            i = j + 1
            continue
        if ln.startswith("|"):
            flush_para()
            close_lists()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            head = _cells(rows[0])
            body = [r for r in rows[1:] if not _TABLE_SEP.match(r)]
            t = ["<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"]
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in _cells(r)) + "</tr>")
            out.append("".join(t) + "</table>")
            continue
        m = _HEAD.match(ln)
        if m:
            flush_para()
            close_lists()
            lvl, text = len(m.group(1)), m.group(2).strip()
            out.append(f'<h{lvl} id="{_slug(text, used)}">{inline(text)}</h{lvl}>')
            i += 1
            continue
        if ln.strip() == "---":
            flush_para()
            close_lists()
            out.append("<hr/>")
            i += 1
            continue
        if ln.startswith(">"):
            flush_para()
            close_lists()
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i][1:].strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        m = _LIST.match(ln)
        if m:
            flush_para()
            ind, mark, text = len(m.group(1)), m.group(2), m.group(3)
            tag = "ol" if mark[0].isdigit() else "ul"
            if stack and stack[-1][0] == ind:
                out.append("</li>")
            elif not stack or ind > stack[-1][0]:
                out.append(f"<{tag}>")
                stack.append((ind, tag))
            else:
                close_lists(ind)
                if stack and stack[-1][0] == ind:
                    out.append("</li>")
                else:
                    out.append(f"<{tag}>")
                    stack.append((ind, tag))
            out.append("<li>" + inline(text))
            i += 1
            continue
        if not ln.strip():
            flush_para()
            if not (i + 1 < len(lines) and _LIST.match(lines[i + 1])):
                close_lists()
            i += 1
            continue
        if stack:                                      # продолжение пункта списка
            out.append(" " + inline(ln.strip()))
        else:
            para.append(ln.strip())
        i += 1
    flush_para()
    close_lists()
    return "\n".join(out)


def source_html(doc, title: str) -> str:
    ing = doc.ingest
    pages = {p.page: p for p in ing.pages} if ing else {}
    out = [f'<h2 id="source">{html.escape(title)}</h2>']
    partial = ing is not None and any(p.status in ("unreadable", "partial", "uncertain", "skipped") for p in ing.pages)
    note = ("Номера L…. — строки канонического транскрипта (не номера строк исходного PDF/DOCX). "
            if doc.fmt not in ("Markdown", "TXT") else "Номер L0001 соответствует строке 1 исходного файла. ")
    out.append("<p>" + ("<b>⚠️ Транскрипт неполный:</b> часть содержимого не прочитана (см. раздел B панели). "
                        if partial else "") + note + "Строки на жёлтом фоне распознаны OCR. Текст документа "
               "приведён как данные; перенос длинных строк не меняет нумерацию.</p>")
    from .ingest import METHOD_RU, PAGE_RU
    cur = None
    by_page_done = set()

    def page_header(pn):
        p = pages.get(pn)
        extra = f" · {METHOD_RU.get(p.method, p.method)} · {PAGE_RU.get(p.status, p.status)}" if p else ""
        out.append(f'<p class="pg">Исходная страница {pn}{html.escape(extra)}</p>')
        if p and p.status in ("unreadable", "partial", "uncertain"):
            out.append(f'<p class="gap">[Диагностика, не текст документа] Содержимое страницы {pn} прочитано не '
                       f'полностью: {html.escape(PAGE_RU.get(p.status, p.status))}. См. раздел B панели.</p>')
        by_page_done.add(pn)

    for ln in doc.lines:
        if ln.page is not None and ln.page != cur:
            # страницы без строк (пустые/нечитаемые) между текущей и следующей
            for pn in range((cur or 0) + 1, ln.page):
                if pn in pages and pn not in by_page_done:
                    page_header(pn)
                    if pages[pn].status in ("blank", "skipped", "unreadable"):
                        out.append(f'<p class="gap">[Диагностика] Строк нет: '
                                   f'{html.escape(PAGE_RU.get(pages[pn].status, pages[pn].status))}.</p>')
            cur = ln.page
            page_header(cur)
        cls = "src ocr" if ln.src == "ocr" else "src"
        out.append(f'<p class="{cls}" id="L{ln.gid:04d}"><span class="ln">L{ln.gid:04d}'
                   f'{" OCR" if ln.src == "ocr" else ""} │</span> {html.escape(ln.text) or "&#160;"}</p>')
    for pn in sorted(pages):
        if pn and pn not in by_page_done and pages[pn].status in ("blank", "skipped", "unreadable"):
            page_header(pn)
            out.append(f'<p class="gap">[Диагностика] Строк нет: '
                       f'{html.escape(PAGE_RU.get(pages[pn].status, pages[pn].status))}.</p>')
    return "\n".join(out)


def write_pdf(md: str, out_path: Path, doc=None, source_title: str = "", footer: str = "") -> Path:
    body = md_to_html(md)
    if doc is not None:
        body += "\n" + source_html(doc, source_title)
    return write_pdf_html(body, out_path, CSS, footer)


PAGEBREAK = "<!--PAGEBREAK-->"


# MuPDF Story иногда ошибается при переносе таблицы на новую страницу: в зависимости от того, где именно
# пришёлся разрыв, он либо бесконечно создаёт пустые страницы, либо молча теряет оставшиеся строки таблицы
# (так пропадали тест-кейсы). Поэтому каждый фрагмент проверяется: текст в PDF должен совпасть с текстом HTML
# символ в символ; при сбое фрагмент перерисовывается с чуть другой высотой области текста — место разрыва
# сдвигается, и ошибка не повторяется.
_BOTTOMS = (46, 52, 58, 49, 55, 61, 64, 43)


def _plain_len(s: str) -> int:
    return len(re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", " ", s))))


def _render_part(part: str, css: str, page, titles: dict):
    import time

    import pymupdf
    want = _plain_len(part)
    best = None
    for bottom in _BOTTOMS:
        content = page + (42, 40, -42, -bottom)
        pos: list[tuple] = []
        tc: list[list] = []
        t0 = time.monotonic()
        limit = 60 + want // 400

        def rectfn(rect_num, filled):
            if rect_num > limit or time.monotonic() - t0 > 90:
                raise RuntimeError("runaway layout")
            return page, content, None

        def positionfn(p):
            if not (p.open_close & 1):
                return
            pos.append((p.id, p.href, p.page_num - 1, tuple(p.rect)))
            if p.heading and p.heading <= 3:
                tc.append([p.heading, titles.get(p.id) or p.text.strip(), p.page_num - 1, p.rect])
        stream = io.BytesIO()
        writer = pymupdf.DocumentWriter(stream)
        try:
            pymupdf.Story(html=f"<body>{part}</body>", user_css=css).write(writer, rectfn, positionfn=positionfn)
            writer.close()
        except Exception:  # noqa: BLE001 — сбой раскладки: пробуем другую высоту
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            continue
        data = stream.getvalue()
        with pymupdf.open("pdf", data) as d:
            got = len(re.sub(r"\s+", "", "".join(p_.get_text() for p_ in d)))
        score = got / want if want else 1.0
        if best is None or abs(1 - score) < abs(1 - best[0]):
            best = (score, data, pos, tc)
        if abs(1 - score) <= 0.002:
            break
    if best is None:
        raise RuntimeError("PDF: не удалось разложить фрагмент отчёта")
    return best[1], best[2], best[3]


def write_pdf_html(body: str, out_path: Path, css: str, footer: str = "") -> Path:
    import pymupdf

    out_path = Path(out_path)
    page = pymupdf.paper_rect("a4")
    toc: list[list] = []
    titles = {m.group(2): html.unescape(re.sub(r"<[^>]+>", "", m.group(3)))
              for m in re.finditer(r'<h(\d) id="([^"]+)">(.*?)</h\1>', body, re.S)}
    # Разрывы страниц: CSS page-break в MuPDF Story может зацикливаться, поэтому каждый фрагмент между
    # метками PAGEBREAK — отдельный Story в одном документе (всегда с новой страницы); внутренние ссылки
    # между фрагментами расставляются по собранным позициям элементов.
    parts = [p_ for p_ in body.split(PAGEBREAK) if p_.strip()]
    pdf = pymupdf.open()
    positions: list[tuple] = []           # (id, href, page_index, rect)
    for part in parts:
        data, pos, tc = _render_part(part, css, page, titles)
        offset = pdf.page_count
        with pymupdf.open("pdf", data) as sub:
            pdf.insert_pdf(sub)
        positions += [(a, b, pg + offset, r) for a, b, pg, r in pos]
        toc += [[lvl, t, pg + offset + 1, r] for lvl, t, pg, r in tc]
    targets: dict[str, tuple] = {}
    for pid, _, pg, rect in positions:
        if pid and pid not in targets:
            targets[pid] = (pg, rect)
    for _, href, pg, rect in positions:
        if not href or pg >= pdf.page_count:
            continue
        if href.startswith("#"):
            t = targets.get(href[1:])
            if t is None or t[0] >= pdf.page_count:
                continue
            pdf[pg].insert_link({"kind": pymupdf.LINK_GOTO, "from": pymupdf.Rect(rect), "page": t[0],
                                 "to": pymupdf.Point(t[1][0], t[1][1])})
        else:
            pdf[pg].insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(rect), "uri": href})
    n = pdf.page_count
    for pno in range(n):
        pg = pdf[pno]
        r = pymupdf.Rect(42, page.height - 32, page.width - 42, page.height - 16)
        pg.insert_htmlbox(r, f"<p style='font-family:sans-serif;font-size:6.8pt;color:#8a96a0;text-align:right'>"
                             f"{html.escape(footer)} · {pno + 1} / {n}</p>")
    clean, last = [], 0
    base = min((t[0] for t in toc), default=1)
    for lvl, text, pnum, rect in toc:
        lvl = min(lvl - base + 1, last + 1)
        last = lvl
        pnum = max(1, min(n, pnum if pnum >= 1 else pnum + 1))
        clean.append([lvl, text[:120], pnum, {"kind": pymupdf.LINK_GOTO, "to": pymupdf.Point(rect[0], rect[1])}])
    if clean:
        try:
            pdf.set_toc(clean)
        except Exception:  # noqa: BLE001
            pdf.set_toc([c[:3] for c in clean])
    pdf.set_metadata({"title": "Отчёт о статической проверке требований", "creator": "req-review-agent", "author": "Mog department",
                      "producer": "PyMuPDF"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(out_path), garbage=3, deflate=True)
    pdf.close()
    return out_path


def validate_pdf(path: Path, must_contain: str = "провер") -> dict:
    """Проверка результата: файл существует, открывается как PDF, текст на русском извлекается, ссылки целы."""
    import pymupdf
    info = {"ok": False, "pages": 0, "links": 0, "broken_links": 0, "toc": 0}
    with pymupdf.open(str(path)) as d:
        info["pages"] = d.page_count
        info["toc"] = len(d.get_toc())
        text = d[0].get_text()
        for p in d:
            for lk in p.get_links():
                if lk.get("kind") == pymupdf.LINK_GOTO:
                    info["links"] += 1
                    if not 0 <= lk.get("page", -1) < d.page_count:
                        info["broken_links"] += 1
    info["ok"] = info["pages"] > 0 and must_contain in text.lower() and info["broken_links"] == 0
    return info
