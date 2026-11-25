"""Чтение документа требований: PDF, DOCX, Markdown, TXT.

Документ превращается в список строк с номерами (и страницами для PDF),
чтобы каждое замечание в отчёте можно было привязать к конкретному месту.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from .util import collapse_ws, norm_chars

SUPPORTED = {".pdf", ".docx", ".md", ".markdown", ".txt", ".text"}

_NUM_HEADING = re.compile(r"^\s*(\d+(\.\d+)*)\.?\s+[A-ZА-ЯЁ]")


@dataclass
class Line:
    gid: int            # сквозной номер строки (внутренний)
    no: int             # номер строки для отчёта (в файле или на странице PDF)
    page: int | None    # страница PDF (для остальных форматов None)
    text: str
    is_heading: bool = False


@dataclass
class Document:
    path: Path
    fmt: str
    lines: list[Line]
    sha256: str
    title: str
    _display: str = field(default="", repr=False)
    _norm: str = field(default="", repr=False)
    _starts: list[int] = field(default_factory=list, repr=False)
    _owners: list[int] = field(default_factory=list, repr=False)
    _section: dict[int, str] = field(default_factory=dict, repr=False)

    # ---------- построение индексов ----------
    def build(self) -> "Document":
        buf, starts, owners = [], [], []
        pos = 0
        section = ""
        for ln in self.lines:
            if ln.is_heading:
                section = collapse_ws(ln.text.lstrip("#").strip())[:90]
            self._section[ln.gid] = section
            t = collapse_ws(ln.text)
            if not t:
                continue
            starts.append(pos)
            owners.append(ln.gid)
            buf.append(t + " ")
            pos += len(t) + 1
        self._display = "".join(buf)
        self._norm = norm_chars(self._display)
        self._starts, self._owners = starts, owners
        return self

    @property
    def by_gid(self) -> dict[int, Line]:
        return {ln.gid: ln for ln in self.lines}

    def nonempty(self) -> list[Line]:
        return [ln for ln in self.lines if ln.text.strip()]

    def char_count(self) -> int:
        return len(self._display)

    # ---------- представление для модели ----------
    def numbered(self, lines: list[Line] | None = None) -> str:
        """Текст с метками [L<gid>] для каждой непустой строки."""
        out, cur_page = [], None
        for ln in (lines if lines is not None else self.lines):
            if not ln.text.strip():
                continue
            if ln.page is not None and ln.page != cur_page:
                cur_page = ln.page
                out.append(f"===== страница {ln.page} =====")
            out.append(f"[L{ln.gid}] {collapse_ws(ln.text)}")
        return "\n".join(out)

    def text_of(self, g1: int, g2: int) -> str:
        return " ".join(collapse_ws(ln.text) for ln in self.lines
                        if g1 <= ln.gid <= g2 and ln.text.strip())

    # ---------- поиск цитат ----------
    def _gid_at(self, pos: int) -> int:
        i = bisect.bisect_right(self._starts, pos) - 1
        return self._owners[max(i, 0)]

    def find(self, quote: str, near: tuple[int, int] | None = None):
        """Ищет цитату в документе. Возвращает (дословный_текст, gid_начала, gid_конца) или None.

        Сначала точное совпадение (без учёта регистра/пробелов/вида кавычек),
        затем — нечёткое в окрестности near (если LLM слегка исказила цитату).
        """
        q = norm_chars(collapse_ws(quote)).strip(" .…\"'")
        if len(q) < 3 or not self._norm:
            return None
        # цитата с пропуском "..." — проверяем самый длинный фрагмент
        parts = [p.strip(" .…\"'") for p in re.split(r"\.\.\.|…", q)]
        parts = [p for p in parts if len(p) >= 3]
        if not parts:
            return None
        q = max(parts, key=len)
        candidates = []
        start = 0
        while True:
            p = self._norm.find(q, start)
            if p < 0:
                break
            candidates.append(p)
            start = p + 1
        if candidates:
            if near:
                candidates.sort(key=lambda p: abs(self._gid_at(p) - near[0]))
            p = candidates[0]
            return (self._display[p:p + len(q)].strip(),
                    self._gid_at(p), self._gid_at(p + len(q) - 1))
        if near:
            return self._fuzzy(q, near)
        return None

    def _fuzzy(self, q: str, near: tuple[int, int]):
        from difflib import SequenceMatcher
        g1, g2 = near[0] - 2, near[1] + 2
        idx = [i for i, g in enumerate(self._owners) if g1 <= g <= g2]
        if not idx:
            return None
        a = self._starts[idx[0]]
        b = self._starts[idx[-1]] + 10**9 if idx[-1] == len(self._starts) - 1 else self._starts[idx[-1] + 1]
        region = self._norm[a:b]
        sm = SequenceMatcher(None, region, q, autojunk=False)
        blocks = [bl for bl in sm.get_matching_blocks() if bl.size >= 3]
        if not blocks:
            return None
        covered = sum(bl.size for bl in blocks)
        if covered < 0.85 * len(q):
            return None
        s = a + blocks[0].a
        e = a + blocks[-1].a + blocks[-1].size
        if e - s > 1.5 * len(q) + 20:
            return None
        return self._display[s:e].strip(), self._gid_at(s), self._gid_at(e - 1)

    # ---------- местоположение ----------
    def location(self, g1: int, g2: int | None = None) -> str:
        lines = self.by_gid
        l1 = lines.get(g1)
        l2 = lines.get(g2 if g2 is not None else g1, l1)
        if l1 is None:
            return "место не определено"
        if l1.page is not None:
            loc = f"с. {l1.page}, строка {l1.no}"
            if l2 and (l2.page, l2.no) != (l1.page, l1.no):
                loc += f"–{'с. ' + str(l2.page) + ', строка ' if l2.page != l1.page else ''}{l2.no}"
        else:
            loc = f"строка {l1.no}" + (f"–{l2.no}" if l2 and l2.no != l1.no else "")
        sec = self._section.get(g1)
        if sec:
            loc += f", раздел «{sec}»"
        return loc

    def section_of(self, gid: int) -> str:
        return self._section.get(gid, "")

    def headings_outline(self) -> str:
        return "\n".join(f"[L{ln.gid}] {collapse_ws(ln.text)}" for ln in self.lines if ln.is_heading)


# ======================= чтение файлов =======================

def _is_heading_text(t: str) -> bool:
    s = t.strip()
    if s.startswith("#"):
        return True
    return bool(_NUM_HEADING.match(s)) and len(s) < 90 and not s.endswith((".", ";", ","))


def _from_plain(text: str) -> list[Line]:
    lines = []
    in_code = False
    for i, raw in enumerate(text.splitlines(), start=1):
        if raw.strip().startswith("```"):
            in_code = not in_code
        t = raw.rstrip()
        lines.append(Line(gid=i, no=i, page=None, text=t,
                          is_heading=(not in_code) and _is_heading_text(t)))
    return lines


def _read_pdf(path: Path) -> tuple[list[Line], str]:
    import pymupdf  # PyMuPDF

    lines, gid = [], 0
    with pymupdf.open(path) as doc:
        title = (doc.metadata or {}).get("title") or ""
        for pno, page in enumerate(doc, start=1):
            text = page.get_text("text")
            for i, raw in enumerate(text.splitlines(), start=1):
                gid += 1
                t = raw.rstrip()
                lines.append(Line(gid=gid, no=i, page=pno, text=t, is_heading=_is_heading_text(t)))
    if not any(ln.text.strip() for ln in lines):
        raise ValueError("В PDF не найден текстовый слой (возможно, это скан). "
                         "Сохраните документ как PDF с текстом или в формате DOCX/MD.")
    return lines, title


def _read_docx(path: Path) -> tuple[list[Line], str]:
    import docx  # python-docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    title = d.core_properties.title or ""
    lines: list[Line] = []

    def add(text: str, heading: bool = False):
        for part in (text.splitlines() or [""]):
            n = len(lines) + 1
            lines.append(Line(gid=n, no=n, page=None, text=part, is_heading=heading))

    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            style = (p.style.name if p.style is not None else "") or ""
            heading = style.lower().startswith(("heading", "заголовок", "title")) or _is_heading_text(p.text)
            add(p.text, heading)
        elif tag == "tbl":
            t = Table(child, d)
            for row in t.rows:
                cells, prev = [], None
                for c in row.cells:
                    if c._tc is prev:
                        continue
                    prev = c._tc
                    cells.append(collapse_ws(c.text))
                add("| " + " | ".join(cells) + " |")
    return lines, title


def load_document(path: str | Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Формат {ext or '(без расширения)'} не поддерживается. "
                         f"Поддерживаются: {', '.join(sorted(SUPPORTED))}")
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if ext == ".pdf":
        lines, meta_title = _read_pdf(path)
        fmt = "PDF"
    elif ext == ".docx":
        lines, meta_title = _read_docx(path)
        fmt = "DOCX"
    else:
        text = data.decode("utf-8-sig", errors="replace")
        lines, meta_title = _from_plain(text), ""
        fmt = "Markdown" if ext in {".md", ".markdown"} else "TXT"

    title = meta_title.strip()
    if not title:
        for ln in lines:
            if ln.text.strip():
                title = ln.text.strip().lstrip("#").strip()
                break
    doc = Document(path=path, fmt=fmt, lines=lines, sha256=sha, title=title[:150] or path.name)
    return doc.build()
