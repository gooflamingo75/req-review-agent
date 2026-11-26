"""Чтение документа требований: PDF, DOCX, Markdown, TXT.

Документ превращается в список строк с номерами (и страницами для PDF),
чтобы каждое замечание в отчёте можно было привязать к конкретному месту.
"""
from __future__ import annotations

import bisect
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .util import collapse_ws, norm, norm_chars


_NUM_HEADING = re.compile(r"^\s*(\d+(\.\d+)*)\.?\s+[A-ZА-ЯЁ]")


@dataclass
class Line:
    gid: int            # сквозной номер строки (внутренний)
    no: int             # номер строки для отчёта (в файле или на странице PDF)
    page: int | None    # страница PDF (для остальных форматов None)
    text: str
    is_heading: bool = False
    src: str = "native"          # native — текстовый слой/файл; ocr — распознано OCR
    conf: float | None = None    # средняя уверенность OCR по строке (диагностика движка, не вероятность)


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
    _part: dict[int, str] = field(default_factory=dict, repr=False)
    _item: dict[int, str] = field(default_factory=dict, repr=False)
    misplaced: list = field(default_factory=list, repr=False)
    ingest: object = field(default=None, repr=False)
    _spans: dict = field(default_factory=dict, repr=False)
    _canon_len: int = 0

    # ---------- построение индексов ----------
    def build(self) -> "Document":
        buf, starts, owners = [], [], []
        pos = 0
        part = ""
        stack: list[tuple[int, str]] = []          # (уровень, заголовок) — путь по иерархии заголовков
        part_re = re.compile(r"^\s*#{0,3}\s*((Часть|Part|Документ|Приложение)\s+[\wА-Яа-я№-]+)", re.I)
        for ln in self.lines:
            if ln.is_heading:
                lvl = _heading_level(ln.text)
                title = collapse_ws(ln.text.lstrip("#").strip())[:90]
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, title))
                m = part_re.match(ln.text)
                if m:
                    part = m.group(1).rstrip(".")
            section = stack[-1][1] if stack else ""
            # Пункт с собственной меткой («**401.** …») под заголовком с ДРУГОЙ меткой («200. …») —
            # по смыслу относится к родительскому разделу; запоминаем это как структурную аномалию.
            item = _item_label(ln.text) if not ln.is_heading else ""
            if item and stack:
                head = _item_label(stack[-1][1])
                if head and head != item and _same_shape(head, item):
                    parent = next((t for lv, t in reversed(stack[:-1])), "")
                    if parent:
                        self.misplaced.append((ln.gid, item, stack[-1][1], parent))
                        section = parent
            self._section[ln.gid] = section
            self._item[ln.gid] = item
            self._part[ln.gid] = part
            t = collapse_ws(ln.text)
            if not t:
                continue
            starts.append(pos)
            owners.append(ln.gid)
            buf.append(t + " ")
            pos += len(t) + 1
        # канонический транскрипт: строки как есть, разделённые переводом строки; символ = кодовая точка Unicode
        off = 0
        for i, ln in enumerate(self.lines):
            self._spans[ln.gid] = (off, off + len(ln.text))
            off += len(ln.text) + (1 if i < len(self.lines) - 1 else 0)
        self._canon_len = off
        self._display = "".join(buf)
        self._norm = norm_chars(self._display)
        self._starts, self._owners = starts, owners
        return self

    @property
    def by_gid(self) -> dict[int, Line]:
        if len(getattr(self, "_by", {}) or {}) != len(self.lines):
            self._by = {ln.gid: ln for ln in self.lines}
        return self._by

    def nonempty(self) -> list[Line]:
        return [ln for ln in self.lines if ln.text.strip()]

    def char_count(self) -> int:
        return len(self._display)

    @property
    def canon_len(self) -> int:
        """Число символов канонического транскрипта (с пробелами и переводами строк)."""
        return self._canon_len

    def span(self, g1: int, g2: int | None = None) -> tuple[int, int]:
        """Полуоткрытый интервал [start, end) строк g1..g2 в каноническом транскрипте (включая переводы строк между ними)."""
        a = self._spans.get(g1)
        b = self._spans.get(g2 if g2 is not None else g1, a)
        if a is None:
            return (0, 0)
        return (a[0], max(a[1], b[1]))

    def lines_span(self, lines) -> list[tuple[int, int]]:
        return [self.span(ln.gid) for ln in lines]

    def line_label(self, gid: int) -> str:
        return f"L{gid:04d}"

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
            out.append(f"[L{ln.gid}]{' (OCR)' if ln.src == 'ocr' else ''} {collapse_ws(ln.text)}")
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
    def location(self, g1: int, g2: int | None = None, link: bool = False) -> str:
        """Человекочитаемое место: «строка 12–14» (MD/TXT — исходные строки файла), «с. 3, L0042–L0045»
        (PDF — строки канонического транскрипта), «L0042» (DOCX). link=True — ссылка на приложение «Исходный текст»."""
        lines = self.by_gid
        l1 = lines.get(g1)
        g2 = g2 if g2 is not None else g1
        l2 = lines.get(g2, l1)
        if l1 is None:
            return "место не определено"
        if l1.page is not None:
            loc = f"с. {l1.page}, L{l1.gid:04d}"
            if l2 and l2.gid != l1.gid:
                loc += "–" + (f"с. {l2.page}, " if l2.page != l1.page else "") + f"L{l2.gid:04d}"
        elif self.fmt in ("Markdown", "TXT"):
            loc = f"строка {l1.no}" + (f"–{l2.no}" if l2 and l2.no != l1.no else "")
        else:
            loc = f"L{l1.gid:04d}" + (f"–L{l2.gid:04d}" if l2 and l2.gid != l1.gid else "")
        if link:
            loc = f"[{loc}](#L{l1.gid:04d})"
        if any(ln.src == "ocr" for g, ln in lines.items() if l1.gid <= g <= (l2.gid if l2 else l1.gid)):
            loc += " [текст распознан OCR]"
        sec = self._section.get(g1)
        if sec:
            loc += f", раздел «{sec}»"
        item = self._item.get(g1)
        if item and not (sec or "").startswith(item):
            loc += f", пункт «{item}»"
        return loc

    def section_of(self, gid: int) -> str:
        return self._section.get(gid, "")

    def part_of(self, gid: int) -> str:
        return self._part.get(gid, "")

    def headings_outline(self) -> str:
        return "\n".join(f"[L{ln.gid}] {collapse_ws(ln.text)}" for ln in self.lines if ln.is_heading)


# ======================= чтение файлов =======================

_ITEM_RE = re.compile(r"^\s*(?:[-*]\s*)?\*\*\s*([A-Za-zА-Яа-яЁё]{0,6}[-_.]?\d+(?:\.\d+)*)\.?\s*\*\*")
_HEAD_LABEL_RE = re.compile(r"^\s*([A-Za-zА-Яа-яЁё]{0,6}[-_.]?\d+(?:\.\d+)*)[.\s]")


def _item_label(t: str) -> str:
    """Метка пункта: «**401.** …» → «401», «**Ф-1.** …» → «Ф-1»; для заголовка «200. Successful» → «200»."""
    m = _ITEM_RE.match(t) or _HEAD_LABEL_RE.match(t.lstrip("#").strip() + " ")
    return m.group(1) if m else ""


def _same_shape(a: str, b: str) -> bool:
    """Метки одного вида: обе числовые одной длины (200 и 401) или с одинаковым буквенным префиксом (Ф-1 и Ф-2).
    Номер главы «2» и требование «КБ-1» — разные виды, это не аномалия структуры."""
    if a.isdigit() and b.isdigit():
        return len(a) == len(b)
    pa, pb = re.match(r"[^\d]*", a).group(0), re.match(r"[^\d]*", b).group(0)
    return bool(pa) and pa == pb


def _heading_level(t: str) -> int:
    s = t.lstrip()
    if s.startswith("#"):
        return len(s) - len(s.lstrip("#"))
    m = _NUM_HEADING.match(s)
    return (m.group(1).count(".") + 1) if m else 1


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
        lines.append(Line(gid=i, no=i, page=None, text=raw,
                          is_heading=(not in_code) and _is_heading_text(t)))
    return lines


def _dedupe_ocr(native_lines: list[str], ocr_lines: list[dict]) -> list[dict]:
    """Оставляет строки OCR, которых нет в текстовом слое (смешанная страница: текстовый заголовок + скан)."""
    from difflib import SequenceMatcher
    nat = [norm(x) for x in native_lines if norm(x)]
    joined = " ".join(nat)
    out = []
    for o in ocr_lines:
        t = norm(o.get("text", ""))
        if len(t) < 3:
            continue
        if t in joined or any(SequenceMatcher(None, t, n, autojunk=False).ratio() > 0.8 for n in nat):
            continue
        out.append(o)
    return out


def _read_pdf(data: bytes, ing, ocr, snapshot: Path) -> tuple[list[Line], str]:
    import pymupdf  # PyMuPDF

    from .ingest import InputError, Issue, PageRec, _image_cover, _nonblank_fraction, garble_score
    ing.parser = f"PyMuPDF {getattr(pymupdf, 'VersionBind', '')}".strip()
    try:
        pymupdf.TOOLS.mupdf_display_errors(False)
        pymupdf.TOOLS.mupdf_warnings(reset=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # noqa: BLE001
        ing.status = "unreadable"
        ing.add(Issue("processing", "подтверждено", "весь файл", f"PDF не открывается: {str(e)[:200]}",
                      recovery="PyMuPDF с автоматическим восстановлением структуры", result="не удалось",
                      effect="Проверка по существу невозможна.",
                      next_action="Экспортируйте документ в PDF заново или пришлите DOCX/MD."))
        raise InputError(ing, "PDF не открывается") from e
    lines: list[Line] = []
    with doc:
        if doc.needs_pass:
            pw = os.getenv("PDF_PASSWORD", "")
            if not pw or not doc.authenticate(pw):
                ing.status = "encrypted"
                ing.add(Issue("source", "подтверждено", "весь файл", "PDF защищён паролем"
                              + (" (пароль из PDF_PASSWORD не подошёл)." if pw else "."),
                              recovery="обход пароля не выполняется", result="—",
                              effect="Проверка по существу невозможна.",
                              next_action="Пришлите копию без пароля или укажите пароль в PDF_PASSWORD в .env."))
                raise InputError(ing, "PDF защищён паролем")
        if doc.is_repaired:
            ing.add(Issue("source", "подозрение", "весь файл",
                          "Структура PDF повреждена; MuPDF автоматически восстановил её при открытии.",
                          recovery="автоматическое восстановление структуры", result="файл открыт",
                          effect="Восстановление не доказывает, что сохранились все исходные страницы.",
                          next_action="Сверьте число страниц с оригиналом; при расхождении пришлите неповреждённый файл."))
        title = (doc.metadata or {}).get("title") or ""
        n = doc.page_count
        ing.expected_pages = None if doc.is_repaired else n
        maxp = ing.limits.get("max_pages", 200)
        gid = 0
        ocr_note_added = False
        for pno in range(1, n + 1):
            rec = PageRec(page=pno)
            ing.pages.append(rec)
            if pno > maxp:
                rec.status, rec.method = "skipped", "—"
                continue
            try:
                page = doc[pno - 1]
                native = page.get_text("text")
            except Exception as e:  # noqa: BLE001
                page, native = None, ""
                rec.note = f"ошибка извлечения: {str(e)[:120]}"
            nchar = len(native.strip())
            rec.native_chars = len(native)
            ing.native_chars += len(native)
            g = garble_score(native)
            reason = ""
            if page is None:
                reason = "текст страницы не извлекается (ошибка парсера)"
            elif nchar == 0:
                try:
                    nb = _nonblank_fraction(page)
                except Exception:  # noqa: BLE001
                    nb = 1.0
                if nb < 0.002:
                    rec.status, rec.method, rec.note = "blank", "—", "текста нет, рендер страницы пустой"
                    continue
                reason = "на странице нет текстового слоя, но она не пустая (скан или изображение)"
            elif g > 0.05:
                reason = f"текст извлечён с искажениями ({g:.0%} подозрительных символов)"
            elif _image_cover(page) > 0.35 and nchar < 400:
                reason = "крупное изображение при малом объёме текста (возможен скан внутри страницы)"
            native_lines = native.splitlines()
            chosen: list[tuple[str, str, float | None]] = [(t, "native", None) for t in native_lines]
            if reason:
                res = ocr.pdf_page(snapshot, pno) if ocr.available else {"ok": False, "code": ocr.health().get("code")}
                if res.get("ok"):
                    ing.ocr_state = "выполнен"
                    olines = res.get("lines") or []
                    rec.ocr_chars = sum(len(o["text"]) + 1 for o in olines)
                    ing.ocr_chars += rec.ocr_chars
                    rec.ocr_conf = res.get("mean_conf")
                    if nchar == 0 or g > 0.05 or page is None:
                        if g > 0.05:
                            rec.native_candidate = native
                        chosen = [(o["text"], "ocr", o.get("conf")) for o in olines]
                        rec.method = "ocr"
                    else:
                        extra = _dedupe_ocr(native_lines, olines)
                        chosen += [(o["text"], "ocr", o.get("conf")) for o in extra]
                        rec.method = "mixed" if extra else "native"
                        if not extra:
                            rec.note = "OCR не нашёл текста сверх текстового слоя"
                    if not olines:
                        rec.status = "uncertain"
                        ing.add(Issue("processing", "подозрение", f"с. {pno}", reason + "; OCR не распознал текст.",
                                      recovery="OCR страницы", result="текст не найден",
                                      effect="Пустой результат OCR не доказывает, что страница пустая.",
                                      next_action="Проверьте страницу вручную."))
                    elif rec.ocr_conf is not None and rec.ocr_conf < 60:
                        rec.status = "uncertain"
                        ing.add(Issue("processing", "подозрение", f"с. {pno}",
                                      f"{reason}; OCR-распознавание с низкой уверенностью ({rec.ocr_conf:.0f}).",
                                      recovery="OCR страницы", result="текст получен, качество под сомнением",
                                      effect="Числа, отрицания и знаки сравнения на странице могут быть распознаны "
                                             "неверно; критичные выводы по ним требуют сверки с оригиналом.",
                                      next_action="Сверьте страницу с оригиналом или пришлите более чёткий файл."))
                    elif rec.native_candidate:
                        ing.add(Issue("processing", "подтверждено", f"с. {pno}",
                                      f"{reason}. Выбран текст OCR; искажённый текстовый слой сохранён в приложении.",
                                      recovery="OCR страницы", result="успешно",
                                      effect="Выводы по странице основаны на распознанном тексте.",
                                      next_action="Сверьте критичные значения страницы с оригиналом.",
                                      limits_review=False))
                else:
                    code = res.get("code", "")
                    from .ocr import ERROR_RU
                    why = ERROR_RU.get(code, code) or "OCR недоступен"
                    if ing.ocr_state != "выполнен":
                        ing.ocr_state = "отключён" if code == "OCR_DISABLED" else (
                            "недоступен" if code in ("OCR_ENGINE_UNAVAILABLE", "OCR_PYTHON_PACKAGES_MISSING",
                                                     "OCR_LANGUAGES_UNAVAILABLE", "OCR_FIXTURE_FAILED") else "ошибка")
                    if nchar and g <= 0.05:
                        rec.status = "partial"
                    elif nchar:
                        rec.status = "uncertain"
                    else:
                        rec.status, rec.method = "unreadable", "—"
                        chosen = []
                    ing.add(Issue("limitation" if not res.get("detail") else "processing", "ограничение инструмента",
                                  f"с. {pno}", reason + ".", recovery="OCR страницы",
                                  result=f"не выполнен: {why}",
                                  effect=("Содержимое страницы не проверено; отсутствие требований на ней не "
                                          "доказано." if not nchar else
                                          "Часть содержимого страницы (изображение/скан) не проверена."),
                                  next_action=("Установите OCR (см. README, раздел «Сканы и OCR») и запустите "
                                               "снова, или пришлите PDF с текстовым слоем.")))
                    if not ocr_note_added and ing.ocr_state in ("недоступен", "отключён"):
                        ing.notes.append(f"OCR: {ocr.describe()}")
                        ocr_note_added = True
            for i, (t, src, conf) in enumerate(chosen, start=1):
                gid += 1
                lines.append(Line(gid=gid, no=i, page=pno, text=t.rstrip("\r"),
                                  is_heading=_is_heading_text(t), src=src, conf=conf))
        skipped = [p.page for p in ing.pages if p.status == "skipped"]
        if skipped:
            ing.add(Issue("limitation", "подтверждено", f"с. {skipped[0]}–{skipped[-1]}",
                          f"Документ длиннее лимита MAX_PAGES={maxp}; {len(skipped)} стр. не обработаны.",
                          effect="Эти страницы не проверены.", next_action="Увеличьте MAX_PAGES в .env."))
    try:
        warn = collapse_ws(pymupdf.TOOLS.mupdf_warnings())
        if warn:
            ing.notes.append("Сообщения парсера PDF: " + warn[:300])
    except Exception:  # noqa: BLE001
        pass
    return lines, title


def _docx_images(child) -> list[tuple[str, int]]:
    """Список (rId, ширина в EMU) изображений внутри элемента тела DOCX."""
    out = []
    ns_a = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    ns_r = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    ns_wp = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
    for drawing in child.iter():
        if not str(drawing.tag).endswith("}drawing"):
            continue
        cx = 0
        for ext in drawing.iter(f"{ns_wp}extent"):
            cx = int(ext.get("cx", "0") or 0)
        for blip in drawing.iter(f"{ns_a}blip"):
            rid = blip.get(f"{ns_r}embed")
            if rid:
                out.append((rid, cx))
    return out


def _read_docx(data: bytes, ing, ocr, workdir: Path) -> tuple[list[Line], str]:
    import io

    import docx  # python-docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    from .ingest import InputError, Issue
    ing.parser = "python-docx"
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        ing.status = "unreadable"
        ing.add(Issue("processing", "подтверждено", "весь файл", f"DOCX не открывается: {str(e)[:200]}",
                      result="не удалось", effect="Проверка по существу невозможна.",
                      next_action="Откройте файл в Word и сохраните заново как .docx."))
        raise InputError(ing, "DOCX не открывается") from e
    title = d.core_properties.title or ""
    lines: list[Line] = []

    def add(text: str, heading: bool = False, src: str = "native", conf=None):
        for part in (text.splitlines() or [""]):
            n = len(lines) + 1
            lines.append(Line(gid=n, no=n, page=None, text=part, is_heading=heading, src=src, conf=conf))

    big_unread, img_no = 0, 0
    for child in d.element.body.iterchildren():
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
        for rid, cx in _docx_images(child):
            img_no += 1
            ing.images += 1
            large = cx > 2_286_000           # шире ~6 см: вероятно, содержательное изображение, а не логотип
            if not large:
                continue
            res = {"ok": False, "code": ocr.health().get("code")} if not ocr.available else None
            if res is None:
                try:
                    part = d.part.related_parts[rid]
                    f = workdir / f"img{img_no}{Path(part.partname).suffix or '.png'}"
                    f.write_bytes(part.blob)
                    res = ocr.image(f)
                except Exception as e:  # noqa: BLE001
                    res = {"ok": False, "code": "OCR_IMAGE_UNREADABLE", "detail": str(e)[:200]}
            if res.get("ok") and res.get("lines"):
                ing.ocr_state = "выполнен"
                for o in res["lines"]:
                    add(o["text"], False, "ocr", o.get("conf"))
                ing.ocr_chars += sum(len(o["text"]) + 1 for o in res["lines"])
            else:
                big_unread += 1
                if not res.get("ok") and ing.ocr_state != "выполнен":
                    ing.ocr_state = "недоступен"
    if big_unread:
        from .ocr import ERROR_RU
        ing.add(Issue("limitation", "ограничение инструмента", f"{big_unread} изобр. в теле документа",
                      f"В документе {big_unread} крупных изображений (схемы, скриншоты); текст на них не прочитан.",
                      recovery="OCR изображений", result=f"не выполнен: {ocr.describe()}" if not ocr.available
                      else "текст не найден",
                      effect="Требования, заданные только на изображениях (BPMN, макеты), не проверены.",
                      next_action="Установите OCR или продублируйте ключевые правила текстом."))
    hf = []
    for sec in d.sections:
        for part in (sec.header, sec.footer):
            try:
                t = collapse_ws(" ".join(p.text for p in part.paragraphs))
            except Exception:  # noqa: BLE001
                t = ""
            if t and t not in hf:
                hf.append(t)
    if hf:
        ing.notes.append("Колонтитулы просмотрены и не содержат требований для отдельной проверки: «"
                         + "»; «".join(x[:80] for x in hf[:3]) + "».")
    ing.expected_pages = None
    return lines, title


_REF_RE = re.compile(r"(?:см\.?|согласно|в|во|по)\s+(приложени[еияю]\s+(?:№\s*)?[A-ZА-ЯЁ0-9]{1,3})\b", re.I)


def _check_references(doc: "Document", ing) -> None:
    """Ссылки на приложения, которых нет в документе — признак неполного комплекта (подозрение, не дефект)."""
    from .ingest import Issue
    heads = " ".join(norm(ln.text) for ln in doc.lines if ln.is_heading or len(ln.text) < 80)
    seen = set()
    for ln in doc.lines:
        for m in _REF_RE.finditer(ln.text):
            name = collapse_ws(m.group(1))
            key = norm(name)
            if key in seen:
                continue
            seen.add(key)
            # приложение считается присутствующим, если есть короткая строка/заголовок, начинающаяся с его имени
            present = any(norm(x.text).lstrip("# ").startswith(key) for x in doc.lines
                          if x.gid != ln.gid and len(x.text) < 120)
            if not present and key not in heads.replace(norm(ln.text), ""):
                ing.add(Issue("suspected", "подозрение", doc.location(ln.gid),
                              f"Документ ссылается на «{name}», но такого приложения в файле нет.",
                              effect="Требования из приложения не проверены; выводы о полноте ограничены.",
                              next_action="Приложите недостающее приложение или уберите ссылку.",
                              limits_review=True))


def load_document(path: str | Path, ocr=None) -> Document:
    """Читает документ с полной диагностикой. При невозможности проверки бросает ingest.InputError."""
    from .ingest import (SUPPORTED_EXT, InputError, Issue, PageRec, capture, decode_text)
    from .ocr import Ocr
    path = Path(path)
    t0 = time.time()
    ocr = ocr or Ocr()
    max_bytes = int(float(os.getenv("MAX_INPUT_MB", "50")) * 1048576)
    data, ing = capture(path, max_bytes)
    ext = ing.ext
    if not data:
        ing.status = "empty_file"
        ing.add(Issue("source", "подтверждено", "весь файл", "Файл пустой: 0 байт.",
                      effect="Проверка по существу невозможна; оценка качества не выставляется.",
                      next_action="Пришлите корректный файл документа."))
        raise InputError(ing, "Файл пустой (0 байт)")
    fmt = SUPPORTED_EXT.get(ext)
    det = ing.detected
    mismatch = (fmt == "PDF" and det != "PDF") or (fmt == "DOCX" and det != "DOCX") or \
               (fmt in ("Markdown", "TXT") and det in ("PDF", "DOCX", "ZIP", "PNG", "JPEG", "GIF", "TIFF", "двоичный",
                                                    "DOC/OLE (старый формат Word)"))
    if fmt is None or mismatch:
        ing.status = "unsupported"
        what = f"расширение «{ext or '(нет)'}», фактическое содержимое: {det}"
        hint = ("Файл .docx в формате OLE обычно означает защищённый паролем документ или старый .doc — "
                "сохраните его как обычный .docx." if det.startswith("DOC/OLE") else
                "Поддерживаются PDF, DOCX, MD, TXT; изображение нужно вставить в PDF или распознать.")
        ing.add(Issue("source", "подтверждено", "весь файл", f"Формат не поддерживается или не совпадает с "
                      f"расширением ({what}).", effect="Проверка по существу невозможна.", next_action=hint))
        raise InputError(ing, f"Формат не поддерживается ({what})")
    ing.fmt = fmt
    workdir = Path(tempfile.mkdtemp(prefix="rra-"))
    try:
        snapshot = workdir / f"snapshot{ext}"
        snapshot.write_bytes(data)          # неизменяемая рабочая копия; оригинал не трогаем
        os.chmod(snapshot, 0o400)
        if fmt == "PDF":
            lines, meta_title = _read_pdf(data, ing, ocr, snapshot)
        elif fmt == "DOCX":
            lines, meta_title = _read_docx(data, ing, ocr, workdir)
        else:
            text = decode_text(data, ing)
            ing.parser = f"строгое декодирование ({ing.encoding})"
            lines, meta_title = _from_plain(text), ""
            ing.native_chars = len(text)
    finally:
        for f in workdir.iterdir():
            try:
                os.chmod(f, 0o600)
                f.unlink()
            except OSError:
                pass
        try:
            workdir.rmdir()
        except OSError:
            pass
    if fmt in ("Markdown", "TXT"):
        ing.pages = [PageRec(page=0, status="readable", method="native", native_chars=ing.native_chars)]
    elif fmt == "DOCX":
        ing.pages = [PageRec(page=0, status="readable", method="mixed" if ing.ocr_state == "выполнен" else "native",
                             native_chars=sum(len(ln.text) for ln in lines if ln.src == "native"),
                             ocr_chars=ing.ocr_chars)]
    if not any(ln.text.strip() for ln in lines):
        unread = [p.page for p in ing.pages if p.status in ("unreadable", "uncertain", "partial")]
        if unread or (fmt == "PDF" and ing.expected_pages is None):
            ing.status = "unreadable"
            if not unread:
                ing.add(Issue("source", "подтверждено", "весь файл",
                              "После восстановления повреждённой структуры PDF текст не найден ни в текстовом "
                              "слое, ни на изображениях страниц.", recovery="восстановление структуры + рендер страниц",
                              result="содержимое не восстановлено",
                              effect="Нельзя утверждать, что документ пустой: вероятно, файл обрезан или повреждён.",
                              next_action="Пришлите неповреждённый файл (проверьте, что загрузка/копирование завершились)."))
            raise InputError(ing, "Текст документа не удалось извлечь")
        ing.status = "empty_content"
        ing.add(Issue("source", "подтверждено", "весь документ",
                      "Файл прочитан успешно, но не содержит текста" +
                      (" (все страницы проверены рендером и пустые)." if fmt == "PDF" else " (только пробелы)."),
                      effect="Проверять нечего; оценка качества не выставляется.",
                      next_action="Проверьте, тот ли файл передан."))
        raise InputError(ing, "Документ не содержит текста")

    title = meta_title.strip()
    if not title:
        for ln in lines:
            if ln.text.strip():
                title = ln.text.strip().lstrip("#").strip()
                break
    doc = Document(path=path, fmt=fmt, lines=lines, sha256=ing.sha256, title=title[:150] or path.name)
    doc.ingest = ing
    doc.build()
    _check_references(doc, ing)
    if any(i.limits_review for i in ing.issues):
        ing.status = "partial"
    ing.seconds = time.time() - t0
    if ocr._health is not None:
        ing.ocr_engine = ocr.describe()
    return doc
