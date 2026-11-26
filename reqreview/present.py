"""Представление отчёта для конечного читателя: «ответ — первым», детали — только там, где есть проблемы.

Источник — единая модель данных отчёта (словарь, который сохраняется в report.json). Из неё строятся
две версии одного и того же отчёта: PDF (основная, с оформлением) и Markdown. Поэтому отчёт можно
перестроить из сохранённого JSON без повторного обращения к модели (review.py --from-json).

Порядок: резюме для руководителя (вывод, блокеры, нужные решения, объём проверки) → 1 Паспорт → 2 Нарезка →
3 Требования с замечаниями → 4 Набор требований → 5 Статистика → 6 Вердикт и план действий → приложения.
"""
from __future__ import annotations

import html
import re
from collections import Counter

from .criteria import INDIVIDUAL, SET, CRITICAL_HINT
from .util import collapse_ws, md_cell, short

ACCENT = "#00ab75"          # BCC green
SEV = {"CRITICAL": ("Блокер", "#c0392b"), "MAJOR": ("Существенный", "#d68910"), "MINOR": ("Замечание", "#7b8a8b")}
SEV_ORDER = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
SET_STATUS = {"ok": ("выполнен", "ok"), "partial": ("частично", "warn"), "fail": ("не выполнен", "fail"),
              "n/a": ("не оценивался", "na")}
MARK = {"ok": "✓", "warn": "!", "fail": "✗", "na": "·"}
RUN = {"completed": ("Завершена", "ok"), "partial": ("Частичная", "warn"), "failed": ("Не выполнена", "fail"),
       "na": ("Нет оцениваемых требований", "na")}
THEMES = {"REST-METHOD": "контракт API (HTTP-методы)", "STATUS-200": "контракт API (коды ответов)",
          "STATUS-CODE": "контракт API (коды ответов)", "FORMAT": "контракт API (формат)",
          "UNITS": "единицы и типы данных", "DATA-TYPE": "единицы и типы данных",
          "CONTRADICTION": "противоречия бизнес-правил", "EDGE-CASE": "пограничные сценарии",
          "INCOMPLETE": "неполные требования", "UNTESTABLE": "тестируемость",
          "ACCEPTANCE": "критерии приёмки", "SECTION-EMPTY": "незаполненные разделы",
          "PLACEHOLDER": "незаполненные разделы", "APPROVALS": "оформление и согласование",
          "VERSIONING": "оформление и согласование", "STRUCTURE": "структура документа",
          "TRACEABILITY": "прослеживаемость", "AMBIGUITY": "однозначность формулировок",
          "SECURITY": "безопасность", "IDEMPOTENCY": "идемпотентность", "ERROR-HANDLING": "обработка ошибок"}
WHY = [("trigger", "Когда проявляется"), ("development", "Разработка"), ("testing", "Тестирование"),
       ("business", "Бизнес и клиент"), ("exploitation", "Злоупотребление"), ("scenario", "Сценарий"),
       ("resolution", "Нужное решение"), ("closure", "Условие закрытия")]


def plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


def num(n) -> str:
    return "—" if n is None else f"{n:,}".replace(",", " ")


# ====================================================================== блоки и два вывода
class Out:
    """Минимальная разметка, которую умеют выводить и HTML (для PDF), и Markdown."""

    def __init__(self):
        self.blocks: list[tuple] = []

    def h(self, level, text, anchor=None):
        self.blocks.append(("h", level, text, anchor))

    def p(self, text, cls=""):
        self.blocks.append(("p", text, cls))

    def note(self, text):
        self.blocks.append(("p", text, "note"))

    def callout(self, text, kind="info"):
        self.blocks.append(("callout", text, kind))

    def bullets(self, items, cls=""):
        if items:
            self.blocks.append(("ul", list(items), cls))

    def table(self, head, rows, widths=None, cls=""):
        self.blocks.append(("table", head, rows, widths, cls))

    def kv(self, rows, cls="kv"):
        self.blocks.append(("kv", rows, cls))

    def tiles(self, items):
        self.blocks.append(("tiles", items))

    def card(self, anchor, sev, title, rows, meta=""):
        self.blocks.append(("card", anchor, sev, title, rows, meta))

    def html(self, raw, md=""):
        self.blocks.append(("raw", raw, md))

    def rule(self):
        self.blocks.append(("hr",))

    def pagebreak(self):
        self.blocks.append(("pb",))


def _inline_html(t: str) -> str:
    from .pdf import inline
    s = inline(t or "")
    s = re.sub(r"\{(CRITICAL|MAJOR|MINOR|ok|warn|fail|na)\}", r'<span class="\1">', s)
    return s.replace("{/}", "</span>")


def _strip_tags(t: str) -> str:
    return re.sub(r"\{(CRITICAL|MAJOR|MINOR|ok|warn|fail|na|/)\}", "", t or "")


def _strip_md(t: str) -> str:
    return re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"\1", t or "").replace("**", "")


def to_markdown(o: Out) -> str:
    out = []
    for b in o.blocks:
        k = b[0]
        if k == "h":
            out.append("#" * b[1] + " " + b[2] + "\n")
        elif k == "p":
            out.append((f"*{b[1]}*" if b[2] == "note" else b[1]) + "\n")
        elif k == "callout":
            out.append("> " + b[1].replace("\n", "\n> ") + "\n")
        elif k == "ul":
            out.extend(f"- {x}" for x in b[1])
            out.append("")
        elif k == "table":
            head, rows = b[1], b[2]
            out.append("| " + " | ".join(head) + " |")
            out.append("|" + " :--- |" * len(head))
            for r in rows:
                out.append("| " + " | ".join(md_cell_keep(c) for c in r) + " |")
            out.append("")
        elif k == "kv":
            out.extend(f"- **{a}:** {v}" for a, v in b[1])
            out.append("")
        elif k == "tiles":
            out.append(" · ".join(f"**{v}** {lab}" + (f" ({sub})" if sub else "") for v, lab, sub, _ in b[1]) + "\n")
        elif k == "card":
            anchor, sev, title, rows, meta = b[1:]
            out.append(f"#### {_strip_tags(title)}" + (f" — {meta}" if meta else "") + "\n")
            for lab, val in rows:
                if lab == "__why__":
                    out.append("- **Почему это критично:**")
                    out.extend(f"  - *{a}:* {x}" for a, x in val)
                    continue
                if isinstance(val, list):
                    out.append(f"- **{lab}:**")
                    out.extend(f"  - {x}" for x in val)
                else:
                    out.append(f"- **{lab}:** {val}")
            out.append("")
        elif k == "raw":
            if b[2]:
                out.append(b[2])
        elif k == "hr":
            out.append("---\n")
    return "\n".join(out).rstrip() + "\n"


def md_cell_keep(s: str) -> str:
    return collapse_ws(str(s)).replace("|", "\\|")


CSS = f"""
body {{ font-family: sans-serif; font-size: 8.8pt; line-height: 1.38; color: #1d2b36; }}
h1 {{ font-size: 18pt; color: #1d2b36; margin: 0 0 2pt 0; }}
h2 {{ font-size: 13.5pt; color: {ACCENT}; margin: 16pt 0 4pt 0; padding-bottom: 2pt; border-bottom: 1.2px solid {ACCENT}; }}
h3 {{ font-size: 10.5pt; color: #1d2b36; margin: 10pt 0 3pt 0; }}
h4 {{ font-size: 9.5pt; color: #1d2b36; margin: 8pt 0 2pt 0; }}
p {{ margin: 3pt 0; }}
.note {{ color: #5b6770; font-size: 8pt; }}
.lead {{ font-size: 12.5pt; font-weight: bold; color: #1d2b36; margin: 6pt 0 8pt 0; line-height: 1.3; }}
.eyebrow {{ color: {ACCENT}; font-size: 8pt; font-weight: bold; letter-spacing: 1px; margin: 0; }}
.sub {{ color: #5b6770; font-size: 8.5pt; margin: 0 0 6pt 0; }}
ul {{ margin: 2pt 0 4pt 0; padding-left: 12pt; }}
li {{ margin: 1.5pt 0; }}
table {{ width: 100%; margin: 3pt 0 7pt 0; }}
th {{ text-align: left; font-size: 7.6pt; color: #5b6770; font-weight: bold; padding: 3pt 4pt;
      border-bottom: 1px solid #1d2b36; }}
td {{ font-size: 8pt; padding: 3pt 4pt; vertical-align: top; border-bottom: 0.5px solid #d5dde2; }}
table.kv td {{ border-bottom: 0.5px solid #e3e8eb; }}
table.kv td.k {{ color: #5b6770; width: 32%; }}
table.tiles td {{ border: 0; padding: 4pt 6pt 6pt 0; width: 16%; vertical-align: top; }}
.tv {{ font-size: 17pt; font-weight: bold; color: {ACCENT}; line-height: 1.1; margin: 0; }}
.tv.crit {{ color: #c0392b; }}
.tl {{ font-size: 7.6pt; color: #1d2b36; font-weight: bold; margin: 1pt 0 0 0; }}
.ts {{ font-size: 7.2pt; color: #5b6770; margin: 0; }}
.callout {{ border-left: 3px solid {ACCENT}; padding: 3pt 0 3pt 8pt; margin: 6pt 0; }}
.co-warn {{ border-left: 3px solid #d68910; }}
.co-fail {{ border-left: 3px solid #c0392b; }}
.kvl {{ margin: 0 0 2.5pt 0; padding-bottom: 2pt; border-bottom: 0.5px solid #e3e8eb; }}
.k {{ color: #5b6770; font-weight: bold; }}
.card {{ border-left: 3px solid #7b8a8b; padding: 1pt 0 2pt 8pt; margin: 8pt 0 6pt 0; }}
.cCRITICAL {{ border-left: 3px solid #c0392b; }}
.cMAJOR {{ border-left: 3px solid #d68910; }}
.tv.txt {{ font-size: 11pt; padding-top: 4pt; }}
.ct {{ font-weight: bold; font-size: 9.2pt; margin: 0 0 2pt 0; }}
.cm {{ color: #5b6770; font-size: 7.8pt; margin: 0 0 2pt 0; }}
.cl {{ color: #5b6770; font-size: 7.8pt; font-weight: bold; margin: 3pt 0 0 0; }}
.cv {{ margin: 0 0 1pt 0; }}
.why td {{ font-size: 7.9pt; }}
.why td.k {{ color: #5b6770; width: 22%; }}
.b {{ font-weight: bold; font-size: 7.4pt; }}
.CRITICAL {{ color: #c0392b; }} .MAJOR {{ color: #d68910; }} .MINOR {{ color: #7b8a8b; }}
.ok {{ color: {ACCENT}; font-weight: bold; }} .warn {{ color: #d68910; font-weight: bold; }}
.fail {{ color: #c0392b; font-weight: bold; }} .na {{ color: #9aa5ad; }}
.q {{ color: #5b6770; }}
table.mx td, table.mx th {{ text-align: center; padding: 2pt 1pt; font-size: 7.6pt; }}
table.mx td.l, table.mx th.l {{ text-align: left; }}
a {{ color: #00875d; text-decoration: none; }}
code {{ font-family: monospace; font-size: 7.6pt; color: #1d2b36; }}
hr {{ border: 0; border-top: 0.5px solid #d5dde2; margin: 8pt 0; }}
.band {{ border-top: 5px solid {ACCENT}; padding-top: 8pt; margin-bottom: 4pt; }}
.src {{ font-family: monospace; font-size: 7.3pt; margin: 0; white-space: pre-wrap; }}
.ln {{ color: #9aa5ad; }}
.ocr {{ color: #8a5a00; }}
.pg {{ font-size: 8pt; font-weight: bold; color: {ACCENT}; margin: 6pt 0 2pt 0; }}
.gap {{ font-size: 8pt; color: #c0392b; margin: 2pt 0; }}
"""


def to_html(o: Out) -> str:
    out = []
    used: set = set()
    for b in o.blocks:
        k = b[0]
        if k == "h":
            aid = b[3] or f"h{len(out)}"
            used.add(aid)
            out.append(f'<h{b[1]} id="{aid}">{_inline_html(b[2])}</h{b[1]}>')
        elif k == "p":
            out.append(f'<p class="{b[2]}">{_inline_html(b[1])}</p>')
        elif k == "callout":
            out.append(f'<div class="callout co-{b[2]}">' + "".join(f"<p>{_inline_html(x)}</p>" for x in b[1].split("\n"))
                       + "</div>")
        elif k == "ul":
            out.append(f'<ul class="{b[2]}">' + "".join(f"<li>{_inline_html(x)}</li>" for x in b[1]) + "</ul>")
        elif k == "table":
            head, rows, widths, cls = b[1:]
            w = widths or [None] * len(head)
            t = [f'<table class="{cls}"><tr>'
                 + "".join(f'<th{f" width=\"{x}%\"" if x else ""}{" class=l" if cls == "mx" and i == 0 else ""}>'
                           f'{_inline_html(h)}</th>' for i, (h, x) in enumerate(zip(head, w))) + "</tr>"]
            for r in rows:
                t.append("<tr>" + "".join(f'<td{" class=l" if cls == "mx" and i == 0 else ""}>{_cell_html(c)}</td>'
                                          for i, c in enumerate(r)) + "</tr>")
            out.append("".join(t) + "</table>")
        elif k == "kv":
            out.append('<div class="kvs">' + "".join(
                f'<p class="kvl"><span class="k">{_inline_html(a)}:</span> {_inline_html(v)}</p>' for a, v in b[1])
                       + "</div>")
        elif k == "tiles":
            cells = []
            for v, lab, sub, cls in b[1]:
                if not re.match(r"^[\d%., —-]+$", str(v)):
                    cls = (cls + " txt").strip()
                cells.append(f'<td><p class="tv {cls}">{html.escape(str(v))}</p><p class="tl">{html.escape(lab)}</p>'
                             f'<p class="ts">{html.escape(sub)}</p></td>')
            out.append('<table class="tiles"><tr>' + "".join(cells) + "</tr></table>")
        elif k == "card":
            anchor, sev, title, rows, meta = b[1:]
            h = [f'<div class="card c{sev}" id="{anchor}"><p class="ct">{_inline_html(title)}</p>']
            if meta:
                h.append(f'<p class="cm">{_inline_html(meta)}</p>')
            for lab, val in rows:
                if lab == "__why__":
                    h.append('<p class="cl">Почему это критично</p>' + "".join(
                        f'<p class="kvl"><span class="k">{html.escape(a)}:</span> {_inline_html(v)}</p>' for a, v in val))
                    continue
                h.append(f'<p class="cl">{html.escape(lab)}</p>')
                if isinstance(val, list):
                    h.append("<ul>" + "".join(f"<li>{_inline_html(x)}</li>" for x in val) + "</ul>")
                else:
                    h.append(f'<p class="cv">{_inline_html(val)}</p>')
            out.append("".join(h) + "</div>")
        elif k == "raw":
            out.append(b[1])
        elif k == "hr":
            out.append("<hr/>")
        elif k == "pb":
            out.append('<p style="page-break-before: always; margin:0"></p>')
    return "\n".join(out)


def _cell_html(c) -> str:
    s = str(c)
    m = re.match(r"^\{(\w+)\}(.*)$", s, re.S)       # «{fail}✗» — цветная метка
    if m:
        return f'<span class="{m.group(1)}">{_inline_html(m.group(2))}</span>'
    return _inline_html(s)


def _md_tagged(s: str) -> str:
    return re.sub(r"^\{\w+\}", "", str(s))


# ====================================================================== вспомогательное
_LOC_POS = re.compile(r"^(с\. \d+, L(\d{4})(?:–(?:с\. \d+, )?L\d{4})?|L(\d{4})(?:–L\d{4})?|строка (\d+)(?:–\d+)?)")


def link_loc(piece: str) -> str:
    """«строка 104, раздел «…»» → «[строка 104](#L0104), раздел «…»»."""
    piece = piece.strip()
    m = _LOC_POS.match(piece)
    if not m:
        return piece
    gid = int(m.group(2) or m.group(3) or m.group(4))
    return f"[{m.group(1)}](#L{gid:04d})" + piece[m.end():]


def locs_of(f: dict) -> list[str]:
    return [x for x in (f.get("location") or "").split("; ") if x.strip()]


def where_short(f: dict) -> str:
    ls = locs_of(f)
    if not ls:
        return "—"
    first = link_loc(ls[0]).split(", пункт")[0]
    return first + (f" · ещё {len(ls) - 1}" if len(ls) > 1 else "")


def sev_tag(sev: str) -> str:
    return "{" + sev + "}" + sev


class View:
    def __init__(self, data: dict):
        self.d = data
        self.reqs = data.get("requirements") or []
        self.req_by = {r["id"]: r for r in self.reqs}
        self.set_f = data.get("set_findings") or []
        self.req_f = [dict(f, _req=r["id"]) for r in self.reqs for f in r.get("findings") or []]
        self.all_f = self.req_f + self.set_f
        self.by_fid = {f["id"]: f for f in self.all_f}
        self.stats = data.get("stats") or {}
        self.status = (data.get("run_status") or {}).get("code", "completed")
        self.reasons = (data.get("run_status") or {}).get("reasons") or []
        self.ing = data.get("ingestion") or {}
        self.cov = data.get("coverage") or {}
        known = set(self.req_by)
        self._unknown = re.compile(r"\bREQ-\d+\b")
        self.line_word = "строка" if self.ing.get("fmt", "Markdown") in ("Markdown", "TXT") else "L"
        self.known = known

    def clean(self, text: str) -> str:
        """Убирает ссылки модели на несуществующие требования (например, REQ-68 при 46 требованиях)."""
        t = self._unknown.sub(lambda m: m.group(0) if m.group(0) in self.known else "", text or "")
        t = re.sub(r"\[L(\d+)\]", lambda m: self.line_link(int(m.group(1))), t)
        t = re.sub(r"\(\s*(,\s*)*\)", "", t)
        t = re.sub(r"\(\s*,\s*", "(", t)
        t = re.sub(r",\s*\)", ")", t)
        return re.sub(r"\s+([,.;)])", r"\1", collapse_ws(t))

    def line_link(self, gid: int) -> str:
        label = f"строка {gid}" if self.line_word == "строка" else f"L{gid:04d}"
        return f"[{label}](#L{gid:04d})"

    def gids(self, f: dict) -> set:
        out = set()
        for piece in locs_of(f):
            m = _LOC_POS.match(piece.strip())
            if m:
                nums = [int(x) for x in re.findall(r"(?:L|строка |–)(\d+)", m.group(1))]
                if nums:
                    out.update(range(min(nums), max(nums) + 1))
        return out

    def root_groups(self, fs: list[dict]) -> list[list[dict]]:
        """Группы дефектов с общей первопричиной: общие строки + близкий текст, либо очень близкий текст."""
        from .consolidate import similarity
        parent = {f["id"]: f["id"] for f in fs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        txt = {f["id"]: f["problem"] + " " + (f.get("recommendation") or "") for f in fs}
        g = {f["id"]: self.gids(f) for f in fs}
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                sim = similarity(txt[a["id"]], txt[b["id"]])
                if sim >= 0.45 or (sim >= 0.15 and g[a["id"]] & g[b["id"]]):
                    parent[find(b["id"])] = find(a["id"])
        groups: dict[str, list[dict]] = {}
        for f in fs:
            groups.setdefault(find(f["id"]), []).append(f)
        out = list(groups.values())
        for grp in out:
            grp.sort(key=lambda f: (-self.places(f), f["id"]))
        out.sort(key=lambda grp: (-sum(self.places(f) for f in grp), grp[0]["id"]))
        return out

    def ref(self, rid: str) -> str:
        r = self.req_by.get(rid)
        return f"{rid}" + (f" ({r['doc_id']})" if r and r.get("doc_id") else "")

    def refs(self, ids, limit=6) -> str:
        ids = [i for i in ids if i in self.req_by]
        if not ids:
            return "—"
        if len(ids) <= limit:
            return ", ".join(self.ref(i) for i in ids)
        order = {r["id"]: n for n, r in enumerate(self.reqs)}
        ids = sorted(ids, key=order.get)
        parts, run = [], []
        for i in ids:
            if run and order[i] == order[run[-1]] + 1:
                run.append(i)
            else:
                if run:
                    parts.append(run)
                run = [i]
        parts.append(run)
        return ", ".join(f"{p[0]}–{p[-1]}" if len(p) > 2 else ", ".join(p) for p in parts) + f" (всего {len(ids)})"

    def flink(self, fid: str) -> str:
        return f"[{fid}](#{fid})"

    def by_sev(self, sev):
        return [f for f in self.all_f if f["severity"] == sev]

    def places(self, f) -> int:
        return max(1, int(f.get("places") or len(locs_of(f)) or 1))

    def theme(self, f) -> str:
        return THEMES.get(f.get("tag") or "", (f.get("type") or "прочее").lower())


# ====================================================================== разделы
def build(data: dict, doc=None, include_source: bool = True) -> Out:
    v = View(data)
    o = Out()
    if data.get("findings") is None and not v.reqs and v.status == "failed":
        _diagnostic(v, o)
    else:
        _executive(v, o)
        _passport(v, o)
        _inventory(v, o)
        _requirements(v, o)
        _set(v, o)
        _stats(v, o)
        _verdict(v, o)
        _appendices(v, o)
    if include_source and doc is not None:
        _source_md(doc, o)
    return o


def _title(v: View, o: Out, kind: str):
    doc = v.d.get("document") or {}
    date = (v.d.get("date") or "")[:16].replace("T", " ")
    title = doc.get("title") or doc.get("file") or ""
    o.html(f'<div class="band"><p class="eyebrow">{html.escape(kind.upper())}</p>'
           f'<h1 id="top">{html.escape(title)}</h1>'
           f'<p class="sub">{html.escape(doc.get("file", ""))} · {html.escape(date)} · '
           f'{html.escape(v.d.get("model", ""))} · агент {html.escape(v.d.get("agent_version", ""))}</p></div>',
           f"# {kind}: «{title}»\n\n*{doc.get('file', '')} · {date} · {v.d.get('model', '')} · "
           f"агент {v.d.get('agent_version', '')}*\n")


def _headline(v: View) -> tuple[str, str]:
    s = v.stats
    bs = s.get("by_severity") or {}
    nc, nm = bs.get("critical", 0), bs.get("major", 0)
    crit = v.by_sev("CRITICAL")
    themes = list(dict.fromkeys(v.theme(f) for f in crit))
    partial = "" if v.status == "completed" else " (по проверенной части документа)"
    if v.status == "na":
        return "В документе нет оцениваемых требований — оценка качества не выставляется.", "na"
    if nc:
        ng = len(v.root_groups(crit))
        return (f"Документ не готов к разработке{partial}: {nc} {plural(nc, 'блокер', 'блокера', 'блокеров')} "
                + (f"сводятся к {ng} {plural(ng, 'решению', 'решениям', 'решениям')} " if ng < nc else "")
                + f"({', '.join(themes[:3])}) — их нужно принять до старта спринта.", "fail")
    if nm:
        return (f"Блокеров нет{partial}, но {nm} {plural(nm, 'существенный дефект', 'существенных дефекта', 'существенных дефектов')} "
                "нужно закрыть до начала работ по затронутым функциям.", "warn")
    if v.status != "completed":
        return "В пределах выполненных проверок существенных дефектов не выявлено; общий вердикт не выносится.", "warn"
    return "Документ готов к разработке: существенных дефектов нет, замечания можно закрыть в рабочем порядке.", "ok"


def _executive(v: View, o: Out):
    _title(v, o, "Отчёт о проверке требований")
    s = v.stats
    bs = s.get("by_severity") or {}
    head, kind = _headline(v)
    o.h(2, "Резюме для руководителя", "summary")
    o.p(head, "lead")
    reqs_ok = s.get("without_issues", 0)
    pct = v.cov.get("completed_pct")
    st_label, st_cls = RUN.get(v.status, (v.status, "na"))
    o.tiles([
        (s.get("requirements", 0), "требований", f"{reqs_ok} без замечаний", ""),
        (s.get("findings", 0), "уникальных дефектов", f"в {s.get('occurrences', 0)} местах", ""),
        (bs.get("critical", 0), "блокеров (CRITICAL)", "до старта спринта", "crit"),
        (bs.get("major", 0), "существенных (MAJOR)", f"{bs.get('minor', 0)} мелких (MINOR)", ""),
        (f"{pct:.0f}%" if pct is not None else "—", "текста проверено", "посчитано кодом", ""),
        (st_label, "статус проверки", "см. «Объём проверки»", "crit" if st_cls == "fail" else ""),
    ])
    share = s.get("primary_share")
    if share is not None and share < 0.5:
        o.callout(f"Основная модель обработала только {round(share * 100)}% запросов (остальные — резервные модели "
                  "из-за перегрузки сервиса). Глубина анализа может быть ниже; рекомендуется повторный запуск.", "warn")
    crit = v.by_sev("CRITICAL")
    if crit:
        groups = v.root_groups(crit)
        o.h(3, "Что блокирует разработку" + (f" — {len(groups)} {plural(len(groups), 'решение', 'решения', 'решений')}"
                                              if len(groups) < len(crit) else ""))
        rows = []
        for n, grp in enumerate(groups, 1):
            f = grp[0]
            dec = (f.get("why_critical") or {}).get("resolution") or f.get("recommendation") or ""
            ids = f"**{v.flink(f['id'])}**" + (" + " + ", ".join(v.flink(x["id"]) for x in grp[1:]) if grp[1:] else "")
            rows.append([str(n), ids, short(v.clean(f["problem"]), 170), where_short(f), short(v.clean(dec), 150)])
        o.table(["#", "Дефекты", "Проблема", "Где", "Нужное решение"], rows, [4, 11, 40, 17, 28])
        if len(groups) < len(crit):
            o.note("Дефекты в одной строке таблицы имеют общую первопричину: одно решение закрывает их все.")
    themes = Counter(v.theme(f) for f in v.all_f if f["severity"] in ("CRITICAL", "MAJOR"))
    if themes:
        o.h(3, "Где сосредоточены проблемы")
        o.bullets([f"**{t[:1].upper() + t[1:]}** — {n} {plural(n, 'дефект', 'дефекта', 'дефектов')} (CRITICAL/MAJOR)"
                   for t, n in themes.most_common(5)])
    qs = [q for r in v.reqs for q in r.get("questions") or []] + [e["question"] for e in v.d.get("edge_cases") or []
                                                                   if e.get("question")]
    o.h(3, "Что нужно сделать")
    steps = []
    if crit:
        steps.append(f"Аналитику и владельцу продукта — принять решения по {len(crit)} "
                     f"{plural(len(crit), 'блокеру', 'блокерам', 'блокерам')} (таблица выше) и обновить документ.")
    maj = v.by_sev("MAJOR")
    if maj:
        steps.append(f"Закрыть {len(maj)} {plural(len(maj), 'существенный дефект', 'существенных дефекта', 'существенных дефектов')} "
                     "до начала работ по затронутым функциям — план в разделе [6](#verdict).")
    if qs:
        steps.append(f"Ответить на {len(qs)} {plural(len(qs), 'вопрос', 'вопроса', 'вопросов')} по пограничным "
                     "сценариям — от ответов зависят тест-кейсы ([раздел 6](#verdict)).")
    steps.append("После исправлений — повторно прогнать документ через агента и сравнить отчёты.")
    o.bullets(steps)
    o.h(3, "Объём и надёжность проверки", "scope")
    issues = v.ing.get("issues") or []
    checks = v.cov.get("checks") or []
    done = sum(1 for c in checks if c["status"] == "выполнена")
    line = (f"Статус: **{st_label}**. " + (" ".join(v.reasons) + " " if v.reasons else "")
            + f"Проверено {num(v.cov.get('completed_chars'))} из {num(v.cov.get('canonical_chars'))} символов "
            f"извлечённого текста; проверок документа целиком выполнено {done} из {len(checks)}. "
            + ("Инцидентов чтения файла нет." if not issues else
               f"Инцидентов чтения файла: {len(issues)} — см. [раздел 1.1](#scope-detail)."))
    o.callout(line, st_cls if st_cls in ("warn", "fail") else "info")
    dropped = len(v.d.get("dropped") or []) or _dropped_count(v)
    if dropped:
        o.note(f"Самопроверка: второй ревьюер отклонил {dropped} {plural(dropped, 'кандидат', 'кандидата', 'кандидатов')} "
               "в дефекты как ложные срабатывания; повторы одной первопричины объединены "
               f"({s.get('merged', 0)}). Все цитаты сверены с текстом документа.")
    o.rule()


def _dropped_count(v: View) -> int:
    for w in v.d.get("warnings") or []:
        m = re.search(r"отклонено ложных срабатываний — (\d+)", w)
        if m:
            return int(m.group(1))
    return 0


def _model_line(v: View) -> str:
    st = v.d.get("stats") or {}
    used = st.get("model_usage") or {}
    if not used:
        return v.d.get("model", "—")
    total = sum(used.values())
    parts = ", ".join(f"{m} — {n}" for m, n in sorted(used.items(), key=lambda x: -x[1]))
    line = f"{v.d.get('model', '—')} (запросов: {total}; по моделям: {parts})"
    share = st.get("primary_share")
    if share is not None and share < 0.5:
        line += f". Основная модель обработала только {round(share * 100)}% запросов — рекомендуется повторный запуск"
    return line


def _passport(v: View, o: Out):
    doc = v.d.get("document") or {}
    ing = v.ing
    o.h(2, "1. Паспорт прогона", "passport")
    o.kv([("Документ", f"«{doc.get('title', '')}» · `{doc.get('file', '')}`"),
          ("Дата и время прогона", (v.d.get("date") or "").replace("T", " ")),
          ("Модель", _model_line(v)),
          ("Версия агента", v.d.get("agent_version", "—")),
          ("Время анализа", f"{v.d.get('duration_sec', 0):.0f} сек"),
          ("SHA-256 документа", f"`{doc.get('sha256', '')}`"),
          ("Стандарты", "ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6; IEEE Std 830-1998 п. 4.3; руководство "
                        "организаторов (Приложение А)")])
    o.h(3, "1.1. Что прочитано и проверено", "scope-detail")
    _scope_table(v, o)
    issues = ing.get("issues") or []
    if issues:
        o.table(["ID", "Где", "Что наблюдается", "Достоверность", "Влияние", "Что сделать"],
                [[i["id"], i["location"], i["observation"], i["certainty"], i.get("effect") or "—",
                  i.get("next_action") or "—"] for i in issues], [7, 12, 30, 12, 20, 19])
        o.note("ING-… — проблемы чтения файла, а не дефекты спецификации; в счётчики дефектов не входят.")


def _scope_table(v: View, o: Out):
    ing, cov = v.ing, v.cov
    from .ingest import STATUS_RU, human_size
    rows = [("Файл", f"{num(ing.get('size'))} байт ({human_size(ing.get('size') or 0)}) · формат {ing.get('detected', '—')}"
             + (f" · {ing['encoding']}" if ing.get("encoding") else "")),
            ("Приём документа", STATUS_RU.get(ing.get("status", ""), ing.get("status", "—"))
             + f" · {ing.get('parser', '')} · OCR: {ing.get('ocr_state', '—')}")]
    pages = ing.get("pages") or []
    if ing.get("fmt") == "PDF":
        c = Counter(p["status"] for p in pages)
        rows.append(("Страницы", f"всего {num(ing.get('expected_pages'))}; прочитано {c.get('readable', 0)}, частично "
                                 f"{c.get('partial', 0)}, под сомнением {c.get('uncertain', 0)}, не прочитано "
                                 f"{c.get('unreadable', 0)}, пустых {c.get('blank', 0)}"))
    if cov:
        rows += [("Символов извлечено", num(cov.get("canonical_chars"))),
                 ("Передано модели / проверено", f"{num(cov.get('submitted_chars'))} / {num(cov.get('completed_chars'))}"
                  + (f" ({cov['completed_pct']:.1f}%)".replace(".", ",") if cov.get("completed_pct") is not None else "")),
                 ("Не проверено", num(cov.get("unreviewed_chars"))),
                 ("Проверки документа целиком", "; ".join(f"{c['title']} — {c['status']}" for c in cov.get("checks") or []))]
    o.kv(rows)


def _inventory(v: View, o: Out):
    o.h(2, "2. Нарезка и инвентаризация требований", "inventory")
    no_id = sum(1 for r in v.reqs if not r.get("doc_id"))
    kinds = Counter(r.get("kind") for r in v.reqs)
    o.p(f"Выделено **{len(v.reqs)}** элементарных требований"
        + (f"; у {no_id} нет номера в документе — присвоен REQ-xx, цитата сохранена дословно" if no_id else "") + ". "
        + "По типам: " + ", ".join(f"{k} — {n}" for k, n in kinds.most_common()) + ".")
    rows = []
    for r in v.reqs:
        own = [f for f in r.get("findings") or []]
        worst = min([f["severity"] for f in own] + [v.by_fid[s]["severity"] for s in r.get("systemic_defects") or []
                                                   if s in v.by_fid], key=lambda x: SEV_ORDER[x], default=None)
        ids = [f["id"] for f in own] + list(r.get("systemic_defects") or [])
        res = ("{" + worst + "}" + ", ".join(ids)) if worst else "{ok}✓"
        loc = link_loc((r.get("location") or "").split(", раздел")[0])
        sec = " · ".join(x for x in (r.get("part"), r.get("section")) if x)
        rid = f"**{r['id']}**" + (f"<br>{r['doc_id']}" if r.get("doc_id") else "")
        rows.append([rid.replace("<br>", " · "), f"{loc}<br>{sec}".replace("<br>", " · "),
                     f"**{r.get('summary', '')}** — «{r.get('text', '')}»", res])
    o.table(["ID", "Где", "Формулировка и точная цитата", "Итог"], rows, [10, 22, 56, 12])
    o.note("Итог: ✓ — замечаний нет; иначе — ID дефектов (F — дефект требования, S — системный дефект набора).")


def _finding_card(v: View, o: Out, f: dict, with_reqs=True):
    sev = f["severity"]
    crit = INDIVIDUAL.get(f.get("criterion")) or SET.get(f.get("criterion"))
    title = f"{f['id']} · {{{sev}}}{SEV[sev][0]} ({sev}){{/}} · {short(v.clean(f['problem']), 120)}"
    meta = " · ".join(x for x in (
        (crit[0] if crit else f.get("criterion", "")), f.get("part") or "", f.get("tag") or "",
        ("автоматическая проверка" if f.get("source") == "правило" else ""),
        (f"мест: {v.places(f)}" if v.places(f) > 1 else "")) if x)
    rows = []
    if with_reqs and f.get("requirements"):
        rows.append(("Требования", v.refs(f["requirements"])))
    ls, qs = locs_of(f), f.get("quotes") or []
    if len(ls) == len(qs) and ls:
        places = [f"{link_loc(a)} — «{short(q, 220)}»" for a, q in zip(ls, qs)]
    else:
        places = [link_loc(a) for a in ls] + [f"«{short(q, 220)}»" for q in qs[:3]]
    rows.append(("Где", places if len(places) > 1 else (places[0] if places else "—")))
    if len(v.clean(f["problem"])) > 120:
        rows.append(("Суть", v.clean(f["problem"])))
    if f.get("recommendation"):
        rows.append(("Как исправить", v.clean(f["recommendation"])))
    if f.get("caveat"):
        rows.append(("Оговорка", f["caveat"]))
    if sev == "CRITICAL":
        why = f.get("why_critical") or {}
        if why:
            rows.append(("__why__", [(t, v.clean(why[k])) for k, t in WHY if why.get(k)]))
        else:
            rows.append(("Почему это критично", "обоснование не сформировано (см. журнал проверки)."))
    o.card(f["id"], sev, title, rows, meta)


def _requirements(v: View, o: Out):
    o.h(2, "3. Проверка требований", "requirements")
    issues = [r for r in v.reqs if r.get("findings") or r.get("related") or r.get("questions")
              or any(st in ("warn", "fail") for st in (r.get("criteria") or {}).values())]
    systemic_only = [r for r in v.reqs if r not in issues and r.get("systemic_defects")]
    clean_n = len(v.reqs) - len(issues) - len(systemic_only)
    o.p(f"**{clean_n}** {plural(clean_n, 'требование соответствует', 'требования соответствуют', 'требований соответствуют')} "
        f"всем критериям — подробно не описываются (см. таблицу в разделе 2 и полную матрицу в "
        f"[Приложении Г](#app-matrix)). Ниже — только требования с замечаниями: **{len(issues)}** с собственными "
        f"дефектами или вопросами и **{len(systemic_only)}** только с системными дефектами набора.")
    if issues:
        o.h(3, "3.1. Матрица по критериям (требования с замечаниями)")
        codes = list(INDIVIDUAL)
        rows = []
        for r in issues + systemic_only:
            cr = r.get("criteria") or {}
            rows.append([r["id"]] + [("{" + cr.get(c, "na") + "}" + MARK[cr.get(c, "na")]) for c in codes])
        o.table(["Требование"] + codes, rows, None, "mx")
        o.note("✓ выполнен · ! замечание · ✗ нарушение · · неприменим. Коды: " + "; ".join(
            f"{c} — {INDIVIDUAL[c][0]}" for c in codes) + ".")
        o.h(3, "3.2. Требования с собственными замечаниями")
        for r in issues:
            cr = r.get("criteria") or {}
            own_c = {f.get("criterion") for f in r.get("findings") or []}
            bad = [c for c in INDIVIDUAL if cr.get(c) in ("warn", "fail") and c not in own_c]
            title = f"{r['id']}" + (f" ({r['doc_id']})" if r.get("doc_id") else "") + f" · {short(r.get('summary', ''), 90)}"
            o.h(4, title, f"req-{r['id']}")
            o.p(f"{link_loc(r.get('location', ''))} · «{r.get('text', '')}»", "q")
            if bad:
                o.p(("**Также нарушены:** " if own_c else "**Нарушенные критерии:** ") + "; ".join(
                    f"{INDIVIDUAL[c][0]} — {v.clean((r.get('criteria_comments') or {}).get(c, '')) or 'см. дефект ниже'}"
                    for c in bad))
            for f in r.get("findings") or []:
                _finding_card(v, o, f, with_reqs=False)
            if r.get("related"):
                o.p("**Противоречит другим требованиям:** " + ", ".join(
                    f"{v.flink(fid)} у {v.ref(rid)}" for fid, rid in r["related"]))
            if r.get("systemic_defects"):
                o.p("**Также затронуто системными дефектами:** " + ", ".join(v.flink(s) for s in r["systemic_defects"]))
            if r.get("test_ideas"):
                o.p("**Идеи тест-кейсов:** " + "; ".join(r["test_ideas"]))
            if r.get("questions"):
                o.p("**Вопросы к аналитику:** " + " ".join(r["questions"]))
    if systemic_only:
        o.h(3, "3.3. Требования, затронутые только системными дефектами")
        groups: dict[tuple, list[str]] = {}
        for r in systemic_only:
            groups.setdefault(tuple(r["systemic_defects"]), []).append(r["id"])
        o.table(["Требования", "Системные дефекты (описаны в разделе 4)"],
                [[v.refs(ids), ", ".join(f"{v.flink(s)} — {short(v.clean(v.by_fid[s]['problem']), 90)}"
                                         for s in key if s in v.by_fid)] for key, ids in groups.items()], [30, 70])


SET_GROUPS = [("4.1. Системная полнота", ["S-COMPL", "S-RANK"]),
              ("4.2. Взаимная непротиворечивость", ["S-CONS", "S-COMP"]),
              ("4.3. Архитектурная осуществимость и стандарты", ["S-FEAS", "S-BND"]),
              ("4.4. Прослеживаемость и модифицируемость", ["S-TRC", "S-MOD"])]


def _set(v: View, o: Out):
    o.h(2, "4. Проверка набора требований в целом", "set")
    sc = v.d.get("set_criteria") or {}
    rows = []
    for c, val in sc.items():
        lab, cls = SET_STATUS.get(val.get("status"), (val.get("status"), "na"))
        fs = [f["id"] for f in v.set_f if f.get("criterion") == c]
        rows.append([f"**{SET[c][0]}**<br>{SET[c][1]}".replace("<br>", " · "), "{" + cls + "}" + lab,
                     v.clean(val.get("comment") or "—"), ", ".join(v.flink(x) for x in fs) or "—"])
    o.table(["Критерий набора", "Оценка", "Комментарий", "Дефекты"], rows, [26, 11, 49, 14])
    for title, codes in SET_GROUPS:
        fs = [f for f in v.set_f if f.get("criterion") in codes]
        fs.sort(key=lambda f: (SEV_ORDER[f["severity"]], -v.places(f)))
        o.h(3, title)
        if not fs:
            o.p("Дефектов не выявлено.")
            continue
        for f in fs:
            if f["severity"] == "MINOR":
                continue
            _finding_card(v, o, f)
        minors = [f for f in fs if f["severity"] == "MINOR"]
        if minors:
            o.table(["ID", "Замечание (MINOR)", "Где", "Как исправить"],
                    [[f"**{f['id']}**", v.clean(f["problem"]), where_short(f), v.clean(f.get("recommendation") or "—")]
                     for f in minors], [7, 43, 20, 30])
            for f in minors:
                o.html(f'<p id="{f["id"]}" style="margin:0"></p>')
    ec = v.d.get("edge_cases") or []
    o.h(3, "4.5. Непокрытые пограничные сценарии", "edge")
    if ec:
        o.p("Документ не определяет поведение в этих сценариях. Это вопросы к аналитику, а не дефекты.")
        o.table(["№", "Сценарий", "Требования", "Риск", "Вопрос"],
                [[f"EC-{i:02d}", e["scenario"], v.refs(e.get("req_ids") or []), e.get("risk") or "—",
                  e.get("question") or "—"] for i, e in enumerate(ec, 1)], [6, 24, 14, 28, 28])
    else:
        o.p("Дополнительных пограничных сценариев не выделено.")
    o.p("Матрица прослеживаемости (цель → UI → требование → API/данные → приёмка) — в [Приложении Г](#app-matrix).",
        "note")


def _stats(v: View, o: Out):
    s = v.stats
    bs = s.get("by_severity") or {}
    o.h(2, "5. Сводная статистика", "stats")
    req_sev = Counter(f["severity"] for f in v.req_f)
    set_sev = Counter(f["severity"] for f in v.set_f)
    rows = [[f"{SEV[k][0]} ({k})", str(req_sev.get(k, 0)), str(set_sev.get(k, 0)),
             f"**{bs.get(k.lower(), 0)}**", str(sum(v.places(f) for f in v.all_f if f['severity'] == k))]
            for k in SEV]
    rows.append(["**Итого**", str(len(v.req_f)), str(len(v.set_f)), f"**{s.get('findings', 0)}**",
                 str(s.get("occurrences", 0))])
    o.table(["Критичность", "В требованиях", "В наборе (системные)", "Уникальных", "Мест в документе"], rows,
            [30, 17, 19, 16, 18])
    o.kv([("Требований всего / без замечаний / с замечаниями",
           f"{s.get('requirements', 0)} / {s.get('without_issues', 0)} / {s.get('with_issues', 0)}"),
          ("Объединено повторов одной первопричины", str(s.get("merged", 0))),
          ("Найдено автоматическими проверками (без LLM)", str(s.get("rule_findings", 0))),
          ("Отклонено самопроверкой как ложные срабатывания", str(len(v.d.get("dropped") or []) or _dropped_count(v))),
          ("Пограничных сценариев (вопросы, не дефекты)", str(len(v.d.get("edge_cases") or []))),
          ("Дефекты требований по критериям", ", ".join(f"{c} — {n}" for c, n in sorted(
              (s.get("by_criterion") or {}).items(), key=lambda x: -x[1]) if n) or "—"),
          ("По частям документа", ", ".join(f"{p} — {n}" for p, n in (s.get("by_part") or {}).items()) or "—")])


def _verdict(v: View, o: Out):
    o.h(2, "6. Итоговый вердикт и план действий", "verdict")
    head, kind = _headline(v)
    text = head
    if v.status != "completed":
        text += "\nСтатус проверки: " + RUN.get(v.status, (v.status,))[0] + ". " + " ".join(v.reasons) + \
            " Вывод относится только к проверенной части; общий вердикт «готов к разработке» не выносится."
    o.callout(text, kind if kind in ("warn", "fail") else "info")
    prio = sorted([f for f in v.all_f if f["severity"] in ("CRITICAL", "MAJOR")],
                  key=lambda f: (SEV_ORDER[f["severity"]], -v.places(f), f["id"]))
    if prio:
        o.h(3, "План исправлений (по приоритету)")
        groups = v.root_groups([f for f in prio if f["severity"] == "CRITICAL"]) + \
            v.root_groups([f for f in prio if f["severity"] == "MAJOR"])
        rows = []
        for i, grp in enumerate(groups, 1):
            f = grp[0]
            reqs = list(dict.fromkeys(x for g in grp for x in (g.get("requirements") or ([g["_req"]] if g.get("_req") else []))))
            rows.append([str(i), sev_tag(f["severity"]),
                         v.clean((f.get("why_critical") or {}).get("resolution") or f.get("recommendation") or f["problem"]),
                         ", ".join(v.flink(g["id"]) for g in grp), v.refs(reqs, 3)])
        o.table(["#", "Приоритет", "Что сделать", "Дефекты", "Требования"], rows, [4, 11, 52, 12, 21])
    minors = [f for f in v.all_f if f["severity"] == "MINOR"]
    if minors:
        o.p("Замечания MINOR (" + str(len(minors)) + ") — исправить в рабочем порядке: "
            + ", ".join(v.flink(f["id"]) for f in minors) + ".")
    qs = [(r["id"], q) for r in v.reqs for q in r.get("questions") or []] + \
         [(f"EC-{i:02d}", e["question"]) for i, e in enumerate(v.d.get("edge_cases") or [], 1) if e.get("question")]
    from .consolidate import similarity
    uniq: list[tuple[list, str]] = []           # один вопрос — одна строка, со всеми источниками
    for a, q in qs:
        host = next((u for u in uniq if similarity(u[1], q) >= 0.6), None)
        if host is None:
            uniq.append(([a], q))
        elif a not in host[0]:
            host[0].append(a)
    if uniq:
        o.h(3, "Вопросы к аналитику и бизнесу")
        o.table(["Откуда", "Вопрос"], [[", ".join(v.ref(a) if a in v.req_by else a for a in src), q]
                                       for src, q in uniq], [14, 86])


def _appendices(v: View, o: Out):
    o.h(2, "Приложение А. Критерии и методика", "app-a")
    o.table(["Код", "Критерий", "Источник"], [[c, n, s] for c, (n, s, _) in INDIVIDUAL.items()]
            + [[c, n, s] for c, (n, s, _) in SET.items()], [10, 45, 45])
    o.p("**Критичность:** " + " ".join(x.strip() for x in CRITICAL_HINT.split("\n")))
    o.p("**Как работает агент:** приём файла (SHA-256, формат, целостность, OCR сканов) → канонический транскрипт с "
        "номерами строк → выделение требований с дословной сверкой → проверка каждого требования по 10 критериям → "
        "проверка набора в три прохода + автоматические правила → объединение по первопричине → самопроверка "
        "(второй ревьюер, запрет выдуманных значений, согласованность отметок) → «почему это критично» → подсчёт "
        "покрытия и статуса кодом.")
    o.note("Символ — кодовая точка Unicode извлечённого текста, включая пробелы и переводы строк; метки строк, "
           "промпты и текст отчёта не считаются. 100% — покрытие извлечённого текста, а не всего файла, если в "
           "разделе 1.1 есть нечитаемые части.")
    o.h(2, "Приложение Б. Журнал проверки", "app-b")
    o.bullets(list(dict.fromkeys(v.d.get("warnings") or [])) or ["Предупреждений нет."])
    dropped = v.d.get("dropped") or []
    if dropped:
        o.table(["Где", "Отклонённый кандидат", "Почему", "Доказательство"],
                [[d_["where"], short(d_["problem"], 200), d_.get("reason") or "—", d_.get("evidence") or "—"]
                 for d_ in dropped], [12, 40, 24, 24])
    if v.cov:
        o.h(2, "Приложение В. Приём документа и покрытие", "app-c")
        pages = v.ing.get("pages") or []
        if v.ing.get("fmt") == "PDF" and pages:
            from .ingest import METHOD_RU, PAGE_RU
            o.table(["Стр.", "Статус", "Способ", "Символов (текст)", "Символов (OCR)", "Уверенность OCR"],
                    [[str(p["page"]), PAGE_RU.get(p["status"], p["status"]), METHOD_RU.get(p["method"], p["method"]),
                      str(p["native_chars"]), str(p["ocr_chars"] or "—"),
                      f"{p['ocr_conf']:.0f}" if p.get("ocr_conf") is not None else "—"] for p in pages])
        man = v.cov.get("manifest") or []
        ext = [c for c in man if c["kind"] == "extract"]
        rev = [c for c in man if c["kind"] == "review"]
        o.p(f"Фрагментов выделения: {len(ext)} (успешно {sum(c['status'] == 'completed' for c in ext)}); "
            f"проверок требований: {len(rev)} (успешно {sum(c['status'] == 'completed' for c in rev)}). "
            f"Инвариант проверено ≤ передано ≤ извлечено: {num(v.cov.get('completed_chars'))} ≤ "
            f"{num(v.cov.get('submitted_chars'))} ≤ {num(v.cov.get('canonical_chars'))}.")
        for n in v.ing.get("notes") or []:
            o.note(n)
    o.h(2, "Приложение Г. Полная матрица критериев и прослеживаемость", "app-matrix")
    codes = list(INDIVIDUAL)
    o.table(["Требование"] + codes, [[r["id"]] + [("{" + (r.get("criteria") or {}).get(c, "na") + "}"
                                                   + MARK[(r.get("criteria") or {}).get(c, "na")]) for c in codes]
                                     for r in v.reqs], None, "mx")
    tr = v.d.get("traceability") or []
    if tr:
        cell = lambda x: ", ".join(x) if x else "{fail}нет"  # noqa: E731
        o.table(["Требование", "Цель", "UI/UX", "API / данные", "Приёмка", "Разрыв"],
                [[v.ref(t["req"]), cell(t["goal"]), cell(t["ui"]), cell(t["api_data"]), cell(t["acceptance"]),
                  v.clean(t.get("gap") or "—")] for t in tr], [14, 10, 10, 14, 12, 40])


def _diagnostic(v: View, o: Out):
    _title(v, o, "Диагностика: проверка не выполнена")
    o.h(2, "Резюме", "summary")
    o.p("Проверка требований по существу не выполнялась — оценка качества не выставляется "
        "(это не значит, что документ хороший или плохой).", "lead")
    o.callout("Причина: " + " ".join(v.reasons), "fail")
    acts = list(dict.fromkeys(i.get("next_action") for i in v.ing.get("issues") or [] if i.get("next_action")))
    o.h(3, "Что сделать")
    o.bullets(acts or ["Устраните причину и запустите проверку снова."])
    o.h(3, "Что прочитано", "scope-detail")
    _scope_table(v, o)
    issues = v.ing.get("issues") or []
    if issues:
        o.table(["ID", "Где", "Что наблюдается", "Достоверность", "Что предпринято → результат", "Влияние"],
                [[i["id"], i["location"], i["observation"], i["certainty"], f"{i['recovery']} → {i['result']}",
                  i.get("effect") or "—"] for i in issues], [7, 10, 33, 13, 20, 17])
    o.kv([("Модель", v.d.get("model", "—")), ("Версия агента", v.d.get("agent_version", "—")),
          ("SHA-256", f"`{(v.d.get('document') or {}).get('sha256') or v.ing.get('sha256', '—')}`")])


def _source_md(doc, o: Out):
    """Исходный текст: в PDF — отдельный HTML с якорями строк, в Markdown — блок кода."""
    from .pdf import source_html
    lines = ["```text"]
    cur = None
    for ln in doc.lines:
        if ln.page is not None and ln.page != cur:
            cur = ln.page
            lines.append(f"===== с. {cur} =====")
        lines.append(f"L{ln.gid:04d}{' OCR' if ln.src == 'ocr' else '    '} │ {ln.text}")
    lines.append("```")
    title = "Приложение Д. Исходный текст (канонический транскрипт)"
    o.html(source_html(doc, title).replace('<h2 id="source">', '<h2 id="source">'),
           f"## {title}\n\nСтроки исходного документа с номерами; на них ссылаются все места в отчёте.\n\n"
           + "\n".join(lines) + "\n")


# ====================================================================== выход
def render_markdown(data: dict, doc=None) -> str:
    md = to_markdown(build(data, doc))
    return re.sub(r"\{(ok|warn|fail|na|CRITICAL|MAJOR|MINOR|/)\}", "", md).replace("<br>", " · ")


def write_pdf(data: dict, doc, path, footer: str = ""):
    from .pdf import write_pdf_html
    o = build(data, doc)
    body = to_html(o)
    return write_pdf_html(body, path, CSS, footer)
