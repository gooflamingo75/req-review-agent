"""Детерминированные (без LLM) проверки: словарь размытых слов, маркеры TBD,
проектные решения, неатомарность, дубли и пропуски идентификаторов.

Результаты используются как подсказки для модели и как самостоятельные
замечания, если модель их не учла.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from .util import norm

_I = re.IGNORECASE

# (регулярное выражение, [критерии], пояснение, серьёзность)
WORD_RULES = [
    # --- размытые качественные оценки: неоднозначны и непроверяемы ---
    (r"\bбыстр\w*", ["UNA", "VER"], "оценка «быстро» без числовой меры (время, единицы, условия)", "major"),
    (r"\bмедленн\w*", ["UNA", "VER"], "оценка «медленно» без числовой меры", "major"),
    (r"\bудобн\w*", ["UNA", "VER"], "субъективная оценка «удобно» без измеримого критерия", "major"),
    (r"\bинтуитивн\w*", ["UNA", "VER"], "субъективная оценка «интуитивно» без критерия приёмки", "major"),
    (r"\bпонятн\w*", ["UNA", "VER"], "субъективная оценка «понятно» без критерия приёмки", "major"),
    (r"\bпрост(ой|ая|ое|ые|ого|ым|о)\b", ["UNA", "VER"], "субъективная оценка «простой» без критерия", "major"),
    (r"\bлегк(о|ий|ая|ое|ие)\b", ["UNA", "VER"], "субъективная оценка «легко» без критерия", "major"),
    (r"\bэффективн\w*", ["UNA", "VER"], "«эффективно» — не указано, чем и как измеряется эффективность", "major"),
    (r"\bоптимальн\w*", ["UNA", "VER"], "«оптимально» — не указан критерий оптимальности", "major"),
    (r"\bнад[её]жн\w*", ["UNA", "VER"], "«надёжно» — не указаны показатели надёжности (MTBF, доступность, %)", "major"),
    (r"\bбезопасн\w*", ["UNA", "VER"], "«безопасно» — не указаны конкретные меры/стандарт безопасности", "minor"),
    (r"\bдостаточн\w*", ["UNA", "VER"], "«достаточно» — не указано пороговое значение", "major"),
    (r"\bприемлем\w*", ["UNA", "VER"], "«приемлемо» — не указан порог приемлемости", "major"),
    (r"\bсовременн\w*", ["UNA", "VER"], "«современный» — неизмеримая и устаревающая характеристика", "major"),
    (r"\bдружелюбн\w*", ["UNA", "VER"], "«дружелюбный» — субъективная оценка", "major"),
    (r"\bгибк\w*", ["UNA", "VER"], "«гибкий» — не определено, какие изменения должны поддерживаться", "minor"),
    (r"\bсвоевременн\w*", ["UNA", "VER"], "«своевременно» — не указан срок", "major"),
    (r"\bоперативн\w*", ["UNA", "VER"], "«оперативно» — не указан срок/время реакции", "major"),
    (r"\bв\s+кратчайш\w+\s+срок\w*|\bкак\s+можно\s+(быстрее|скорее)", ["UNA", "VER"],
     "срок не задан числом — «в кратчайшие сроки» нельзя проверить", "major"),
    (r"\bмгновенн\w*", ["FEA", "VER"], "«мгновенно» — недостижимо буквально; нужно допустимое время", "major"),
    (r"\bкачественн\w*", ["UNA", "VER"], "«качественно» — не указан критерий качества", "minor"),
    (r"\bкорректн\w*", ["UNA"], "«корректно» — не определено, что считается корректным поведением", "minor"),
    (r"\bнормальн\w*", ["UNA"], "«нормально/нормальный» — не определено", "minor"),
    (r"\bвысок(ая|ой|ую|ий|ое|ие|ого|ую)\s+(производительн|скорост|доступн|надежн|надёжн|качеств|точност|степен)\w*",
     ["UNA", "VER"], "«высокий/высокая …» без числового значения", "major"),
    (r"\bмаксимальн\w*\s+(быстр|удобн|прост|эффективн|надежн|надёжн)\w*", ["UNA", "VER"],
     "«максимально …» без числового значения", "major"),
    (r"\bбольш(ое|ого|ом|им|ой|ую)\s+(количеств|числ|объ[её]м)\w*", ["UNA", "VER"],
     "«большое количество/объём» без числа", "major"),
    (r"\bмного\b|\bмножеств\w*", ["UNA"], "неопределённое количество («много»)", "minor"),
    (r"\bнескольк\w*", ["UNA"], "неопределённое количество («несколько»)", "minor"),
    (r"\bнекотор\w*", ["UNA", "COM"], "неопределённое множество («некоторые»)", "minor"),
    (r"\bразличн\w*", ["UNA", "COM"], "неопределённое множество («различные») — перечень не задан", "minor"),
    (r"\bи\s+т\.\s?[дп]\.?|\bи\s+др\.|\bи\s+прочее\b|\bи\s+так\s+далее\b", ["UNA", "COM"],
     "открытый перечень («и т.д./и т.п.») — состав не определён", "major"),
    (r"\bи/или\b", ["UNA"], "«и/или» допускает несколько толкований", "minor"),
    (r"\bпо\s+возможности\b|\bесли\s+возможно\b|\bпо\s+мере\s+возможност\w*", ["UNA", "VER"],
     "«по возможности» делает требование необязательным и непроверяемым", "major"),
    (r"\bпри\s+необходимости\b|\bв\s+случае\s+необходимости\b|\bесли\s+потребуется\b", ["UNA", "COM"],
     "«при необходимости» — не определено, кто и когда решает о необходимости", "major"),
    (r"\bкак\s+правило\b|\bобычно\b|\bв\s+основном\b|\bв\s+большинстве\s+случаев\b", ["UNA", "VER"],
     "оговорка «как правило/обычно» — неясно, когда требование не действует", "major"),
    (r"\bсоответствующ\w*|\bнадлежащ\w*", ["UNA", "COM"], "«соответствующий/надлежащий» — не указано, чему именно", "minor"),
    # --- нечёткая обязательность ---
    (r"\bжелательно\b|\bрекомендуется\b|\bпо\s+желанию\b|\bпредпочтительно\b", ["UNA", "CNF"],
     "нечёткая обязательность: «желательно/рекомендуется» вместо «должен»", "minor"),
    (r"\bможет\b|\bмогут\b", ["UNA"], "«может» — неясно, обязательна ли возможность", "minor"),
    # --- незавершённость ---
    (r"\bTBD\b|\bTBC\b|\bTODO\b|\bуточн(ить|яется|ится)\b|\bбудет\s+определ\w*|\bбудут\s+определ\w*|\?\?|\bXXX\b|\bN/A\b|\bн/д\b",
     ["COM"], "незаполненное место (TBD/«уточнить») — требование неполно", "critical"),
    # --- абсолюты ---
    (r"\b100\s?%|\bвсегда\b|\bникогда\b|\bлюб(ой|ая|ое|ые|ого|ых)\b|\bвсе(х|м|ми)?\b\s+(пользовател|данн|запрос|случа|операц)\w*",
     ["FEA", "VER"], "абсолютная формулировка — проверить реалистичность и проверяемость", "minor"),
    # --- английские размытые слова ---
    (r"\b(user[- ]friendly|fast|quickly|easy|easily|efficient|adequate|appropriate|flexible|robust|"
     r"as appropriate|if possible|as needed|etc\.?|and/or|seamless|intuitive)\b",
     ["UNA", "VER"], "размытая формулировка без измеримого критерия", "major"),
]

# Проектные решения / технологии (возможное нарушение «уместности»)
DESIGN_RE = re.compile(
    r"(?<![\w-])(MySQL|PostgreSQL|Postgres|Oracle|MS\s?SQL|SQL\s?Server|MongoDB|Redis|Kafka|RabbitMQ|"
    r"Elasticsearch|ClickHouse|React|Angular|Vue(\.js)?|jQuery|Bootstrap|Java|Kotlin|Swift|Python|PHP|Go(lang)?|"
    r"C\#|\.NET|Node\.?js|Spring|Django|Flask|Laravel|Docker|Kubernetes|K8s|AWS|Azure|GCP|Firebase|"
    r"микросервис\w*|монолит\w*|REST(ful)?|SOAP|GraphQL|gRPC|JSON|XML|кнопк\w*|выпадающ\w*\s+спис\w*|"
    r"чекбокс\w*|всплывающ\w*\s+окн\w*|модальн\w*\s+окн\w*|красн\w*\s+цвет\w*|зел[её]н\w*\s+цвет\w*)(?![\w-])", _I)

MODAL_RE = re.compile(r"\b(должен|должна|должно|должны|обязан\w*|необходимо|shall|must)\b", _I)

ID_RE = re.compile(r"^([A-Za-zА-Яа-яЁё]+(?:[-_.][A-Za-zА-Яа-яЁё]+)*)[-_.]?(\d+(?:\.\d+)*)$")

_compiled = [(re.compile(p, _I), c, e, s) for p, c, e, s in WORD_RULES]


# ограничение по времени: «в течение минуты», «не более 2 секунд», «за 500 мс», «within 5 seconds»
TIME_RE = re.compile(
    r"\b(в\s+течение|не\s+более|не\s+дольше|до|за|within|in\s+under|under)\s+"
    r"(\d+[\s,.]?\d*\s*)?(мс|миллисекунд\w*|сек\w*|с\b|минут\w*|мин\b|час\w*|ms|millisec\w*|sec\w*|minutes?|hours?)", _I)
START_RE = re.compile(r"(после|с\s+момента|от\s+момента|после\s+получения|after|from)\s+[\w«»\-]+(\s+[\w«»\-]+)?", _I)
PERCENTILE_RE = re.compile(r"перцентил\w*|процентил\w*|\bp\d{2}\b|\d{2}(?:[.,]\d+)?\s*%|percentile|"
                           r"в\s+\d+\s+из\s+\d+|при\s+нагрузке|под\s+нагрузкой|\bRPS\b|\bTPS\b", _I)


def _context(text: str, start: int, end: int, words: int = 3) -> str:
    """Дословный фрагмент вокруг совпадения (±N слов) — для цитаты."""
    left = text[:start].split(" ")
    right = text[end:].split(" ")
    lt = " ".join(left[-(words + 1):]) if start > 0 else ""
    rt = " ".join(right[: words + 1]) if end < len(text) else ""
    return (lt + text[start:end] + rt).strip(" ,;:")


def _rec(crits: list[str], sev: str) -> str:
    if crits[0] == "COM" and sev == "critical":
        return "Заполнить пропуск: описать конкретное поведение/значение и согласовать его с владельцем требования."
    if crits[0] == "UNA" and "CNF" in crits:
        return "Для обязательного требования использовать «должен»; опциональное — явно пометить и указать приоритет."
    if crits == ["UNA"]:
        return "Уточнить формулировку так, чтобы она допускала одно толкование (перечислить конкретные значения/условия)."
    return "Заменить на измеримую формулировку: конкретное значение, единицы, условия и способ проверки."


def check_requirement(text: str) -> list[dict]:
    """Правила для одного требования. Возвращает список замечаний."""
    hits, seen = [], set()
    for rx, crits, expl, sev in _compiled:
        for m in rx.finditer(text):
            key = (m.group(0).lower(), crits[0])
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "criterion": crits[0], "criteria": crits, "severity": sev,
                "trigger": m.group(0), "quote": _context(text, m.start(), m.end()),
                "problem": f"Найдено «{m.group(0)}»: {expl}.",
                "recommendation": _rec(crits, sev),
            })
    for m in DESIGN_RE.finditer(text):
        key = (m.group(0).lower(), "APP")
        if key in seen:
            continue
        seen.add(key)
        hits.append({
            "criterion": "APP", "criteria": ["APP"], "severity": "minor", "trigger": m.group(0),
            "quote": _context(text, m.start(), m.end()),
            "problem": f"Упоминается конкретное проектное решение/технология «{m.group(0)}» — "
                       "возможно, это лишнее ограничение реализации.",
            "recommendation": "Если это не обоснованное внешнее ограничение, сформулировать требуемое свойство, а не способ реализации; "
                              "иначе указать источник ограничения.",
        })
    for m in TIME_RE.finditer(text):
        if PERCENTILE_RE.search(text):
            break
        start = START_RE.search(text)
        missing = "доли событий (перцентиля) и условий нагрузки" + ("" if start else ", а также точки начала отсчёта")
        hits.append({
            "criterion": "VER", "criteria": ["VER", "UNA"], "severity": "major", "trigger": m.group(0), "tag": "NFR-METRIC",
            "quote": _context(text, m.start(), m.end()),
            "problem": f"Ограничение по времени «{m.group(0)}» задано без {missing}"
                       + (f"; точка начала указана («{start.group(0)}»)" if start else "")
                       + " — неясно, при каком результате замера требование выполнено.",
            "recommendation": "Указать согласованную долю событий (перцентиль), условия нагрузки и способ замера"
                              + ("" if start else ", а также момент начала отсчёта") +
                              "; значения — решение владельца продукта (не выдумывать).",
        })
        break
    modals = MODAL_RE.findall(text)
    if len(modals) >= 2:
        m2 = list(MODAL_RE.finditer(text))[1]
        hits.append({
            "criterion": "SIN", "criteria": ["SIN"], "severity": "minor", "trigger": m2.group(0),
            "quote": _context(text, m2.start(), m2.end()),
            "problem": f"В одной формулировке {len(modals)} модальных глагола обязательности — вероятно, "
                       "объединено несколько требований.",
            "recommendation": "Разделить на отдельные атомарные требования с собственными идентификаторами.",
        })
    return hits


def check_set(reqs: list[dict]) -> list[dict]:
    """Правила для набора: дубли и пропуски идентификаторов, почти одинаковые формулировки,
    отсутствие идентификаторов."""
    out = []
    # дубликаты ID
    by_id: dict[str, list[dict]] = {}
    for r in reqs:
        if r.get("doc_id"):
            by_id.setdefault(r["doc_id"].upper(), []).append(r)
    for did, rs in by_id.items():
        if len(rs) > 1:
            out.append({
                "type": "Трассируемость", "criterion": "S-TRC", "severity": "major",
                "requirement_ids": [r["id"] for r in rs],
                "quotes": [], "problem": f"Идентификатор {did} используется {len(rs)} раза — требования неразличимы при трассировке.",
                "recommendation": "Присвоить уникальные идентификаторы.",
            })
    # пропуски в нумерации
    groups: dict[str, list[tuple[int, dict]]] = {}
    for r in reqs:
        m = ID_RE.match((r.get("doc_id") or "").strip())
        if m and "." not in m.group(2):
            groups.setdefault(m.group(1).upper(), []).append((int(m.group(2)), r))
    for prefix, items in groups.items():
        nums = sorted({n for n, _ in items})
        if len(nums) < 3:
            continue
        missing = [n for n in range(nums[0], nums[-1] + 1) if n not in nums]
        if missing and len(missing) <= 20:
            after = {n: r for n, r in items}
            prev = max(n for n in nums if n < missing[0])
            out.append({
                "type": "Пропуск", "criterion": "S-COMPL", "severity": "major",
                "requirement_ids": [after[prev]["id"]], "quotes": [],
                "problem": f"В нумерации требований {prefix} пропущены номера: "
                           + ", ".join(f"{prefix}-{n}" for n in missing)
                           + " — возможно, требования удалены или потеряны.",
                "recommendation": "Проверить, не утеряны ли требования; если удалены — отметить как «удалено».",
            })
    # почти дубли
    normed = [(r, norm(r["text"])) for r in reqs]
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            a, b = normed[i][1], normed[j][1]
            if min(len(a), len(b)) < 25 or abs(len(a) - len(b)) > 0.3 * max(len(a), len(b)):
                continue
            ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if ratio >= 0.88:
                out.append({
                    "type": "Дублирование", "criterion": "S-CONS", "severity": "major" if ratio < 0.99 else "minor",
                    "requirement_ids": [normed[i][0]["id"], normed[j][0]["id"]], "quotes": [],
                    "problem": f"Формулировки почти совпадают (сходство {ratio:.0%}). Если они различаются в деталях — "
                               "это потенциальное противоречие; если нет — дублирование.",
                    "recommendation": "Объединить в одно требование или явно описать различие.",
                })
    # идентификаторы
    with_id = sum(1 for r in reqs if r.get("doc_id"))
    if reqs and with_id == 0:
        out.append({
            "type": "Трассируемость", "criterion": "S-TRC", "severity": "major",
            "requirement_ids": [reqs[0]["id"]], "quotes": [],
            "problem": "Ни у одного требования в документе нет собственного идентификатора; "
                       "агент присвоил номера R-xxx самостоятельно.",
            "recommendation": "Присвоить каждому требованию уникальный устойчивый идентификатор.",
        })
    elif 0 < with_id < len(reqs):
        no_id = [r["id"] for r in reqs if not r.get("doc_id")]
        out.append({
            "type": "Трассируемость", "criterion": "S-TRC", "severity": "minor",
            "requirement_ids": no_id, "quotes": [],
            "problem": f"{len(no_id)} из {len(reqs)} требований не имеют идентификатора в документе.",
            "recommendation": "Присвоить идентификаторы всем требованиям.",
        })
    return out


# ====================================================================== проверки документа целиком
def _ctx_line(text: str, m: re.Match, words: int = 4) -> str:
    return _context(text, m.start(), m.end(), words)


_NAME_COL = re.compile(r"^(parameter|param|field|name|attribute|параметр|поле|атрибут|наименование|имя)\w*$", _I)
_TYPE_COL = re.compile(r"^(format|type|data\s*type|формат|тип(\s+данных)?)$", _I)


def _cells(text: str) -> list[str] | None:
    t = text.strip()
    if not (t.startswith("|") and t.endswith("|")) or t.count("|") < 3:
        return None
    return [c.strip().strip("`*").strip() for c in t[1:-1].split("|")]


def _param_rows(lines) -> list[tuple[str, str, object]]:
    """Строки таблиц, у которых есть колонка имени поля и колонка типа/формата:
    (имя поля, нормализованный тип, строка)."""
    out, cols = [], None
    for ln in lines:
        cells = _cells(ln.text)
        if cells is None:
            cols = None
            continue
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        ni = next((i for i, c in enumerate(cells) if _NAME_COL.match(c)), None)
        ti = next((i for i, c in enumerate(cells) if _TYPE_COL.match(c)), None)
        if ni is not None and ti is not None:
            cols = (ni, ti)
            continue
        if cols and len(cells) > max(cols):
            name, fmt = cells[cols[0]], cells[cols[1]].lower()
            if name and fmt and fmt not in {"—", "-"} and len(name) <= 64:
                out.append((name, fmt, ln))
    return out


_RESP_HEAD = re.compile(r"ответ|response|статус|status|\b[2-5]\d\d\b", re.I)
_REQ_HEAD = re.compile(r"запрос|request|query|path|body|параметр|parameters", re.I)


def _direction_of(text: str) -> str | None:
    t = text.lstrip("#").strip()
    if _RESP_HEAD.search(t):
        return "resp"
    if _REQ_HEAD.search(t):
        return "req"
    return None


def _schema_kind(fmt: str) -> str | None:
    f = fmt.lower()
    if re.search(r"\b(int|integer|long|number|numeric|decimal|float|double|число|целое)\b", f):
        return "number"
    if re.search(r"\b(string|str|text|строка)\b", f):
        return "string"
    if re.search(r"\b(bool|boolean|логическ\w*)\b", f):
        return "boolean"
    return None


def _json_kind(v) -> str | None:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    return None


def _schema_vs_examples(lines):
    """По направлениям (запрос / ответ): поля таблиц-схем и JSON-примеры, найденные под заголовками этого
    направления. Направление задаёт ближайший заголовок («Ответ», «Статусы», «200 …» — ответ; «Запрос»,
    «Query-параметры», «Request body» — запрос)."""
    direction, level = None, 0
    tagged = []
    for ln in lines:
        if ln.is_heading:
            lvl = len(ln.text) - len(ln.text.lstrip("#")) or 3
            d = _direction_of(ln.text)
            if d:
                direction, level = d, lvl
            elif lvl <= level:
                direction = None
        tagged.append((ln, direction))
    out = []
    for d in ("req", "resp"):
        sub = [ln for ln, x in tagged if x == d]
        rows = {}
        for name, fmt, ln in _param_rows(sub):
            rows.setdefault(name.strip("`"), (fmt, ln))
        examples = []
        for ln in sub:
            t = ln.text.strip()
            if t.startswith("{") and t.endswith("}"):
                try:
                    obj = json.loads(t)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    examples.append((ln, obj))
        if rows and examples:
            out.append((d, rows, examples))
    return out


def check_document(doc) -> list[dict]:
    """Детерминированные проверки по всему тексту документа (единицы измерения, HTTP-семантика,
    заглушки шаблона). Возвращает замечания к набору с дословными цитатами и строками."""
    lines = [ln for ln in doc.lines if ln.text.strip()]
    out: list[dict] = []

    def find(rx, flags=_I):
        r = re.compile(rx, flags)
        return [(ln, m) for ln in lines for m in [r.search(ln.text)] if m]

    # 1. Единицы измерения: тиыны и тенге без правила конвертации
    # общий случай: дробная денежная единица (тиын, копейка, цент) и основная (тенге, рубль, доллар…)
    tiyn = find(r"\b(тиын|копе[йе]к|копейк|цент|cents?|minor\s+units?)\w*")
    tenge = find(r"\b(тенге|рубл|доллар|евро|сом|сум|dollars?|tenge|rubles?)\w*|[₸₽$€]")
    conv = find(r"конверт\w*|пересч[её]т\w*\s+(в|из)\s+(тиын|копе|цент|тенге|рубл)|=\s*100\s*(тиын|копе|цент)|"
                r"×\s?100|\*\s?100|/\s?100\b|\b100\s+(тиын|копе|цент|cents)")
    if tiyn and tenge and not conv:
        a, b = tiyn[0], next((t for t in tenge if t[0].gid != tiyn[0][0].gid), tenge[0])
        out.append({
            "type": "Несогласованность", "criterion": "S-CONS", "severity": "major", "tag": "UNITS",
            "locs": [(a[0].gid, _ctx_line(a[0].text, a[1])), (b[0].gid, _ctx_line(b[0].text, b[1]))],
            "problem": f"Денежные суммы задаются в разных единицах: «{a[1].group(0)}» (дробная единица) и "
                       f"«{b[1].group(0)}» (основная единица); правило конвертации и округления в документе не описано. "
                       "Риск ошибки в 100 раз при сравнении и передаче сумм между системами.",
            "recommendation": "Зафиксировать единую единицу хранения и передачи сумм (например, целое число в "
                              "дробных единицах) и явно описать конвертацию и округление на границах API/UI.",
        })

    # 2. HTTP GET для операции, изменяющей состояние
    get_ = find(r"\bGET\b|curl\s+-G\b", 0)
    mut = find(r"/[\w/]*(save|update|create|delete|set|add|remove)\w*|\bсохранени\w*|\bсохраня\w*|\bзаменить\b|\bизменени\w*")
    if get_ and mut:
        g = get_[0]
        mm = next((x for x in mut if "/" in x[1].group(0)), mut[0])
        out.append({
            "type": "Нарушение стандарта", "criterion": "S-FEAS", "severity": "critical", "tag": "REST-METHOD",
            "locs": [(g[0].gid, _ctx_line(g[0].text, g[1])), (mm[0].gid, _ctx_line(mm[0].text, mm[1]))],
            "problem": "Для операции, изменяющей состояние (сохранение/замена настройки), указан HTTP-метод GET. "
                       "По RFC 9110 (§9.2.1) GET — безопасный метод без побочных эффектов: запрос могут повторить "
                       "или закешировать прокси/браузер, параметры попадают в логи и историю.",
            "recommendation": "Использовать POST (создание/сохранение) или PUT (идемпотентная замена) и передавать "
                              "параметры в теле запроса (application/json).",
        })

    # 3. Ошибка под кодом 200
    cand = [i for i, ln in enumerate(lines) if re.search(r"\b200\b", ln.text) and re.search(r"success|успеш", ln.text, _I)]
    cand.sort(key=lambda i: (not lines[i].text.lstrip().startswith("#"), i))
    for i in cand:
        ln = lines[i]
        window = lines[i:i + 10]
        err = next(((w, m) for w in window for m in [re.search(r'"result"\s*:\s*"ERROR"', w.text)] if m), None)
        if err is not None:
            if err:
                out.append({
                    "type": "Нарушение стандарта", "criterion": "S-FEAS", "severity": "critical", "tag": "STATUS-200",
                    "locs": [(ln.gid, collapse(ln.text)), (err[0].gid, _ctx_line(err[0].text, err[1]))],
                    "problem": "В разделе успешного ответа 200 приведён пример с результатом ERROR: ошибка бизнес-логики "
                               "маскируется под успешный HTTP-код, клиент и мониторинг не отличат успех от отказа.",
                    "recommendation": "Для ошибок валидации возвращать 400 Bad Request (или 422) с телом ошибки; "
                                      "код 200/201 — только для успешного сохранения.",
                })
                break

    # 4. Заглушки и незаполненные места шаблона
    ph = find(r"заполня\w+\s+\w+|\bTBD\b|\bTBC\b|\bTODO\b|буд(ет|ут)\s+(определ|уточн|описан|добавлен)\w*|"
              r"(уточня|определя|описыва)\w*\s+позже|раздел\s+в\s+разработке")
    for ln, m in ph[:3]:
        out.append({
            "type": "Незаполненный раздел", "criterion": "S-COMPL", "severity": "major", "tag": "SECTION-EMPTY",
            "locs": [(ln.gid, _ctx_line(ln.text, m))],
            "problem": f"Раздел не заполнен — вместо содержания заглушка «{m.group(0)}» "
                       f"(раздел «{doc.section_of(ln.gid) or '—'}»).",
            "recommendation": "Заполнить раздел до передачи в разработку (для тестирования — тестовые сценарии, "
                              "включая негативные и граничные).",
        })
    yn = [ln for ln in lines if re.search(r"Да\s*/\s*Нет", ln.text)]
    if len(yn) >= 3:
        out.append({
            "type": "Незаполненный раздел", "criterion": "S-COMPL", "severity": "minor", "tag": "PLACEHOLDER",
            "locs": [(yn[0].gid, collapse(yn[0].text))],
            "problem": f"Чек-лист ревью не заполнен: в {len(yn)} строках оставлены варианты «Да / Нет» без отметки — "
                       "ревью требований формально не проведено.",
            "recommendation": "Провести ревью и проставить отметки «Да»/«Нет»/«Не требуется» с примечаниями.",
        })
    empty_rows = [ln for ln in lines if re.fullmatch(r"\|(\s*\|)+\s*", ln.text.strip())]
    if empty_rows:
        out.append({
            "type": "Незаполненный раздел", "criterion": "S-COMPL", "severity": "minor", "tag": "APPROVALS",
            "locs": [(empty_rows[0].gid, empty_rows[0].text.strip())],
            "problem": f"Пустая строка таблицы (раздел «{doc.section_of(empty_rows[0].gid) or '—'}»): "
                       "не указаны подразделения/ответственные.",
            "recommendation": "Заполнить таблицу (например, блок согласующих: подразделение, ФИО, подпись/дата).",
        })
    dash = find(r"(Prod|Swagger|URL)[^|]*\|\s*—\s*\||Prod:\s*—")
    for ln, m in dash[:2]:
        out.append({
            "type": "Незаполненный раздел", "criterion": "S-COMPL", "severity": "minor", "tag": "PLACEHOLDER",
            "locs": [(ln.gid, collapse(ln.text))],
            "problem": "Параметр спецификации не заполнен (стоит «—»).",
            "recommendation": "Указать значение или явно отметить «не применяется» с причиной.",
        })
    ver = find(r"верси\w*[^.]*не\s+заполн\w*|не\s+заполн\w*[^.]*верси\w*")
    if ver:
        ln, m = ver[0]
        out.append({
            "type": "Версионность", "criterion": "S-MOD", "severity": "minor", "tag": "VERSIONING",
            "locs": [(ln.gid, _ctx_line(ln.text, m, 2))],
            "problem": "Версия и дата документа не заполнены — нельзя однозначно сослаться на согласованную редакцию.",
            "recommendation": "Указать версию, дату и историю изменений документа.",
        })

    # 4б. Структура: пункты с собственной меткой под заголовком с другой меткой
    by_head: dict[str, list] = {}
    for gid, item, head, parent in getattr(doc, "misplaced", []):
        by_head.setdefault(head, []).append((gid, item))
    for head, items in by_head.items():
        ln0 = next(ln for ln in lines if ln.gid == items[0][0])
        out.append({
            "type": "Структура", "criterion": "S-COMP", "severity": "minor", "tag": "STRUCTURE",
            "locs": [(g, collapse(next(ln.text for ln in lines if ln.gid == g))) for g, _ in items],
            "problem": (f"Пункты {', '.join('«' + i + '»' for _, i in items)} размещены" if len(items) > 1
                        else f"Пункт «{items[0][1]}» размещён") + f" под заголовком «{head}», "
                       "который относится к другому пункту, — читатель может отнести их описание не к тому разделу.",
            "recommendation": "Вынести каждый пункт под собственный заголовок или в общий раздел-перечень.",
        })
        del ln0

    # 4в. Схема и пример ОДНОГО И ТОГО ЖЕ сообщения (запрос или ответ) дают полю разные типы.
    # Разные типы в запросе и ответе сами по себе не противоречие: направления сравниваются только между собой.
    for direction, rows, examples in _schema_vs_examples(lines):
        bad = []
        for name, (fmt, row) in rows.items():
            for ex_ln, obj in examples:
                if name in obj:
                    got = _json_kind(obj[name])
                    want = _schema_kind(fmt)
                    if got and want and got != want:
                        bad.append((name, fmt, row, ex_ln, obj[name]))
                        break
        if bad:
            locs, parts = [], []
            for name, fmt, row, ex_ln, val in bad[:4]:
                locs += [(row.gid, collapse(row.text)), (ex_ln.gid, collapse(ex_ln.text))]
                parts.append(f"«{name}»: в таблице {fmt}, в примере {json.dumps(val, ensure_ascii=False)}")
            what = "ответа" if direction == "resp" else "запроса"
            out.append({
                "type": "Несогласованность", "criterion": "S-CONS", "severity": "major", "tag": "DATA-TYPE",
                "locs": locs,
                "problem": f"Схема {what} и пример того же {what} задают полю разный тип (" + "; ".join(parts) +
                           "). Клиент, написанный по схеме, не разберёт ответ из примера (или наоборот).",
                "recommendation": f"Привести схему и пример {what} к одному типу для каждого поля; какой тип верный — "
                                  "решает владелец API.",
            })

    # 5. Заголовок Content-Type при отсутствии тела
    ct = find(r"Content-Type[^a-z]*application/json")
    nobody = find(r"Тело\s+не\s+использ\w*|body\s+is\s+not\s+used")
    if ct and nobody:
        out.append({
            "type": "Несогласованность", "criterion": "S-CONS", "severity": "minor", "tag": "FORMAT",
            "locs": [(ct[0][0].gid, _ctx_line(ct[0][0].text, ct[0][1])), (nobody[0][0].gid, _ctx_line(nobody[0][0].text, nobody[0][1]))],
            "problem": "Указан Content-Type application/json, но тело запроса не используется — заголовок вводит в заблуждение.",
            "recommendation": "Либо передавать параметры в JSON-теле (вместе с переходом на POST/PUT), либо убрать требование к Content-Type.",
        })
    return out


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# «Почему это критично» для критичных дефектов автоматических правил (общие формулировки, без привязки к документу)
WHY_CRITICAL = {
    "REST-METHOD": {
        "trigger": "Клиент вызывает изменяющую операцию методом GET; запрос проходит через кэширующие прокси, "
                   "журналы доступа или автоматический повтор.",
        "development": "Нельзя одновременно соблюсти документ и HTTP-семантику (RFC 9110): реализация либо "
                       "нарушает спецификацию, либо стандарт; блокирован контракт этого метода API.",
        "testing": "Нет однозначного ожидаемого поведения для повторного/кэшированного GET-запроса.",
        "business": "Повтор безопасного по стандарту запроса может повторно изменить данные; параметры попадают в URL "
                    "и журналы.",
        "scenario": "Сохранение настроек GET-запросом → прокси/клиент повторяет запрос → операция выполняется "
                    "дважды или ответ берётся из кэша без сохранения.",
        "resolution": "Выбрать метод, соответствующий операции (POST/PUT/PATCH), и обновить описание, пример и логику.",
        "closure": "Во всех местах документа (описание, пример запроса, логика) указан один и тот же метод, "
                   "соответствующий изменяющей операции.",
    },
    "STATUS-200": {
        "trigger": "Сервис возвращает ошибку с HTTP-кодом 200.",
        "development": "Клиент и мониторинг не могут определить ошибку по коду ответа; обработка ошибок "
                       "блокирована до решения о контракте.",
        "testing": "Тест на ошибочный сценарий не имеет однозначного ожидаемого кода ответа.",
        "business": "Ошибки маскируются как успешные ответы: клиент может считать операцию выполненной.",
        "scenario": "Ошибка валидации → ответ 200 с телом ошибки → клиент, проверяющий только код, показывает успех.",
        "resolution": "Разделить успешный ответ и ошибки по кодам 4xx/5xx и описать тело ошибки.",
        "closure": "Раздел кода 200 содержит только успешный ответ; для каждой ошибки указан свой код и пример.",
    },
}
