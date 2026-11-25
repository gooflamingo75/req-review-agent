"""Детерминированные (без LLM) проверки: словарь размытых слов, маркеры TBD,
проектные решения, неатомарность, дубли и пропуски идентификаторов.

Результаты используются как подсказки для модели и как самостоятельные
замечания, если модель их не учла.
"""
from __future__ import annotations

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
            "requirement_ids": no_id[:15], "quotes": [],
            "problem": f"{len(no_id)} из {len(reqs)} требований не имеют идентификатора в документе.",
            "recommendation": "Присвоить идентификаторы всем требованиям.",
        })
    return out
