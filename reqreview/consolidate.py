"""Консолидация дефектов по первопричине.

Один и тот же дефект (например, «GET для операции сохранения») модель и правила могут найти в нескольких
требованиях и в обоих проходах проверки набора. В отчёте он должен быть ОДНИМ дефектом со списком всех
мест, где проявляется. Модуль:
  1) объединяет дубли и замечания с одной первопричиной среди замечаний к набору;
  2) «поднимает» системные дефекты отдельных требований в соответствующий дефект набора
     (или создаёт дефект набора, если одна первопричина встречается в нескольких требованиях);
  3) убирает дубли внутри одного требования.
Счётчики отчёта считаются по уникальным дефектам; число мест показывается отдельно.
"""
from __future__ import annotations

import re

from .criteria import SEVERITY_ORDER

# Теги дефектов, которые по своей природе системные: одна первопричина — много мест
SYSTEMIC_TAGS = {
    "REST-METHOD", "STATUS-200", "UNITS", "DATA-TYPE", "SECTION-EMPTY", "PLACEHOLDER",
    "VERSIONING", "APPROVALS", "PRIORITY", "STRUCTURE", "ACCEPTANCE", "DESIGN-DETAIL",
}
TAG_TITLES = {
    "DESIGN-DETAIL": "Детали реализации в требованиях",
    "REST-METHOD": "HTTP-метод не соответствует характеру операции",
    "STATUS-200": "Ошибка возвращается под успешным HTTP-кодом",
    "UNITS": "Несогласованные единицы измерения",
    "DATA-TYPE": "Несогласованные типы данных",
    "FORMAT": "Несоответствие формы/формата",
    "SECTION-EMPTY": "Незаполненный раздел",
    "PLACEHOLDER": "Незаполненные поля шаблона",
    "VERSIONING": "Нет версионирования",
    "APPROVALS": "Не заполнен блок согласования",
    "PRIORITY": "Нет приоритетов требований",
    "STRUCTURE": "Нарушена структура документа",
    "ACCEPTANCE": "Нет критериев приёмки",
    "TRACEABILITY": "Нарушена прослеживаемость",
}
# Взаимно близкие теги (модель называет одну проблему по-разному)
TAG_ALIASES = {"STATUS-CODE-200": "STATUS-200", "HTTP-METHOD": "REST-METHOD", "METHOD": "REST-METHOD",
               "EMPTY-SECTION": "SECTION-EMPTY", "TBD": "PLACEHOLDER", "VERSION": "VERSIONING",
               "PRIORITIES": "PRIORITY", "TYPES": "DATA-TYPE", "TYPE": "DATA-TYPE"}

_WORD = re.compile(r"[a-zа-яё0-9]{4,}", re.I)


def _stems(s: str) -> set[str]:
    return {w.lower().replace("ё", "е")[:6] for w in _WORD.findall(s or "")}


def similarity(a: str, b: str) -> float:
    A, B = _stems(a), _stems(b)
    if not A or not B:
        return 0.0
    return len(A & B) / min(len(A), len(B))


def _tag(f) -> str:
    return TAG_ALIASES.get(f.tag, f.tag)


def _note_also(target, problem: str) -> None:
    """Сохраняет формулировку, если она описывает ДРУГОЙ аспект/место той же первопричины
    (пересказы того же самого не сохраняются — иначе отчёт снова наполнится повторами)."""
    if target.tag in SYSTEMIC_TAGS and target.tag not in ("PLACEHOLDER", "SECTION-EMPTY"):
        return                       # одна системная первопричина: все места уже перечислены в «Где»
    limit = 0.5 if target.tag in ("PLACEHOLDER", "SECTION-EMPTY") else 0.3
    if not problem or similarity(problem, target.problem) >= limit:
        return
    if any(similarity(problem, a) >= limit for a in target.also):
        return
    target.also.append(problem)


def _add_occurrence(target, f, adopt: bool = False) -> None:
    """Добавляет места и цитаты замечания f в target (без повторов строк)."""
    seen = {a for a, _ in target.locs}
    pairs = list(zip(f.locs, f.quotes)) if f.locs and len(f.locs) == len(f.quotes) else []
    if not pairs and f.g1 is not None:
        q = f.quote.strip("«»") if f.quote else ""
        pairs = [((f.g1, f.g2 if f.g2 is not None else f.g1), q)]
    for (a, b), q in pairs:
        if a in seen:
            continue
        seen.add(a)
        target.locs.append((a, b))
        target.quotes.append(q)
    for rid in f.req_ids:
        if rid not in target.req_ids:
            target.req_ids.append(rid)
    if SEVERITY_ORDER[f.severity] < SEVERITY_ORDER[target.severity]:
        target.severity = f.severity
    # при слиянии замечаний одного уровня берём самое содержательное описание (предпочтительно — от модели);
    # отличающиеся описания сохраняем как «также проявляется как»
    if adopt and f.source == "LLM" and (target.source != "LLM" or len(f.problem) > len(target.problem) * 1.3):
        old = target.problem
        target.problem, target.source = f.problem, "LLM"
        if f.recommendation:
            target.recommendation = f.recommendation
        _note_also(target, old)
    else:
        _note_also(target, f.problem)
        if not target.recommendation and f.recommendation:
            target.recommendation = f.recommendation
    target.merged += 1 + getattr(f, "merged", 0)


def _normalize_set(f) -> None:
    """Приводит замечание к виду «список мест + цитаты»."""
    if not f.locs and f.g1 is not None:
        f.locs = [(f.g1, f.g2 if f.g2 is not None else f.g1)]
    if len(f.quotes) != len(f.locs):
        q = f.quote.strip("«»") if f.quote and len(f.locs) == 1 else ""
        f.quotes = (f.quotes + [q] * len(f.locs))[: len(f.locs)]


# «Оформительские» теги: одна метка, но разные места документа — это разные дефекты
LOCAL_TAGS = {"PLACEHOLDER", "SECTION-EMPTY", "STRUCTURE", "APPROVALS"}
_SECTION_OF = None


def _sections(f) -> set:
    if _SECTION_OF is None:
        return set()
    return {_SECTION_OF(a) for a, _ in f.locs}


def _same_root(a, b) -> bool:
    ta, tb = _tag(a), _tag(b)
    shared_req = bool(set(a.req_ids) & set(b.req_ids))
    shared_loc = bool({x for x, _ in a.locs} & {x for x, _ in b.locs})
    sim = similarity(a.problem, b.problem)
    if ta and ta == tb and ta in SYSTEMIC_TAGS:
        if ta not in LOCAL_TAGS:
            return True
        # один и тот же «оформительский» тег объединяем только в пределах одного раздела
        sa, sb = _sections(a), _sections(b)
        return shared_loc or bool(sa & sb) if (sa and sb) else (shared_loc or sim >= 0.5)
    # противоречия между РАЗНЫМИ парами требований — разные дефекты, даже если описаны похожими словами
    if "CONTRADICTION" in (ta, tb) or a.criterion == b.criterion == "S-CONS":
        A, B = set(a.req_ids), set(b.req_ids)
        if A and B and not (A <= B or B <= A):
            return False
    # одна и та же строка-источник и близкое описание — одна первопричина, даже если тег разный
    # (например, «GET для сохранения» и «противоречие GET и изменения состояния»)
    if shared_loc and sim >= 0.4:
        return True
    if ta and ta == tb and (shared_req or shared_loc) and sim >= 0.35:
        return True
    if a.criterion == b.criterion and (shared_req or shared_loc) and sim >= 0.5:
        return True
    return sim >= 0.75


def _same_in_requirement(a, b, req_text: str) -> bool:
    """Два замечания к одному требованию описывают одну проблему: близкий текст, либо один и тот же
    процитированный фрагмент (не всё требование целиком) и хоть какая-то смысловая близость."""
    sim = similarity(a.problem + " " + a.recommendation, b.problem + " " + b.recommendation)
    if sim >= 0.45 or (a.criterion == b.criterion and sim >= 0.35):
        return True
    qa, qb = (a.quote or "").strip("«» "), (b.quote or "").strip("«» ")
    whole = (req_text or "").strip()
    if a.verified and b.verified and qa and qb and qa != whole and qb != whole:
        short_, long_ = sorted((qa.lower(), qb.lower()), key=len)
        if len(short_) >= 8 and short_ in long_ and sim >= 0.15:
            return True
    return False


def consolidate(reqs, set_findings, finding_cls, section_of=None):
    """Возвращает (новый список замечаний к набору, статистика объединения).
    section_of(gid) — раздел документа для строки (чтобы не склеивать дефекты из разных разделов)."""
    global _SECTION_OF
    _SECTION_OF = section_of
    stats = {"before": sum(len(r.findings) for r in reqs) + len(set_findings)}
    for f in set_findings:
        f.merged = getattr(f, "merged", 0)
        f.also = getattr(f, "also", None) or []
        f.tag = _tag(f)
        _normalize_set(f)

    # 1) замечания к набору: дубли и одна первопричина
    merged: list = []
    for f in sorted(set_findings, key=lambda x: (SEVERITY_ORDER[x.severity], x.source != "LLM")):
        host = next((m for m in merged if _same_root(m, f)), None)
        if host is None:
            merged.append(f)
        else:
            _add_occurrence(host, f, adopt=True)

    # 2) системные дефекты отдельных требований
    by_tag: dict[str, list] = {}
    for r in reqs:
        r.linked = getattr(r, "linked", [])
        for f in r.findings:
            f.merged = getattr(f, "merged", 0)
            f.tag = _tag(f)
            if f.tag in SYSTEMIC_TAGS:
                by_tag.setdefault(f.tag, []).append((r, f))
    for tag, items in by_tag.items():
        host = next((m for m in merged if m.tag == tag), None)
        if host is None and len({r.id for r, _ in items}) < 2:
            continue                       # одиночный дефект — остаётся у требования
        if host is None:
            r0, f0 = items[0]
            host = finding_cls(criterion={"REST-METHOD": "S-FEAS", "STATUS-200": "S-FEAS", "UNITS": "S-CONS",
                                          "DATA-TYPE": "S-CONS", "FORMAT": "S-COMP", "PRIORITY": "S-RANK",
                                          "TRACEABILITY": "S-TRC", "VERSIONING": "S-MOD",
                                          "DESIGN-DETAIL": "S-MOD"}.get(tag, "S-COMPL"),
                               severity=f0.severity, problem=f0.problem, recommendation=f0.recommendation,
                               type=TAG_TITLES.get(tag, "Системный дефект"), tag=tag, source=f0.source)
            host.merged = -1
            host.also = []
            _add_occurrence(host, f0)
            items = items[1:]
            merged.append(host)
        for r, f in items:
            _add_occurrence(host, f)
        for r, f in by_tag[tag]:
            if f in r.findings:
                r.findings.remove(f)
            if host not in r.linked:
                r.linked.append(host)

    # 2b) несистемные дефекты требования, которые дублируют дефект набора (та же первопричина)
    for r in reqs:
        keep = []
        for f in r.findings:
            fl = {f.g1} if f.g1 is not None else set()
            host = next((m for m in merged if
                         (r.id in m.req_ids and similarity(f.problem, m.problem) >= 0.55) or
                         (fl & {a for a, _ in m.locs} and similarity(f.problem, m.problem) >= 0.35)), None)
            if host is not None:
                _add_occurrence(host, f)
                if host not in r.linked:
                    r.linked.append(host)
            else:
                keep.append(f)
        r.findings = keep

    # 3) одна первопричина внутри требования = один дефект с несколькими критериями (R01/R02)
    for r in reqs:
        uniq: list = []
        for f in sorted(r.findings, key=lambda x: (SEVERITY_ORDER[x.severity], not x.recommendation)):
            dup = next((u for u in uniq if _same_in_requirement(u, f, r.text)), None)
            if dup is None:
                uniq.append(f)
            else:
                dup.merged = getattr(dup, "merged", 0) + 1 + getattr(f, "merged", 0)
                dup.criteria = list(dict.fromkeys((getattr(dup, "criteria", None) or [dup.criterion])
                                                  + (getattr(f, "criteria", None) or [f.criterion])))
                dup.obs = list(dict.fromkeys((getattr(dup, "obs", None) or []) + (getattr(f, "obs", None) or [])))
                if not dup.recommendation and f.recommendation:
                    dup.recommendation = f.recommendation
                if similarity(f.problem, dup.problem) < 0.5 and f.problem not in dup.also:
                    dup.also.append(f.problem)
        r.findings = uniq

    for m in merged:
        m.locs, m.quotes = (list(x) for x in zip(*sorted(zip(m.locs, m.quotes)))) if m.locs else ([], [])
        if m.locs:
            m.g1, m.g2 = m.locs[0]
        m.quote = " … ".join(f"«{q}»" for q in m.quotes if q)
        m.verified = bool(m.quotes and any(m.quotes))
    merged.sort(key=lambda x: (SEVERITY_ORDER[x.severity], -len(x.locs)))
    stats["after"] = sum(len(r.findings) for r in reqs) + len(merged)
    stats["merged"] = stats["before"] - stats["after"]
    return merged, stats
