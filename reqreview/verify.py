"""Самопроверка результатов перед выпуском отчёта.

1. critic()        — LLM-ревьюер второго уровня: отсеивает ложные срабатывания (только с дословным
                     доказательством из документа), калибрует критичность и критерий, убирает выдуманные значения.
2. sanitize()      — детерминированно убирает из рекомендаций URL и пути API, которых нет в документе.
3. self_check()    — согласованность отчёта: у каждой отметки ❌/⚠️ есть дефект с местом; противоречие
                     отражено у обеих сторон; нет «согласовано» рядом с нарушением.
"""
from __future__ import annotations

import re

from .criteria import INDIVIDUAL, SET, SEVERITY
from .llm import LLMError, log
from .util import collapse_ws, norm, short

# Соответствие критериев набора критериям отдельного требования
SET_TO_IND = {"S-CONS": "CON", "S-FEAS": "FEA", "S-COMPL": "COM", "S-COMP": "UNA", "S-TRC": "CNF",
              "S-MOD": "CNF", "S-RANK": "NEC", "S-BND": "NEC"}
CRIT_TAGS = {"UNA": "AMBIGUITY", "COM": "INCOMPLETE", "CON": "CONTRADICTION", "VER": "UNTESTABLE",
             "FEA": "FEASIBILITY", "NEC": "NECESSITY", "APP": "DESIGN-DETAIL", "SIN": "NON-ATOMIC",
             "COR": "CORRECTNESS", "CNF": "FORMAT"}

_URL = re.compile(r"https?://[^\s`'\"»)\]]+")
_PATH = re.compile(r"(?<![\w.])/(?:[\w{}:\-]+/)+[\w{}:\-]+|(?<![\w.])/[\w{}\-]+(?=[`'\"»\s,.;)])")
_LINE_REF = re.compile(r"\[L(\d+)\]|\bстрок[аеиу]?\s+(\d+)", re.I)


# ------------------------------------------------------------------ 1. критик
def critic(doc, reqs, set_findings, llm, doc_text: str, prompts_mod, warnings: list[str], cov=None) -> list[dict]:
    """Возвращает список отклонённых дефектов (для приложения отчёта)."""
    items, index = [], {}
    for r in reqs:
        for f in r.findings:
            index[f.fid] = (r, f)
            items.append({"id": f.fid, "level": "требование", "requirement": r.id, "requirement_text": short(r.text, 300),
                          "criterion": f.criterion, "severity": f.severity, "quote": short(f.quote, 200),
                          "problem": f.problem, "recommendation": f.recommendation})
    for f in set_findings:
        index[f.fid] = (None, f)
        items.append({"id": f.fid, "level": "набор", "requirements": f.req_ids[:10], "criterion": f.criterion,
                      "severity": f.severity, "quotes": [short(q, 160) for q in f.quotes[:4]],
                      "problem": f.problem, "recommendation": f.recommendation})
    title = "Самопроверка: ревьюер второго уровня"
    if llm.model == "mock":
        if cov:
            cov.check("critic", title, "не выполнена", "демо-режим без модели")
        return []
    if not items:
        if cov:
            cov.check("critic", title, "выполнена", "дефектов для проверки нет")
        return []
    log(f"[4/5] Самопроверка: LLM-ревьюер проверяет {len(items)} дефектов на ложные срабатывания и критичность…")
    try:
        data = llm.generate_json("critic", prompts_mod.SYSTEM, prompts_mod.critic_prompt(doc.title, doc_text, items),
                                 {"items": items})
        if cov:
            cov.check("critic", title, "выполнена")
    except LLMError as e:
        warnings.append(f"Самопроверка LLM-ревьюером не выполнена: {e}.")
        if cov:
            cov.check("critic", title, "не выполнена", str(e)[:200])
        return []
    dropped, changed = [], 0
    for rv in (data.get("reviews") or []) if isinstance(data, dict) else []:
        if not isinstance(rv, dict) or rv.get("id") not in index:
            continue
        r, f = index[rv["id"]]
        verdict = str(rv.get("verdict", "keep")).lower()
        reason = collapse_ws(str(rv.get("reason") or ""))
        if verdict == "drop":
            ev = collapse_ws(str(rv.get("evidence_quote") or ""))
            hit = doc.find(ev) if ev else None
            # отклоняем только с дословным доказательством из документа; стандартные поля шаблона —
            # допускаем отклонение без цитаты, если дефект MINOR/MAJOR по критерию APP
            if hit or (f.criterion == "APP" and f.severity != "critical"):
                dropped.append({"id": f.fid, "where": (r.id if r else ", ".join(f.req_ids[:5])),
                                "problem": f.problem, "reason": reason,
                                "evidence": f"{doc.location(hit[1], hit[2])} — «{hit[0]}»" if hit else ""})
                f._dropped = True
                continue
            verdict = "rewrite" if rv.get("problem") else "keep"
        if verdict in ("rewrite", "severity"):
            sev = str(rv.get("severity") or "").lower()
            if sev in SEVERITY and sev != f.severity:
                f.severity = sev
                changed += 1
            c = str(rv.get("criterion") or "").upper()
            if c and ((r is not None and c in INDIVIDUAL) or (r is None and c in SET)) and c != f.criterion:
                f.criterion = c
                changed += 1
            if verdict == "rewrite":
                if rv.get("problem"):
                    f.problem = collapse_ws(str(rv["problem"]))
                if rv.get("recommendation"):
                    f.recommendation = collapse_ws(str(rv["recommendation"]))
                changed += 1
    for r in reqs:
        r.findings = [f for f in r.findings if not getattr(f, "_dropped", False)]
        r.linked = [f for f in r.linked if not getattr(f, "_dropped", False)]
    set_findings[:] = [f for f in set_findings if not getattr(f, "_dropped", False)]
    if dropped or changed:
        warnings.append(f"Самопроверка LLM-ревьюером: отклонено ложных срабатываний — {len(dropped)}, "
                        f"скорректировано (критичность/критерий/формулировка) — {changed}.")
    return dropped


# ------------------------------------------------------------------ 2. выдуманные значения
def sanitize(doc, findings) -> int:
    """Заменяет в рекомендациях URL и пути API, которых нет в документе, на «<указать …>»."""
    text = norm(doc._display)
    fixed = 0
    for f in findings:
        rec = f.recommendation or ""
        new = _URL.sub(lambda m: m.group(0) if norm(m.group(0)) in text else "<указать URL>", rec)
        new = _PATH.sub(lambda m: m.group(0) if norm(m.group(0)) in text else "<указать путь ресурса>", new)
        if new != rec:
            f.recommendation = new
            fixed += 1
    return fixed


# ------------------------------------------------------------------ 3. согласованность отчёта
def _req_at_line(reqs, line: int):
    return next((r for r in reqs if r.g1 <= line <= r.g2), None)


def self_check(doc, reqs, set_findings, finding_cls, final: bool) -> dict:
    """final=False — до критика (создаёт недостающие дефекты из комментариев модели);
    final=True — после критика (снимает отметки, оставшиеся без дефекта)."""
    fixes = {"created": 0, "cleared": 0, "linked": 0, "comments": 0}

    # противоречие должно быть видно у обеих сторон
    for r in reqs:
        for f in r.findings:
            if f.criterion not in ("CON", "COR"):
                continue
            for m in _LINE_REF.finditer(f.problem):
                line = int(m.group(1) or m.group(2))
                other = _req_at_line(reqs, line)
                if other is None or other is r:
                    continue
                rel = getattr(other, "related", [])
                if not any(x is f for x, _ in rel):
                    rel.append((f, r.id))
                    other.related = rel
                    fixes["linked"] += 1
                other.criteria["CON"] = "fail"
                other.comments["CON"] = f"Противоречит {r.id}" + (f" ({r.doc_id})" if r.doc_id else "") + \
                    f" — см. дефект {f.fid or 'у ' + r.id}."

    for r in reqs:
        covered = {f.criterion for f in r.findings}
        covered |= {SET_TO_IND.get(f.criterion) for f in r.linked}
        if getattr(r, "related", None):
            covered.add("CON")
        for c in INDIVIDUAL:
            st = r.criteria.get(c, "na")
            if st not in ("warn", "fail"):
                continue
            cm = r.comments.get(c, "")
            if c in covered:
                if re.search(r"огласован|без нарушени|соответству", cm, re.I):
                    r.comments[c] = ""
                    fixes["comments"] += 1
                continue
            if not final and cm and not re.search(r"огласован|без нарушени", cm, re.I):
                r.findings.append(finding_cls(
                    criterion=c, severity="minor" if st == "warn" else "major", problem=cm,
                    recommendation="", quote=r.text, g1=r.g1, g2=r.g2, verified=True, req_ids=[r.id],
                    tag=CRIT_TAGS.get(c, ""), source="LLM"))
                fixes["created"] += 1
            else:
                r.criteria[c] = "ok"
                r.comments.pop(c, None)
                fixes["cleared"] += 1
        # «согласовано» при выполненном критерии допустимо, при нарушении — нет (проверено выше)
    return fixes


def acceptance_gaps(reqs, trace, finding_cls):
    """Дефект набора: функциональные правила без критериев приёмки (по матрице прослеживаемости)."""
    by_id = {r.id: r for r in reqs}
    missing = []
    for t in trace:
        r = by_id.get(t["req"])
        if r is None or t.get("acceptance"):
            continue
        if any(k in (r.kind or "").lower() for k in ("критерий", "контекст")):
            continue
        missing.append(r)
    if not missing:
        return None
    quotes = [short(r.text, 90).rstrip("…") for r in missing]
    quotes = [q if q in r.text else r.text for q, r in zip(quotes, missing)]
    return finding_cls(
        criterion="S-COMPL", severity="major", type="Пропуск", tag="ACCEPTANCE", source="правило",
        problem=f"Для {len(missing)} правил нет ни одного критерия приёмки (по матрице прослеживаемости): "
                + ", ".join(r.id + (f" ({r.doc_id})" if r.doc_id else "") for r in missing)
                + ". Поведение по ним невозможно принять объективно.",
        recommendation="Добавить критерии приёмки «исходные условия — действие — ожидаемый результат» для каждого "
                       "правила и каждой его ветви (границы, исключения, повторы, смена настроек).",
        req_ids=[r.id for r in missing], locs=[(r.g1, r.g2) for r in missing], quotes=quotes,
        g1=missing[0].g1, g2=missing[0].g2)


# ------------------------------------------------------------------ 5. недостоверный OCR не даёт CRITICAL
def ocr_guard(doc, findings, min_conf: float = 85.0) -> int:
    """Если доказательство дефекта CRITICAL целиком взято из строк, распознанных OCR с невысокой уверенностью,
    дефект не считается подтверждённым критичным: понижаем до MAJOR и требуем сверки с оригиналом."""
    by = doc.by_gid
    n = 0
    for f in findings:
        if f.severity != "critical":
            continue
        spans = f.locs or ([(f.g1, f.g2)] if f.g1 is not None else [])
        lines = [by[g] for a, b in spans for g in range(a, (b or a) + 1) if g in by and by[g].text.strip()]
        if lines and all(ln.src == "ocr" and (ln.conf is None or ln.conf < min_conf) for ln in lines):
            confs = [ln.conf for ln in lines if ln.conf is not None]
            f.severity = "major"
            f.caveat = ("Критичность понижена с CRITICAL: доказательство распознано OCR"
                        + (f" (уверенность движка ≈{min(confs):.0f})" if confs else "")
                        + ". Сверьте числа, отрицания и знаки сравнения с оригиналом страницы.")
            n += 1
    return n


# ------------------------------------------------------------------ 6. «Почему это критично»
WHY_KEYS = ("trigger", "development", "testing", "business", "exploitation", "scenario", "resolution", "closure")


def explain_critical(doc, reqs, set_findings, llm, doc_text: str, prompts_mod, warnings: list[str], cov=None) -> None:
    from . import rules
    crit = [f for r in reqs for f in r.findings if f.severity == "critical"] + \
           [f for f in set_findings if f.severity == "critical"]
    title = "Обоснование критичных дефектов («Почему это критично»)"
    for f in crit:                                          # запасной вариант для правил без LLM
        if f.source == "правило" and f.tag in rules.WHY_CRITICAL:
            f.why = dict(rules.WHY_CRITICAL[f.tag])
    if not crit:
        if cov:
            cov.check("why", title, "выполнена", "критичных дефектов нет")
        return
    if llm.model == "mock":
        if cov:
            cov.check("why", title, "не выполнена", "демо-режим без модели")
        return
    items = [{"id": f.fid, "criterion": f.criterion, "requirements": f.req_ids[:8], "problem": f.problem,
              "quotes": [short(q, 200) for q in (f.quotes or [f.quote])[:4]], "recommendation": f.recommendation}
             for f in crit]
    log(f"[4/5] Обоснование критичности: {len(items)} дефектов…")
    try:
        data = llm.generate_json("why_critical", prompts_mod.SYSTEM,
                                 prompts_mod.why_prompt(doc.title, doc_text, items), {"items": items})
    except LLMError as e:
        warnings.append(f"Обоснование критичных дефектов не сформировано: {e}.")
        if cov:
            cov.check("why", title, "не выполнена", str(e)[:200])
        return
    idx = {f.fid: f for f in crit}
    got = 0
    for x in (data.get("items") or []) if isinstance(data, dict) else []:
        if not isinstance(x, dict) or x.get("id") not in idx:
            continue
        f = idx[x["id"]]
        why = {k: collapse_ws(str(x.get(k) or "")) for k in WHY_KEYS}
        if str(x.get("severity_ok", "yes")).lower() in ("no", "false", "нет"):
            f.severity = "major"
            f.caveat = "Критичность понижена до MAJOR: последствия не подтвердились при обосновании. " + \
                       collapse_ws(str(x.get("severity_reason") or ""))
            continue
        f.why = {k: v for k, v in why.items() if v}
        got += 1
    if cov:
        cov.check("why", title, "выполнена" if got or not crit else "не выполнена",
                  "" if got else "модель не вернула обоснований")
