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
_FID = re.compile(r"\b(?:F-\d{3}|S-\d{2}|EC-\d{2}|Q-\d{2})\b")


def _lines_of(f) -> set[int]:
    out = set()
    for a, b in (f.locs or []):
        out |= set(range(a, (b if b is not None else a) + 1))
    if f.g1 is not None:
        out |= set(range(f.g1, (f.g2 if f.g2 is not None else f.g1) + 1))
    return out


def restamp_dropped(dropped: list[dict]) -> None:
    """Ссылки на номера дефектов в причинах отклонения пересчитываются после окончательной нумерации:
    живой дефект → его итоговый номер, отклонённый → его номер в журнале (D-xx), иначе ссылка убирается."""
    index = dropped[0].get("_index") if dropped else None
    for i, d in enumerate(dropped, 1):
        d["id"] = f"D-{i:02d}"
    if not index:
        return
    old_to_d = {d.get("_old"): d["id"] for d in dropped}

    def sub(m):
        k = m.group(0)
        if k in old_to_d:
            return old_to_d[k]
        obj = index.get(k)
        obj = obj[1] if isinstance(obj, tuple) else obj
        fid = getattr(obj, "fid", None) if obj is not None else None
        return fid if fid and not getattr(obj, "_dropped", False) else "этот дефект"
    for d in dropped:
        for key in ("reason", "problem"):
            if d.get(key):
                d[key] = _FID.sub(sub, d[key])
        d.pop("_index", None)
        d.pop("_old", None)


def critic(doc, reqs, set_findings, llm, doc_text: str, prompts_mod, warnings: list[str], cov=None,
           edge_cases=None) -> list[dict]:
    """Возвращает список отклонённых дефектов (для приложения отчёта)."""
    items, index = [], {}
    edge_cases = edge_cases if edge_cases is not None else []
    for i, e in enumerate(edge_cases, 1):
        items.append({"id": f"EC-{i:02d}", "level": "пограничный сценарий", "requirements": e.get("req_ids"),
                      "problem": e["scenario"], "risk": e.get("risk", ""), "question": e.get("question", "")})
    qmap, qn = {}, 0
    for r in reqs:
        for j, q in enumerate(r.questions):
            qn += 1
            qmap[f"Q-{qn:02d}"] = (r, j)
            items.append({"id": f"Q-{qn:02d}", "level": "вопрос", "requirement": r.id, "problem": q})
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
    answered_ec, answered_q = set(), set()
    for rv in (data.get("reviews") or []) if isinstance(data, dict) else []:
        if not isinstance(rv, dict):
            continue
        rid = str(rv.get("id") or "")
        if (rid.startswith("EC-") or rid.startswith("Q-")) and str(rv.get("verdict", "")).lower() == "drop":
            ev = collapse_ws(str(rv.get("evidence_quote") or ""))
            hit = doc.find(ev) if ev and len(ev) >= 12 else None
            if not hit:
                continue
            if rid.startswith("EC-") and rid[3:].isdigit() and 0 < int(rid[3:]) <= len(edge_cases):
                e = edge_cases[int(rid[3:]) - 1]
                answered_ec.add(int(rid[3:]) - 1)
                where, prob = ", ".join(e.get("req_ids") or []) or "набор", "Сценарий: " + e["scenario"]
            elif rid in qmap:
                answered_q.add(rid)
                where, prob = qmap[rid][0].id, "Вопрос: " + qmap[rid][0].questions[qmap[rid][1]]
            else:
                continue
            dropped.append({"_old": rid, "_index": index, "where": where, "problem": prob,
                            "reason": collapse_ws(str(rv.get("reason") or "")) or "Уже описано в документе.",
                            "evidence": f"{doc.location(hit[1], hit[2])} — «{hit[0]}»"})
            continue
        if rid not in index:
            continue
        r, f = index[rv["id"]]
        verdict = str(rv.get("verdict", "keep")).lower()
        reason = collapse_ws(str(rv.get("reason") or ""))
        if verdict == "drop":
            ev = collapse_ws(str(rv.get("evidence_quote") or ""))
            hit = doc.find(ev) if ev and len(ev) >= 12 else None
            # отклоняем только с дословным доказательством из ДРУГОГО места документа: строка самого дефекта
            # не может опровергнуть пропуск или неоднозначность в ней же. Стандартные поля шаблона (APP, не
            # CRITICAL) допускаем отклонять без цитаты. CRITICAL отклоняем, только если доказательство длиннее
            # формального совпадения (не менее 25 символов).
            own = _lines_of(f)
            if hit and set(range(hit[1], hit[2] + 1)) <= own:
                hit = None
            if hit and f.severity == "critical" and len(ev) < 25:
                hit = None
            if hit or (f.criterion == "APP" and f.severity != "critical"):
                dropped.append({"_old": f.fid, "_index": index, "where": (r.id if r else ", ".join(f.req_ids[:5])),
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
    edge_cases[:] = [e for i, e in enumerate(edge_cases) if i not in answered_ec]
    drop_q = {}
    for k in answered_q:
        r, j = qmap[k]
        drop_q.setdefault(r.id, (r, set()))[1].add(j)
    for r, js in drop_q.values():
        r.questions = [q for j, q in enumerate(r.questions) if j not in js]
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
    quotes = [r.text for r in missing]
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
        for k in ("decision", "impact", "owner"):
            why[k] = collapse_ws(str(x.get(k) or ""))
        opts = x.get("options") if isinstance(x.get("options"), list) else []
        f.why = {k: v for k, v in why.items() if v}
        opts = [collapse_ws(str(o)) for o in opts if collapse_ws(str(o))][:3]
        if len(opts) >= 2:
            f.why["options"] = opts
        got += 1
    if cov:
        cov.check("why", title, "выполнена" if got or not crit else "не выполнена",
                  "" if got else "модель не вернула обоснований")


# ------------------------------------------------------------------ согласованность отметок
def align_marks(reqs, edge_cases) -> int:
    """Отметки требования не должны противоречить найденным дефектам:
    — требование затронуто системным дефектом → соответствующий критерий не может быть «выполнен»;
    — по требованию есть открытый пограничный сценарий → «Полнота» не выше «замечания»."""
    rank = {"na": 0, "ok": 1, "warn": 2, "fail": 3}
    changed = 0

    def bump(r, c, st, why):
        nonlocal changed
        if rank[st] > rank.get(r.criteria.get(c, "na"), 0):
            r.criteria[c] = st
            if not r.comments.get(c):
                r.comments[c] = why
            changed += 1

    for r in reqs:
        for f in r.linked:
            c = SET_TO_IND.get(f.criterion)
            if c:
                bump(r, c, "fail" if f.severity == "critical" else "warn",
                     f"Затронуто системным дефектом {f.fid or ''}".strip() + ".")
    by_id = {r.id: r for r in reqs}
    for i, e in enumerate(edge_cases, 1):
        for rid in e.get("req_ids") or []:
            r = by_id.get(rid)
            if r is not None:
                bump(r, "COM", "warn", f"Не описан пограничный сценарий EC-{i:02d}.")
    return changed


# ------------------------------------------------------------------ ссылки на строки и цитаты в тексте
_REF = re.compile(r"\[L(\d+)(?:\s*[–-]\s*L?(\d+))?\]")
_QUOTE = re.compile(r"«([^«»]{6,300})»")


def _fix_text(doc, text: str, near: tuple[int, int] | None, stat: dict) -> str:
    if not text:
        return text
    lines = doc.by_gid

    # 1) цитаты «…»: дословный текст документа вместо слегка искажённого (опечатки модели)
    def fix_q(m):
        q = m.group(1)
        if doc.find(q):
            return m.group(0)
        hit = doc.find(q, near) if near else None
        if hit and hit[0] and hit[0] != q and len(hit[0]) <= len(q) * 1.5 + 10:
            stat["quotes"] += 1
            return f"«{hit[0]}»"
        return m.group(0)
    text = _QUOTE.sub(fix_q, text)

    # 2) [Lnn]: строка должна существовать, быть непустой и содержать цитату из того же предложения
    def fix_ref(m):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        s0 = max(text.rfind(". ", 0, m.start()), text.rfind("; ", 0, m.start())) + 1
        e0 = min([x for x in (text.find(". ", m.end()), text.find("; ", m.end())) if x >= 0] or [len(text)])
        quotes = _QUOTE.findall(text[s0:e0])
        rng = range(min(a, b), max(a, b) + 1)
        if quotes:
            hits = [doc.find(q, near) for q in quotes]
            hits = [h for h in hits if h]
            if hits and not any(set(range(h[1], h[2] + 1)) & set(rng) for h in hits):
                h = min(hits, key=lambda h: abs(h[1] - a))
                stat["refs"] += 1
                return f"[L{h[1]}]" if h[1] == h[2] else f"[L{h[1]}–L{h[2]}]"
        if not any(lines.get(g) is not None and lines[g].text.strip() for g in rng):
            stat["refs_removed"] += 1
            return ""
        return m.group(0)
    return _REF.sub(fix_ref, text)


def fix_references(doc, reqs, set_findings, edge_cases) -> dict:
    """Перепроверяет ссылки [Lnn] и цитаты «…» в тексте дефектов: ссылка на пустую строку или на строку, где нет
    цитируемого текста, исправляется по найденной цитате или удаляется; искажённая цитата заменяется дословной."""
    stat = {"refs": 0, "refs_removed": 0, "quotes": 0}
    for f in [f for r in reqs for f in r.findings] + set_findings:
        near = (f.g1, f.g2 if f.g2 is not None else f.g1) if f.g1 is not None else None
        f.problem = _fix_text(doc, f.problem, near, stat)
        f.recommendation = _fix_text(doc, f.recommendation, near, stat)
        f.also = [_fix_text(doc, a, near, stat) for a in (getattr(f, "also", None) or [])]
        if getattr(f, "why", None):
            f.why = {k: _fix_text(doc, v, near, stat) if isinstance(v, str) else v for k, v in f.why.items()}
        if getattr(f, "caveat", None):
            f.caveat = _fix_text(doc, f.caveat, near, stat)
    for r in reqs:
        near = (r.g1, r.g2)
        r.comments = {k: _fix_text(doc, v, near, stat) for k, v in r.comments.items()}
        r.questions = [_fix_text(doc, q, near, stat) for q in r.questions]
        r.test_ideas = [_fix_text(doc, q, near, stat) for q in r.test_ideas]
    for e in edge_cases:
        for k in ("scenario", "risk", "question"):
            e[k] = _fix_text(doc, e.get(k) or "", None, stat)
    return stat


def fill_quotes(doc, set_findings) -> int:
    """У каждого места дефекта должна быть цитата: пустую цитату заменяем текстом строки документа."""
    n = 0
    for f in set_findings:
        if len(f.quotes) != len(f.locs):
            continue
        for i, ((a, b), q) in enumerate(zip(f.locs, f.quotes)):
            if q and q.strip():
                continue
            txt = " ".join(doc.by_gid[g].text.strip() for g in range(a, (b or a) + 1)
                           if doc.by_gid.get(g) is not None and doc.by_gid[g].text.strip())
            if txt:
                f.quotes[i] = collapse_ws(txt)
                n += 1
        f.quote = " … ".join(f"«{q}»" for q in f.quotes if q)
    return n
