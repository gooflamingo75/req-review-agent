"""Конвейер проверки: извлечение требований → проверка каждого → проверка набора."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import prompts, rules
from .criteria import INDIVIDUAL, SET, SEVERITY, SEVERITY_ORDER
from .llm import LLMError, log
from .loader import Document
from .util import collapse_ws, norm, short

EXTRACT_CHUNK_CHARS = 18000
FULL_CONTEXT_CHARS = 40000      # до такого размера документ целиком передаётся как контекст при проверке требований
SET_FULL_DOC_CHARS = 180000     # до такого размера документ целиком передаётся при проверке набора


@dataclass
class Finding:
    criterion: str
    severity: str
    problem: str
    recommendation: str
    quote: str = ""
    g1: int | None = None
    g2: int | None = None
    source: str = "LLM"                     # LLM | правило
    verified: bool = True                   # цитата найдена в документе дословно
    req_ids: list[str] = field(default_factory=list)
    type: str = ""
    fid: str = ""


@dataclass
class Requirement:
    id: str
    doc_id: str | None
    text: str
    summary: str
    kind: str
    g1: int
    g2: int
    section: str = ""
    criteria: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    reviewed: bool = False


@dataclass
class Result:
    doc: Document
    model_name: str
    requirements: list[Requirement]
    set_findings: list[Finding]
    set_criteria: dict[str, dict]
    stats: dict
    warnings: list[str]
    started: float
    finished: float = 0.0
    llm_calls: int = 0


def _sev(s) -> str:
    s = str(s or "").lower().strip()
    return s if s in SEVERITY else "major"


# ------------------------------------------------------------------ 1. извлечение
def _chunks(doc: Document) -> list[list]:
    out, cur, size = [], [], 0
    for ln in doc.nonempty():
        t = len(ln.text)
        # режем по заголовкам, если фрагмент уже большой
        if cur and (size + t > EXTRACT_CHUNK_CHARS or (ln.is_heading and size > EXTRACT_CHUNK_CHARS * 0.6)):
            out.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += t + 8
    if cur:
        out.append(cur)
    return out


def extract(doc: Document, llm, warnings: list[str]) -> list[Requirement]:
    chunks = _chunks(doc)
    raw = []
    for i, ch in enumerate(chunks, 1):
        log(f"[1/3] Выделение требований: фрагмент {i}/{len(chunks)}…")
        payload = {"lines": [(ln.gid, ln.text) for ln in ch]}
        data = llm.generate_json("extract", prompts.SYSTEM,
                                 prompts.extract_prompt(doc.title, doc.numbered(ch), i, len(chunks)), payload)
        items = data.get("requirements", []) if isinstance(data, dict) else data
        raw.extend(x for x in (items or []) if isinstance(x, dict))

    gids = {ln.gid for ln in doc.lines}
    reqs, seen, dropped = [], set(), 0
    for it in raw:
        text = collapse_ws(str(it.get("text") or ""))
        try:
            ls = int(str(it.get("line_start") or 0).lstrip("L"))
            le = int(str(it.get("line_end") or ls).lstrip("L"))
        except ValueError:
            ls = le = 0
        if le < ls:
            le = ls
        found = doc.find(text, near=(ls, le) if ls in gids else None) if text else None
        if found:
            text, g1, g2 = found
        elif ls in gids:
            g1, g2 = ls, le if le in gids else ls
            text = doc.text_of(g1, g2)
            warnings.append(f"Формулировка требования на строке L{ls} не совпала с текстом дословно; "
                            "в отчёт взят исходный текст строк документа.")
        else:
            dropped += 1
            continue
        key = norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        doc_id = it.get("doc_id")
        doc_id = collapse_ws(str(doc_id)) if doc_id not in (None, "", "null", "None") else None
        reqs.append(Requirement(id="", doc_id=doc_id, text=text,
                                summary=collapse_ws(str(it.get("summary") or "")) or short(text, 80),
                                kind=collapse_ws(str(it.get("kind") or "")) or "—",
                                g1=g1, g2=g2, section=doc.section_of(g1)))
    if dropped:
        warnings.append(f"{dropped} фрагмент(ов), предложенных моделью как требования, не найдены в документе и отброшены.")
    reqs.sort(key=lambda r: (r.g1, r.g2))
    for n, r in enumerate(reqs, 1):
        r.id = f"R-{n:03d}"
    return reqs


# ------------------------------------------------------------------ 2. проверка каждого требования
def _anchor(doc: Document, quote: str, req: Requirement) -> tuple[str, int, int, bool]:
    """Проверяет, что цитата есть в документе (в районе требования). Если нет — привязывает к требованию целиком."""
    if quote:
        f = doc.find(quote, near=(req.g1, req.g2))
        if f and req.g1 - 2 <= f[1] <= req.g2 + 2:
            return f[0], f[1], f[2], True
        if norm(quote).strip(" .…") and norm(quote).strip(" .…") in norm(req.text):
            return quote, req.g1, req.g2, True
    return req.text, req.g1, req.g2, False


def review_requirements(doc: Document, reqs: list[Requirement], llm, batch_size: int, warnings: list[str]):
    if doc.char_count() <= FULL_CONTEXT_CHARS:
        context = doc.numbered()
    else:
        context = "Структура документа (заголовки):\n" + doc.headings_outline()
    hints = {r.id: rules.check_requirement(r.text) for r in reqs}
    by_id = {r.id: r for r in reqs}
    batches = [reqs[i:i + batch_size] for i in range(0, len(reqs), batch_size)]
    for bi, batch in enumerate(batches, 1):
        log(f"[2/3] Проверка требований: пакет {bi}/{len(batches)} ({batch[0].id}…{batch[-1].id})…")
        payload = [{"id": r.id, "doc_id": r.doc_id, "section": r.section, "text": r.text,
                    "rule_hints": [f"«{h['trigger']}»: {h['problem']}" for h in hints[r.id] if h["trigger"]]}
                   for r in batch]
        try:
            data = llm.generate_json("review", prompts.SYSTEM,
                                     prompts.review_prompt(doc.title, context, payload),
                                     {"requirements": payload})
        except LLMError as e:
            warnings.append(f"Пакет {batch[0].id}…{batch[-1].id} не проверен моделью: {e}. Применены только правила.")
            data = {"results": []}
        results = data.get("results", []) if isinstance(data, dict) else (data or [])
        for res in results:
            if not isinstance(res, dict):
                continue
            r = by_id.get(str(res.get("id", "")).strip())
            if r is None or r.reviewed:
                continue
            r.reviewed = True
            crit = res.get("criteria") or {}
            r.criteria = {c: (str(crit.get(c, "na")).lower() if str(crit.get(c, "na")).lower() in ("ok", "fail", "na") else "na")
                          for c in INDIVIDUAL}
            for f in res.get("findings") or []:
                if not isinstance(f, dict):
                    continue
                c = str(f.get("criterion", "")).upper().strip()
                if c not in INDIVIDUAL:
                    c = "COR"
                problem = collapse_ws(str(f.get("problem") or ""))
                if not problem:
                    continue
                q, g1, g2, ok = _anchor(doc, str(f.get("quote") or ""), r)
                r.findings.append(Finding(criterion=c, severity=_sev(f.get("severity")), problem=problem,
                                          recommendation=collapse_ws(str(f.get("recommendation") or "")),
                                          quote=q, g1=g1, g2=g2, verified=ok, req_ids=[r.id],
                                          source="LLM" if llm.model != "mock" else "правило"))

    # правила: добавляем значимые срабатывания, которые модель не учла
    for r in reqs:
        if not r.reviewed:
            r.criteria = {c: "na" for c in INDIVIDUAL}
            if llm.model != "mock":
                warnings.append(f"{r.id}: модель не вернула результат проверки; применены только правила.")
        if llm.model == "mock" and r.reviewed:
            continue
        for h in hints[r.id]:
            if r.reviewed and h["severity"] == "minor":
                continue
            trig = norm(h["trigger"])
            covered = any(trig and f.verified and trig in norm(f.quote) and (f.criterion in h["criteria"])
                          for f in r.findings)
            if covered:
                continue
            q, g1, g2, ok = _anchor(doc, h["quote"], r)
            r.findings.append(Finding(criterion=h["criterion"], severity=h["severity"], problem=h["problem"],
                                      recommendation=h["recommendation"], quote=q, g1=g1, g2=g2, verified=ok,
                                      req_ids=[r.id], source="правило"))
        for f in r.findings:
            r.criteria[f.criterion] = "fail"
        r.findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], list(INDIVIDUAL).index(f.criterion)))


# ------------------------------------------------------------------ 3. проверка набора
def review_set(doc: Document, reqs: list[Requirement], llm, warnings: list[str]):
    log("[3/3] Проверка набора требований целиком…")
    if doc.char_count() <= SET_FULL_DOC_CHARS:
        doc_text = doc.numbered()
    else:
        doc_text = "(документ большой — приведена структура заголовков)\n" + doc.headings_outline()
    rl = [{"id": r.id, "doc_id": r.doc_id, "text": r.text, "g1": r.g1} for r in reqs]
    try:
        data = llm.generate_json("set", prompts.SYSTEM, prompts.set_prompt(doc.title, doc_text, rl), {"requirements": rl})
    except LLMError as e:
        warnings.append(f"Проверка набора моделью не выполнена: {e}. Применены только правила.")
        data = {}
    if not isinstance(data, dict):
        data = {}
    by_id = {r.id: r for r in reqs}
    findings: list[Finding] = []
    dropped = 0
    for f in data.get("findings") or []:
        if not isinstance(f, dict):
            continue
        ids = [i for i in (str(x).strip() for x in (f.get("requirement_ids") or [])) if i in by_id]
        near = (by_id[ids[0]].g1, by_id[ids[0]].g2) if ids else None
        quotes, locs = [], []
        for q in f.get("quotes") or []:
            hit = doc.find(str(q), near=near)
            if hit:
                quotes.append(hit[0])
                locs.append((hit[1], hit[2]))
        if not quotes and not ids:
            dropped += 1
            continue
        c = str(f.get("criterion", "")).upper().strip()
        if c not in SET:
            c = "S-CONS"
        problem = collapse_ws(str(f.get("problem") or ""))
        if not problem:
            continue
        fd = Finding(criterion=c, severity=_sev(f.get("severity")), problem=problem,
                     recommendation=collapse_ws(str(f.get("recommendation") or "")),
                     quote=" … ".join(f"«{q}»" for q in quotes), req_ids=ids,
                     type=collapse_ws(str(f.get("type") or "Прочее")), verified=bool(quotes))
        if locs:
            fd.g1, fd.g2 = locs[0]
            fd._locs = locs  # type: ignore[attr-defined]
        findings.append(fd)
    if dropped:
        warnings.append(f"{dropped} замечание(й) к набору отброшено: модель не привела ни подтверждаемой цитаты, "
                        "ни ссылки на требование.")

    # если модель отметила «соответствие форме» почти у всех требований — это свойство документа, а не
    # отдельных требований: сворачиваем в одно замечание к набору, чтобы не засорять отчёт
    cnf_reqs = [r for r in reqs if any(f.criterion == "CNF" and f.severity == "minor" for f in r.findings)]
    if reqs and len(cnf_reqs) >= max(5, 0.6 * len(reqs)):
        sample = next(f for f in cnf_reqs[0].findings if f.criterion == "CNF")
        for r in cnf_reqs:
            r.findings = [f for f in r.findings if not (f.criterion == "CNF" and f.severity == "minor")]
            r.criteria["CNF"] = "fail"
        findings.append(Finding(
            criterion="S-COMP", severity="minor", type="Несогласованность",
            problem=f"У {len(cnf_reqs)} из {len(reqs)} требований формулировка не соответствует шаблону "
                    f"(пример — {cnf_reqs[0].id}: {sample.problem})",
            recommendation="Привести требования к единому шаблону «<Субъект> должен <действие> <условие>» "
                           "с уникальным идентификатором.",
            req_ids=[r.id for r in cnf_reqs][:15], g1=cnf_reqs[0].g1, g2=cnf_reqs[0].g2, source="правило"))

    # правила набора
    rd = [{"id": r.id, "doc_id": r.doc_id, "text": r.text} for r in reqs]
    for h in rules.check_set(rd):
        if h["type"] == "Дублирование" and any(set(h["requirement_ids"]) <= set(f.req_ids) for f in findings):
            continue
        findings.append(Finding(criterion=h["criterion"], severity=h["severity"], problem=h["problem"],
                                recommendation=h["recommendation"], req_ids=h["requirement_ids"],
                                type=h["type"], source="правило"))
    for f in findings:
        if f.g1 is None and f.req_ids:
            r0 = by_id[f.req_ids[0]]
            f.g1, f.g2 = r0.g1, r0.g2
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], list(SET).index(f.criterion)))

    sc_raw = data.get("set_criteria") or {}
    set_criteria = {}
    for c in SET:
        v = sc_raw.get(c) if isinstance(sc_raw, dict) else None
        status = str((v or {}).get("status", "")).lower() if isinstance(v, dict) else ""
        comment = collapse_ws(str((v or {}).get("comment", ""))) if isinstance(v, dict) else ""
        related = [f for f in findings if f.criterion == c]
        if any(f.severity == "critical" for f in related):
            status = "fail"
        elif related and status in ("", "ok"):
            status = "partial"
        elif not status:
            status = "ok" if llm.model != "mock" else "n/a"
        if not comment:
            comment = ("Замечания: " + ", ".join(f.fid or "" for f in related)) if related else ""
        set_criteria[c] = {"status": status, "comment": comment, "findings": related}
    return findings, set_criteria


# ------------------------------------------------------------------ всё вместе
def run(doc: Document, llm, batch_size: int = 8) -> Result:
    warnings: list[str] = []
    started = time.time()
    if llm.model == "mock":
        warnings.append("Отчёт сформирован в ДЕМО-режиме без языковой модели (только правила). "
                        "Для сдачи запустите агента с ключом Gemini.")
    reqs = extract(doc, llm, warnings)
    log(f"      найдено требований: {len(reqs)}")
    if not reqs:
        warnings.append("В документе не найдено ни одного требования.")
    review_requirements(doc, reqs, llm, batch_size, warnings)
    set_findings, set_criteria = review_set(doc, reqs, llm, warnings)

    # нумерация замечаний
    n = 0
    for r in reqs:
        for f in r.findings:
            n += 1
            f.fid = f"F-{n:03d}"
    for i, f in enumerate(set_findings, 1):
        f.fid = f"S-{i:02d}"
    for c, v in set_criteria.items():
        if v["findings"] and v["comment"].startswith("Замечания: "):
            v["comment"] = "Замечания: " + ", ".join(f.fid for f in v["findings"])

    all_f = [f for r in reqs for f in r.findings] + set_findings
    stats = {
        "requirements": len(reqs),
        "with_issues": sum(1 for r in reqs if r.findings),
        "without_issues": sum(1 for r in reqs if not r.findings),
        "findings": len(all_f),
        "req_findings": sum(len(r.findings) for r in reqs),
        "set_findings": len(set_findings),
        "by_severity": {s: sum(1 for f in all_f if f.severity == s) for s in SEVERITY},
        "by_criterion": {c: sum(1 for r in reqs for f in r.findings if f.criterion == c) for c in INDIVIDUAL},
        "unverified": sum(1 for f in all_f if not f.verified and f.source == "LLM"),
        "mock": llm.model == "mock",
    }
    return Result(doc=doc, model_name=llm.name, requirements=reqs, set_findings=set_findings,
                  set_criteria=set_criteria, stats=stats, warnings=warnings, started=started,
                  finished=time.time(), llm_calls=llm.calls)
