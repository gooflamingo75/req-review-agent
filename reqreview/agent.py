"""Конвейер проверки:
1) выделение требований → 2) проверка каждого → 3) проверка набора (две части: непротиворечивость/стандарты
и полнота/пограничные сценарии/прослеживаемость) + детерминированные правила → 4) верификация цитат.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from . import prompts, rules, verify
from .consolidate import consolidate, similarity
from .coverage import Coverage, run_status
from .criteria import INDIVIDUAL, SET, SEVERITY, SEVERITY_ORDER
from .llm import LLMError, log
from .loader import Document
from .util import collapse_ws, norm, short

EXTRACT_CHUNK_CHARS = 18000
FULL_CONTEXT_CHARS = 60000      # до такого размера документ целиком передаётся как контекст при проверке требований
SET_FULL_DOC_CHARS = 180000     # до такого размера документ целиком передаётся при проверке набора

RULE_TAGS = {"UNA": "AMBIGUITY", "COM": "PLACEHOLDER", "APP": "DESIGN-DETAIL", "SIN": "NON-ATOMIC",
             "FEA": "FEASIBILITY", "VER": "UNTESTABLE", "CNF": "FORMAT"}
STATUS_RANK = {"na": 0, "ok": 1, "warn": 2, "fail": 3}
_SEV_ALIASES = {"blocker": "critical", "high": "major", "medium": "major", "low": "minor",
                "критический": "critical", "существенный": "major", "замечание": "minor"}


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
    tag: str = ""
    fid: str = ""
    locs: list[tuple[int, int]] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    merged: int = 0                         # сколько повторов той же первопричины объединено в этот дефект
    also: list[str] = field(default_factory=list)   # другие формулировки той же первопричины
    why: dict = field(default_factory=dict)          # «Почему это критично» (для CRITICAL)
    caveat: str = ""                                 # оговорка (например, доказательство распознано OCR)
    criteria: list[str] = field(default_factory=list)   # все затронутые критерии одной первопричины
    kind: str = "defect"                             # defect | clarification | recommendation
    obs: list[str] = field(default_factory=list)     # исходные наблюдения (проходы), слитые в этот дефект

    def __eq__(self, other):                # сравнение по идентичности объекта (для списков и remove)
        return self is other

    __hash__ = object.__hash__


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
    part: str = ""
    criteria: dict[str, str] = field(default_factory=dict)
    comments: dict[str, str] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    test_ideas: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    reviewed: bool = False
    linked: list = field(default_factory=list)   # системные дефекты набора, затрагивающие требование
    related: list = field(default_factory=list)  # (дефект другого требования, его ID) — противоречия с этим


@dataclass
class Result:
    doc: Document
    model_name: str
    requirements: list[Requirement]
    set_findings: list[Finding]
    set_criteria: dict[str, dict]
    edge_cases: list[dict]
    traceability: list[dict]
    stats: dict
    warnings: list[str]
    started: float
    finished: float = 0.0
    llm_calls: int = 0
    dropped: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    status: str = "completed"
    status_reasons: list[str] = field(default_factory=list)


def _sev(s) -> str:
    s = str(s or "").lower().strip()
    s = _SEV_ALIASES.get(s, s)
    return s if s in SEVERITY else "major"


def _kind(s) -> str:
    t = str(s or "").lower().strip()
    if t.startswith(("clar", "вопрос", "уточн")):
        return "clarification"
    if t.startswith(("recom", "рекоменд", "suggest", "предлож")):
        return "recommendation"
    return "defect"


def _tag(s) -> str:
    t = collapse_ws(str(s or "")).upper().replace(" ", "-")
    return "".join(ch for ch in t if ch.isalnum() or ch in "-_/")[:24]


def _strs(v, limit=5) -> list[str]:
    if isinstance(v, str):
        v = [v]
    return [collapse_ws(str(x)) for x in (v or []) if collapse_ws(str(x))][:limit]


# ------------------------------------------------------------------ 1. извлечение
def _chunks(doc: Document) -> list[list]:
    out, cur, size = [], [], 0
    for ln in doc.nonempty():
        t = len(ln.text)
        if cur and (size + t > EXTRACT_CHUNK_CHARS or (ln.is_heading and size > EXTRACT_CHUNK_CHARS * 0.6)):
            out.append(cur)
            cur, size = [], 0
        cur.append(ln)
        size += t + 8
    if cur:
        out.append(cur)
    return out


def extract(doc: Document, llm, warnings: list[str], cov: Coverage | None = None) -> list[Requirement]:
    chunks = _chunks(doc)
    raw = []
    starts = [doc.span(ch[0].gid)[0] for ch in chunks]
    for i, ch in enumerate(chunks, 1):
        log(f"[1/5] Выделение требований: фрагмент {i}/{len(chunks)}…")
        a = 0 if i == 1 else starts[i - 1]
        b = starts[i] if i < len(chunks) else doc.canon_len
        rec = cov.chunk(f"EXT-{i:02d}", "extract", [(a, b)]) if cov else None
        payload = {"lines": [(ln.gid, ln.text) for ln in ch]}
        try:
            if rec:
                cov.sent(rec)
            data = llm.generate_json("extract", prompts.SYSTEM,
                                     prompts.extract_prompt(doc.title, doc.numbered(ch), i, len(chunks)), payload)
        except LLMError as e:
            if rec:
                cov.done(rec, False, str(e)[:200])
            if _fatal(e):
                raise
            warnings.append(f"Фрагмент {i}/{len(chunks)} документа не обработан моделью: {e}. "
                            "Требования из него не выделены и не проверены.")
            continue
        items = data.get("requirements", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            if rec:
                cov.done(rec, False, "ответ модели не соответствует схеме")
            warnings.append(f"Фрагмент {i}/{len(chunks)}: ответ модели не соответствует схеме; фрагмент не проверен.")
            continue
        if rec:
            cov.done(rec, True)
        raw.extend(x for x in (items or []) if isinstance(x, dict))

    gids = {ln.gid for ln in doc.lines}
    reqs, seen, dropped = [], set(), 0
    for it in raw:
        text = collapse_ws(str(it.get("text") or ""))
        try:
            ls = int(str(it.get("line_start") or 0).lstrip("Ll"))
            le = int(str(it.get("line_end") or ls).lstrip("Ll"))
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
                                g1=g1, g2=g2, section=doc.section_of(g1), part=doc.part_of(g1)))
    if dropped:
        warnings.append(f"{dropped} фрагмент(ов), предложенных моделью как требования, не найдены в документе и отброшены.")
    reqs.sort(key=lambda r: (r.g1, r.g2))
    width = 2 if len(reqs) < 100 else 3
    for n, r in enumerate(reqs, 1):
        r.id = f"REQ-{n:0{width}d}"
    return reqs


# ------------------------------------------------------------------ 2. проверка каждого требования
def _anchor(doc: Document, quote: str, req: Requirement) -> tuple[str, int, int, bool]:
    """Проверяет, что цитата есть в документе (в районе требования). Если нет — привязывает к требованию целиком."""
    if quote:
        f = doc.find(quote, near=(req.g1, req.g2))
        if f and req.g1 - 2 <= f[1] <= req.g2 + 2:
            return f[0], f[1], f[2], True
        q = norm(quote).strip(" .…")
        if q and q in norm(req.text):
            return quote, req.g1, req.g2, True
    return req.text, req.g1, req.g2, False


def _bump(r: Requirement, c: str, status: str):
    if STATUS_RANK[status] > STATUS_RANK.get(r.criteria.get(c, "na"), 0):
        r.criteria[c] = status


def parallel_map(fn, items: list) -> list:
    """Выполняет независимые запросы к модели параллельно (PARALLEL_REQUESTS, по умолчанию 4) и возвращает
    результаты в исходном порядке — отчёт от параллельности не зависит."""
    n = max(1, int(os.getenv("PARALLEL_REQUESTS", "4") or 1))
    if n == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as ex:
        return list(ex.map(fn, items))


def review_requirements(doc: Document, reqs: list[Requirement], llm, batch_size: int, warnings: list[str],
                        cov: Coverage | None = None):
    full = doc.char_count() <= FULL_CONTEXT_CHARS
    if full:
        context = doc.numbered()
    else:
        context = "Структура документа (заголовки):\n" + doc.headings_outline()
    ctx_spans = [(0, doc.canon_len)] if full else [doc.span(ln.gid) for ln in doc.lines if ln.is_heading]
    hints = {r.id: rules.check_requirement(r.text) for r in reqs}
    by_id = {r.id: r for r in reqs}
    by_num = {re.sub(r"\D", "", r.id).lstrip("0"): r for r in reqs}
    batches = [reqs[i:i + batch_size] for i in range(0, len(reqs), batch_size)]
    is_mock = llm.model == "mock"
    jobs = []
    for bi, batch in enumerate(batches, 1):
        payload = [{"id": r.id, "doc_id": r.doc_id, "section": r.section, "kind": r.kind, "text": r.text,
                    "rule_hints": [f"«{h['trigger']}»: {h['problem']}" for h in hints[r.id] if h["trigger"]]}
                   for r in batch]
        recs = [cov.chunk(f"REV-{bi:02d}/{r.id}", "review", [doc.span(r.g1, r.g2)], [r.id]) for r in batch] if cov else []
        for rc in recs:
            cov.sent(rc, ctx_spans)
        jobs.append((bi, batch, payload, recs))

    def call(job):
        bi, batch, payload, _ = job
        log(f"[2/5] Проверка требований: пакет {bi}/{len(batches)} ({batch[0].id}…{batch[-1].id}) отправлен…")
        try:
            return llm.generate_json("review", prompts.SYSTEM, prompts.review_prompt(doc.title, context, payload),
                                     {"requirements": payload})
        except LLMError as e:
            return e

    for (bi, batch, payload, recs), data in zip(jobs, parallel_map(call, jobs)):
        if isinstance(data, LLMError):
            for rc in recs:
                cov.done(rc, False, str(data)[:200])
            warnings.append(f"Пакет {batch[0].id}…{batch[-1].id} не проверен моделью: {data}. Применены только правила.")
            data = {"results": []}
        results = data.get("results", []) if isinstance(data, dict) else (data or [])
        for res in results:
            if not isinstance(res, dict):
                continue
            rid = str(res.get("id", "")).strip()
            r = by_id.get(rid) or by_num.get(re.sub(r"\D", "", rid).lstrip("0"))
            if r is None or r.reviewed:
                continue
            r.reviewed = True
            crit = res.get("criteria") or {}
            for c in INDIVIDUAL:
                v = crit.get(c, "na") if isinstance(crit, dict) else "na"
                if isinstance(v, dict):
                    st, cm = str(v.get("status", "na")).lower(), collapse_ws(str(v.get("comment") or ""))
                else:
                    st, cm = str(v).lower(), ""
                r.criteria[c] = st if st in STATUS_RANK else "na"
                if cm:
                    r.comments[c] = cm
            r.test_ideas = _strs(res.get("test_ideas"), 3)
            r.questions = _strs(res.get("questions"), 2)
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
                crits = [x for x in (str(y).upper().strip() for y in (f.get("criteria") or [])) if x in INDIVIDUAL]
                r.findings.append(Finding(criterion=c, severity=_sev(f.get("severity")), problem=problem,
                                          recommendation=collapse_ws(str(f.get("recommendation") or "")),
                                          quote=q, g1=g1, g2=g2, verified=ok, req_ids=[r.id], tag=_tag(f.get("tag")),
                                          source="правило" if is_mock else "LLM",
                                          criteria=list(dict.fromkeys([c] + crits)), kind=_kind(f.get("classification")),
                                          obs=[f"review:{r.id}"]))

        for rc, r in zip(recs, batch):
            if rc.status == "sent":
                cov.done(rc, r.reviewed, "" if r.reviewed else "модель не вернула результат по требованию")

    # правила: добавляем значимые срабатывания, которые модель не учла
    for r in reqs:
        if not r.reviewed:
            r.criteria = {c: "na" for c in INDIVIDUAL}
            if not is_mock:
                warnings.append(f"{r.id}: модель не вернула результат проверки; применены только правила.")
        if not (is_mock and r.reviewed):
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
                                          req_ids=[r.id], source="правило",
                                          tag=h.get("tag") or RULE_TAGS.get(h["criterion"], "")))
        for f in r.findings:
            if f.kind != "defect":
                continue
            for c in (f.criteria or [f.criterion]):
                _bump(r, c, "warn" if f.severity == "minor" else "fail")
        r.findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], list(INDIVIDUAL).index(f.criterion)))


# ------------------------------------------------------------------ 3. проверка набора
def _set_findings_from(data: dict, doc: Document, by_id: dict, counters: dict) -> list[Finding]:
    out = []
    for f in (data.get("findings") or []) if isinstance(data, dict) else []:
        if not isinstance(f, dict):
            continue
        ids = [i for i in (str(x).strip() for x in (f.get("requirement_ids") or [])) if i in by_id]
        near = (by_id[ids[0]].g1, by_id[ids[0]].g2) if ids else None
        quotes, locs = [], []
        for q in _strs(f.get("quotes"), 6):
            hit = doc.find(q, near=near)
            if hit:
                quotes.append(hit[0])
                locs.append((hit[1], hit[2]))
            else:
                counters["bad_quotes"] += 1
        if not quotes and not ids:
            counters["dropped"] += 1
            continue
        c = str(f.get("criterion", "")).upper().strip()
        if c not in SET:
            c = "S-CONS"
        problem = collapse_ws(str(f.get("problem") or ""))
        if not problem:
            continue
        fd = Finding(criterion=c, severity=_sev(f.get("severity")), problem=problem,
                     recommendation=collapse_ws(str(f.get("recommendation") or "")),
                     quotes=quotes, quote=" … ".join(f"«{q}»" for q in quotes), req_ids=ids,
                     type=collapse_ws(str(f.get("type") or "Прочее")), tag=_tag(f.get("tag")),
                     verified=bool(quotes), locs=locs, kind=_kind(f.get("classification")), criteria=[c],
                     obs=["set"])
        if locs:
            fd.g1, fd.g2 = locs[0]
        out.append(fd)
    return out


def _covered(rule: dict, findings: list[Finding]) -> bool:
    rule_gids = {g for g, _ in rule["locs"]}
    for f in findings:
        if f.tag and f.tag == rule["tag"] and (rule["tag"] not in ("PLACEHOLDER", "SECTION-EMPTY")
                                               or any(a <= g <= b for g in rule_gids for a, b in f.locs)):
            return True
        if f.criterion == rule["criterion"] and any(a <= g <= b for g in rule_gids for a, b in f.locs):
            return True
    return False


_ASSUME = re.compile(r"при условии|предполага|допуская|если .{0,40}ещё не|если .{0,40}еще не|assuming", re.I)
RULE_KINDS = ("бизнес-правило", "функциональное", "ограничение", "данные", "нефункциональное")
MAX_PAIRS = 45


def rule_matrix(reqs: list[Requirement]) -> tuple[list[dict], list[dict]]:
    """Пары нормативных правил и критерии приёмки, по которым модель ОБЯЗАНА дать вердикт (ничего не пропустить)."""
    rules_ = [r for r in reqs if any(k in (r.kind or "").lower() for k in RULE_KINDS)]
    acc = [r for r in reqs if "приём" in (r.kind or "").lower() or "прием" in (r.kind or "").lower()]
    pairs = []
    for i, a in enumerate(rules_):
        for b in rules_[i + 1:]:
            pairs.append((similarity(a.text, b.text), a, b))
    if len(pairs) > MAX_PAIRS:
        pairs.sort(key=lambda x: -x[0])
        pairs = pairs[:MAX_PAIRS]
    out = [{"id": f"P{n:02d}", "req_ids": [a.id, b.id], "a": short(a.text, 220), "b": short(b.text, 220)}
           for n, (_, a, b) in enumerate(pairs, 1)]
    acc_out = [{"id": f"A{n:02d}", "req_ids": [r.id], "criterion": short(r.text, 260)} for n, r in enumerate(acc, 1)]
    return out, acc_out


SET_PASSES = [("consistency", "непротиворечивость, архитектура и стандарты"),
              ("completeness", "полнота, пограничные сценарии, прослеживаемость")]


def _set_context(doc: Document, reqs: list[Requirement]):
    if doc.char_count() <= SET_FULL_DOC_CHARS:
        doc_text = doc.numbered()
    else:
        doc_text = "(документ большой — приведена структура заголовков)\n" + doc.headings_outline()
    rl = [{"id": r.id, "doc_id": r.doc_id, "text": r.text, "g1": r.g1} for r in reqs]
    doc_rules = rules.check_document(doc)
    hints = [f"[{h['tag']}] {h['problem']} (строки: {', '.join('L' + str(g) for g, _ in h['locs'])})" for h in doc_rules]
    return doc_text, rl, doc_rules, hints


def set_passes(doc: Document, reqs: list[Requirement], llm) -> dict:
    """Проходы 1 и 2 проверки набора не зависят от проверки отдельных требований — выполняются параллельно
    с ней. Возвращает {проход: ответ модели или LLMError}."""
    doc_text, rl, _, hints = _set_context(doc, reqs)

    def call(item):
        part, title = item
        log(f"[3/5] Проверка набора требований ({title}) отправлена…")
        try:
            return llm.generate_json(f"set_{part}", prompts.SYSTEM,
                                     prompts.set_prompt(part, doc.title, doc_text, rl, hints), {"requirements": rl})
        except LLMError as e:
            return e
    return dict(zip([p for p, _ in SET_PASSES], parallel_map(call, SET_PASSES)))


def review_set(doc: Document, reqs: list[Requirement], llm, warnings: list[str], cov: Coverage | None = None,
               prefetched: dict | None = None):
    doc_text, rl, doc_rules, hints = _set_context(doc, reqs)
    by_id = {r.id: r for r in reqs}
    counters = {"dropped": 0, "bad_quotes": 0}
    findings: list[Finding] = []
    sc_raw: dict = {}
    edge_cases, trace = [], []
    if prefetched is None:
        prefetched = set_passes(doc, reqs, llm)
    for step, (part, title) in enumerate(SET_PASSES, 1):
        data = prefetched.get(part)
        if not isinstance(data, LLMError):
            if cov:
                cov.check(f"set_{part}", f"Набор: {title}", "выполнена")
        else:
            e = data
            warnings.append(f"Проверка набора ({title}) моделью не выполнена: {e}. Применены только правила.")
            data = {}
            if cov:
                cov.check(f"set_{part}", f"Набор: {title}", "не выполнена", str(e)[:200])
        if not isinstance(data, dict):
            data = {}
        findings += _set_findings_from(data, doc, by_id, counters)
        if isinstance(data.get("set_criteria"), dict):
            sc_raw.update(data["set_criteria"])
        for e in data.get("edge_cases") or []:
            if isinstance(e, dict) and collapse_ws(str(e.get("scenario") or "")):
                edge_cases.append({"scenario": collapse_ws(str(e.get("scenario"))),
                                   "req_ids": [i for i in _strs(e.get("requirement_ids"), 6) if i in by_id],
                                   "risk": collapse_ws(str(e.get("risk") or "")),
                                   "question": collapse_ws(str(e.get("question") or ""))})
        for t in data.get("traceability") or []:
            if isinstance(t, dict) and str(t.get("requirement", "")).strip() in by_id:
                trace.append({"req": str(t["requirement"]).strip(),
                              **{k: [i for i in _strs(t.get(k), 8) if i in by_id]
                                 for k in ("goal", "ui", "api_data", "acceptance")},
                              "gap": collapse_ws(str(t.get("gap") or ""))})
    # часть 3: сценарный анализ противоречий между правилами + проверка тест-идей против правил
    log("[3/5] Проверка набора требований (3/3: сценарный анализ противоречий между правилами)…")
    known = [f"{', '.join(f.req_ids)}: {short(f.problem, 160)}" for f in findings if f.criterion == "S-CONS"]
    tests = [f"{r.id}: {t}" for r in reqs for t in r.test_ideas]
    pairs, accept = rule_matrix(reqs)
    assumed = [t for t in tests if _ASSUME.search(t)]
    try:
        data = llm.generate_json("set_conflicts", prompts.SYSTEM,
                                 prompts.conflicts_prompt(doc.title, doc_text, rl, hints, known, tests,
                                                          pairs, accept, assumed),
                                 {"requirements": rl, "pairs": pairs, "acceptance": accept})
        if cov:
            cov.check("set_conflicts", "Набор: сценарный анализ противоречий", "выполнена")
    except LLMError as e:
        warnings.append(f"Сценарный анализ противоречий моделью не выполнен: {e}.")
        data = {}
        if cov:
            cov.check("set_conflicts", "Набор: сценарный анализ противоречий", "не выполнена", str(e)[:200])
    if isinstance(data, dict):
        conf = []
        for c in data.get("conflicts") or []:
            if not isinstance(c, dict):
                continue
            c = dict(c)
            extra = " ".join(x for x in (
                f"Сценарий: {collapse_ws(str(c.get('scenario') or ''))}" if c.get("scenario") else "",
                f"Результат: {collapse_ws(str(c.get('outcomes') or ''))}" if c.get("outcomes") else "") if x)
            c["problem"] = collapse_ws(f"{c.get('problem') or ''} {extra}")
            c.setdefault("type", "Противоречие")
            c.setdefault("criterion", "S-CONS")
            c["tag"] = c.get("tag") or "CONTRADICTION"
            conf.append(c)
        # обязательная матрица: вердикт по КАЖДОЙ паре правил и КАЖДОМУ критерию приёмки
        pmap = {p["id"]: p for p in pairs}
        amap = {a["id"]: a for a in accept}
        covered_pairs = {frozenset(c.get("requirement_ids") or []) for c in conf}
        answered = 0
        for x in (data.get("pairs") or []) + (data.get("acceptance") or []):
            if not isinstance(x, dict):
                continue
            item = pmap.get(str(x.get("id"))) or amap.get(str(x.get("id")))
            if item is None:
                continue
            answered += 1
            verdict = str(x.get("verdict", "")).lower()
            if verdict not in ("conflict", "priority-unclear"):
                continue
            ids = item["req_ids"]
            if frozenset(ids) in covered_pairs:
                continue
            covered_pairs.add(frozenset(ids))
            conf.append({"requirement_ids": ids, "quotes": [by_id[i].text for i in ids],
                         "problem": collapse_ws(" ".join(t for t in (
                             str(x.get("problem") or "Правила дают разный результат в одном сценарии."),
                             f"Сценарий: {x['scenario']}" if x.get("scenario") else "",
                             f"Результат: {x['outcomes']}" if x.get("outcomes") else "") if t)),
                         "recommendation": collapse_ws(str(x.get("recommendation") or
                                                           "Явно указать, какое правило имеет приоритет в этом сценарии.")),
                         "severity": "critical" if verdict == "conflict" else "major", "type": "Противоречие",
                         "criterion": "S-CONS", "tag": "CONTRADICTION"})
        if pairs or accept:
            warnings.append(f"Матрица сценариев: модель дала вердикт по {answered} из {len(pairs) + len(accept)} "
                            "пар правил и критериев приёмки.")
        findings += _set_findings_from({"findings": conf}, doc, by_id, counters)
        for b in data.get("bad_test_ideas") or []:
            if not isinstance(b, dict):
                continue
            r = by_id.get(str(b.get("requirement_id", "")).strip())
            idea = collapse_ws(str(b.get("test_idea") or ""))
            if r is None or not idea:
                continue
            viol = ", ".join(i for i in _strs(b.get("violates"), 5) if i in by_id)
            note = f"⚠️ противоречит {viol}" + (f": {collapse_ws(str(b.get('why') or ''))}" if b.get("why") else "")
            for i, t in enumerate(r.test_ideas):
                if similarity(t, idea) >= 0.5:
                    r.test_ideas[i] = f"{t} — {note}"
                    break
            else:
                r.test_ideas.append(f"{idea} — {note}")

    if counters["dropped"]:
        warnings.append(f"{counters['dropped']} замечание(й) к набору отброшено: модель не привела ни подтверждаемой "
                        "цитаты, ни ссылки на требование.")
    if counters["bad_quotes"]:
        warnings.append(f"{counters['bad_quotes']} цитат(ы) модели в замечаниях к набору не найдены в документе "
                        "дословно и исключены из отчёта (сами замечания сохранены, если есть другие подтверждения).")

    # детерминированные проверки документа — добавляем, если модель их не нашла
    for h in doc_rules:
        if _covered(h, findings):
            continue
        quotes, locs = [], []
        for g, q in h["locs"]:
            hit = doc.find(q, near=(g, g))
            if hit:
                quotes.append(hit[0])
                locs.append((hit[1], hit[2]))
        if not quotes:
            continue
        ids = [r.id for r in reqs if any(r.g1 <= a <= r.g2 for a, _ in locs)]
        findings.append(Finding(criterion=h["criterion"], severity=h["severity"], problem=h["problem"],
                                recommendation=h["recommendation"], quotes=quotes,
                                quote=" … ".join(f"«{q}»" for q in quotes), req_ids=ids, type=h["type"], tag=h["tag"],
                                source="правило", locs=locs, g1=locs[0][0], g2=locs[0][1]))

    # если модель отметила «соответствие форме» почти у всех требований — это свойство документа
    cnf_reqs = [r for r in reqs if any(f.criterion == "CNF" and f.severity == "minor" for f in r.findings)]
    if reqs and len(cnf_reqs) >= max(5, 0.6 * len(reqs)):
        sample = next(f for f in cnf_reqs[0].findings if f.criterion == "CNF")
        for r in cnf_reqs:
            r.findings = [f for f in r.findings if not (f.criterion == "CNF" and f.severity == "minor")]
            r.criteria["CNF"] = "warn"
        findings.append(Finding(
            criterion="S-COMP", severity="minor", type="Несогласованность", tag="FORMAT",
            problem=f"У {len(cnf_reqs)} из {len(reqs)} требований формулировка не соответствует шаблону "
                    f"(пример — {cnf_reqs[0].id}: {sample.problem})",
            recommendation="Привести требования к единому шаблону «<Субъект> должен <действие> <условие>» "
                           "с уникальным идентификатором.",
            req_ids=[r.id for r in cnf_reqs], g1=cnf_reqs[0].g1, g2=cnf_reqs[0].g2, source="правило"))

    # правила набора (ID, дубли)
    rd = [{"id": r.id, "doc_id": r.doc_id, "text": r.text} for r in reqs]
    for h in rules.check_set(rd):
        if h["type"] == "Дублирование" and any(set(h["requirement_ids"]) <= set(f.req_ids) for f in findings):
            continue
        if h["criterion"] == "S-TRC" and any(f.criterion == "S-TRC" and f.source == "LLM" for f in findings):
            continue
        findings.append(Finding(criterion=h["criterion"], severity=h["severity"], problem=h["problem"],
                                recommendation=h["recommendation"], req_ids=h["requirement_ids"], type=h["type"],
                                tag="TRACEABILITY" if h["criterion"] == "S-TRC" else "DUPLICATE", source="правило"))
    for f in findings:
        if f.g1 is None and f.req_ids:
            r0 = by_id[f.req_ids[0]]
            f.g1, f.g2 = r0.g1, r0.g2
        if not f.locs and f.g1 is not None:
            f.locs = [(f.g1, f.g2)]
    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], list(SET).index(f.criterion)))

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
        set_criteria[c] = {"status": status, "comment": comment, "findings": related}
    return findings, set_criteria, edge_cases, trace


# ------------------------------------------------------------------ всё вместе
def _fatal(e: Exception) -> bool:
    """Ошибки, при которых продолжать бессмысленно (ключ, доступ) — их не превращаем в «частичную проверку»."""
    t = str(e).lower()
    return any(k in t for k in ("api key", "api_key", "permission", "unauthenticated", "401", "403", "ключ"))


def run(doc: Document, llm, batch_size: int = 8) -> Result:
    warnings: list[str] = []
    started = time.time()
    cov = Coverage(doc)
    if llm.model == "mock":
        warnings.append("Отчёт сформирован в ДЕМО-режиме без языковой модели (только правила). "
                        "Для сдачи запустите агента с ключом Gemini.")
    reqs = extract(doc, llm, warnings, cov)
    if cov.chunks and all(c.status == "failed" for c in cov.chunks if c.kind == "extract"):
        raise LLMError("ни один фрагмент документа не обработан моделью: " + cov.chunks[-1].note)
    log(f"      найдено требований: {len(reqs)}")
    if not reqs:
        warnings.append("В документе не найдено ни одного требования.")
    # проходы 1–2 проверки набора идут фоном, пока проверяются отдельные требования
    from concurrent.futures import ThreadPoolExecutor
    parallel = int(os.getenv("PARALLEL_REQUESTS", "4") or 1) > 1
    with ThreadPoolExecutor(max_workers=1) as bg:
        fut = bg.submit(set_passes, doc, reqs, llm) if parallel else None
        review_requirements(doc, reqs, llm, batch_size, warnings, cov)
        pre = fut.result() if fut is not None else None
    set_findings, set_criteria, edge_cases, trace = review_set(doc, reqs, llm, warnings, cov, prefetched=pre)
    cov.check("rules", "Автоматические проверки документа (без LLM)", "выполнена")
    acc = verify.acceptance_gaps(reqs, trace, Finding)
    if acc is not None:
        set_findings.append(acc)
    recs: list[dict] = []
    _split_kinds(reqs, set_findings, edge_cases, recs)
    log("[4/5] Объединение дефектов по первопричине…")
    set_findings, cstats = consolidate(reqs, set_findings, Finding, section_of=doc.section_of)
    if cstats["merged"]:
        warnings.append(f"Объединено {cstats['merged']} повторных замечаний с общей первопричиной: каждый дефект "
                        "показан один раз со списком всех мест, где он проявляется.")
    _link_systemic(reqs, set_findings)
    checks = {"pre": verify.self_check(doc, reqs, set_findings, Finding, final=False)}
    _number(reqs, set_findings)

    # самопроверка: LLM-ревьюер второго уровня + детерминированные проверки отчёта
    log("[4/5] Самопроверка: ревьюер второго уровня ищет ложные срабатывания…")
    doc_text = doc.numbered() if doc.char_count() <= SET_FULL_DOC_CHARS else doc.headings_outline()
    dropped = verify.critic(doc, reqs, set_findings, llm, doc_text, prompts, warnings, cov,
                             edge_cases=edge_cases)
    _split_kinds(reqs, set_findings, edge_cases, recs)
    checks["sanitized"] = verify.sanitize(doc, [f for r in reqs for f in r.findings] + set_findings)
    _number(reqs, set_findings)                       # временные номера для обоснования критичности
    allf = [f for r in reqs for f in r.findings] + set_findings
    checks["severity_rubric"] = verify.severity_rubric(allf)
    checks["ocr_guard"] = verify.ocr_guard(doc, allf)
    verify.explain_critical(doc, reqs, set_findings, llm, doc_text, prompts, warnings, cov)
    # окончательный порядок и номера — после всех изменений критичности; все ссылки пересчитываются от них
    for r in reqs:
        r.findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], list(INDIVIDUAL).index(f.criterion)))
    set_findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], -len(f.locs)))
    _link_systemic(reqs, set_findings)
    _number(reqs, set_findings)
    verify.restamp_dropped(dropped)
    checks["post"] = verify.self_check(doc, reqs, set_findings, Finding, final=True)
    checks["marks_aligned"] = verify.align_marks(reqs, edge_cases)
    verify.fill_quotes(doc, set_findings)
    refs = verify.fix_references(doc, reqs, set_findings, edge_cases)
    checks["references"] = refs
    if any(refs.values()):
        warnings.append(f"Проверка ссылок в тексте дефектов: исправлено ссылок на строки — {refs['refs']}, "
                        f"удалено ссылок на пустые строки — {refs['refs_removed']}, цитат приведено к дословному "
                        f"тексту — {refs['quotes']}.")
    if checks["sanitized"]:
        warnings.append(f"В {checks['sanitized']} рекомендациях значения, отсутствующие в документе (URL/пути), "
                        "заменены на «<указать …>».")
    log("[5/5] Формирование отчёта…")
    for c, v in set_criteria.items():
        related = [f for f in set_findings if f.criterion == c]
        v["findings"] = related
        if any(f.severity == "critical" for f in related):
            v["status"] = "fail"
        elif related and v["status"] in ("ok", "n/a"):
            v["status"] = "partial"
        elif not related and v["status"] == "fail":
            v["status"] = "partial"
        if v["comment"].startswith("Замечания: "):
            v["comment"] = ""

    all_f = [f for r in reqs for f in r.findings] + set_findings
    by_part: dict[str, int] = {}
    for f in all_f:
        p = doc.part_of(f.g1) if f.g1 is not None else ""
        by_part[p or "—"] = by_part.get(p or "—", 0) + 1
    stats = {
        "requirements": len(reqs),
        "with_issues": sum(1 for r in reqs if r.findings or r.linked),
        "without_issues": sum(1 for r in reqs if not r.findings and not r.linked),
        "findings": len(all_f),
        "req_findings": sum(len(r.findings) for r in reqs),
        "set_findings": len(set_findings),
        "by_severity": {s: sum(1 for f in all_f if f.severity == s) for s in SEVERITY},
        "by_criterion": {c: sum(1 for r in reqs for f in r.findings if c in (f.criteria or [f.criterion]))
                         for c in INDIVIDUAL},
        "by_part": by_part,
        "occurrences": sum(len(r.findings) for r in reqs) + sum(max(1, len(f.locs)) for f in set_findings),
        "merged": cstats["merged"],
        "unverified": sum(1 for f in all_f if not f.verified and f.source == "LLM"),
        "rule_findings": sum(1 for f in all_f if f.source == "правило"),
        "mock": llm.model == "mock",
        "model_usage": dict(getattr(llm, "used", {}) or {}),
        "primary_model": getattr(llm, "primary", None) if hasattr(llm, "primary_share") else None,
    }
    share = llm.primary_share() if hasattr(llm, "primary_share") else None
    stats["primary_share"] = round(share, 2) if share is not None else None
    stats["model_failures"] = dict(getattr(llm, "failures", {}) or {})
    if share is not None and share < 0.5:
        why = llm.failure_summary() if hasattr(llm, "failure_summary") else ""
        note = getattr(llm, "probe_note", None)
        warnings.append(f"Основная модель {stats['primary_model']} обработала только {round(share * 100)}% запросов; "
                        "остальные — резервные модели. "
                        + (f"Причина по ответам сервиса: {why}. " if why else "")
                        + (f"Проверка перед стартом: {note}. " if note else "")
                        + "Качество анализа может быть ниже: повторите запуск позже или с флагом --primary-only "
                          "(ждать основную модель вместо перехода на резервные).")
    csum = cov.summary()
    status, reasons = run_status(doc.ingest, csum, len(reqs), sum(1 for r in reqs if r.reviewed), llm.model == "mock")
    stats["affected_reqs"] = stats["with_issues"]
    stats["reviewed_reqs"] = sum(1 for r in reqs if r.reviewed)
    return Result(doc=doc, model_name=llm.name, requirements=reqs, set_findings=set_findings,
                  set_criteria=set_criteria, edge_cases=edge_cases, traceability=trace, stats=stats,
                  warnings=warnings, started=started, finished=time.time(), llm_calls=llm.calls,
                  dropped=dropped, checks=checks, coverage=csum, status=status, status_reasons=reasons,
                  recommendations=recs)


def _split_kinds(reqs, set_findings, edge_cases, recs) -> None:
    """Вопросы (clarification) и необязательные улучшения (recommendation) — не дефекты: они уходят в вопросы к
    аналитику и в список рекомендаций и не попадают в счётчики дефектов."""
    def q_of(f):
        t = f.problem.strip()
        return t if t.endswith("?") else (f.recommendation.strip() if f.recommendation.strip().endswith("?") else t)
    for r in reqs:
        keep = []
        for f in r.findings:
            if f.kind == "clarification":
                if all(similarity(q_of(f), q) < 0.6 for q in r.questions):
                    r.questions.append(q_of(f))
            elif f.kind == "recommendation":
                recs.append({"where": r.id, "req_ids": [r.id], "problem": f.problem,
                             "text": f.recommendation or f.problem})
            else:
                keep.append(f)
        r.findings = keep
    keep = []
    for f in set_findings:
        if f.kind == "clarification":
            edge_cases.append({"scenario": f.problem, "req_ids": f.req_ids[:6], "risk": "", "question": q_of(f)})
        elif f.kind == "recommendation":
            recs.append({"where": ", ".join(f.req_ids[:4]) or "набор", "req_ids": f.req_ids[:6], "problem": f.problem,
                         "text": f.recommendation or f.problem})
        else:
            keep.append(f)
    set_findings[:] = keep


def _number(reqs, set_findings):
    n = 0
    for r in reqs:
        for f in r.findings:
            n += 1
            f.fid = f"F-{n:03d}"
    for i, f in enumerate(set_findings, 1):
        f.fid = f"S-{i:02d}"


def _link_systemic(reqs, set_findings):
    """Системные дефекты (MAJOR/CRITICAL) показываем у каждого затронутого требования;
    противоречие с другим правилом переопределяет оценку «Непротиворечивость» требования."""
    by_id = {r.id: r for r in reqs}
    for r in reqs:
        r.linked = [f for f in r.linked if f in set_findings]
    for f in set_findings:
        if f.severity == "minor":
            continue
        for rid in f.req_ids:
            r = by_id.get(rid)
            if r is None:
                continue
            if f not in r.linked:
                r.linked.append(f)
            if f.criterion == "S-CONS" and (f.tag == "CONTRADICTION" or "ротиворе" in f.type):
                others = [x for x in f.req_ids if x != rid]
                r.criteria["CON"] = "fail"
                r.comments["CON"] = ("Противоречит " + ", ".join(others) if others else "Противоречие") + \
                    " — см. системный дефект в разделе 4.2."
