"""Записи, из которых строятся ВСЕ представления отчёта, и проверки отчёта перед выводом.

Канонические дефекты (findings) формирует конвейер агента. Здесь из них кодом строятся:
  - решения (decisions) — нерешённые вопросы, которые блокируют разработку: варианты, влияние, кто решает;
  - действия (plan) — по одному на канонический дефект, с привязкой к дефекту и к решению, от которого зависит;
  - вопросы к аналитику (questions) — по одному на нерешённый вопрос, со ссылками на все источники;
  - тест-кейсы (tests) — с пометкой «условный», если ожидаемый результат зависит от открытого решения;
  - статистика (stats) — пересчитывается из итоговых записей;
  - проверка согласованности (validation) — детерминированные инварианты (ID, ссылки, цитаты, счётчики, открытые
    решения, условные тесты, отметки критериев).
Первая страница, план, вопросы, тесты, матрицы и приложения читают одни и те же записи.
"""
from __future__ import annotations

import re

from .consolidate import similarity
from .util import collapse_ws

OWNER_BY_TAG = {"CONTRADICTION": "Владелец продукта", "PRIORITY": "Владелец продукта",
                "REST-METHOD": "Владелец API", "STATUS-200": "Владелец API", "STATUS-CODE": "Владелец API",
                "FORMAT": "Владелец API", "DATA-TYPE": "Владелец API", "SECURITY": "Владелец API",
                "IDEMPOTENCY": "Владелец API", "ERROR-HANDLING": "Аналитик", "UNITS": "Аналитик",
                "ACCEPTANCE": "QA-лид", "EDGE-CASE": "Аналитик"}
API_TAGS = {"REST-METHOD", "STATUS-200", "STATUS-CODE", "FORMAT", "DATA-TYPE", "SECURITY", "IDEMPOTENCY"}
ROLES = [("analyst", "Аналитик", "формулировки, пропуски, бизнес-правила, оформление документа"),
         ("api", "Владелец API", "контракт API: методы, коды ответов, типы и форматы данных"),
         ("qa", "QA", "критерии приёмки, тестовые разделы и проверяемость")]
BUSINESS_TAGS = {"CONTRADICTION", "PRIORITY", "UNITS", "EDGE-CASE"}
SEV_ORDER = {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
_POSITIVE = re.compile(r"^(реализуем\w*|выполним\w*|соответству\w*|согласован\w*|корректн\w*|без нарушени\w*|"
                       r"указан\w*|определен\w*|описан\w*|однозначн\w*)\b", re.I)
_CONDITIONAL = re.compile(r"\bесли\b|\bпри варианте\b|\bвариант\s+[АБAB]\b|\bзависит от\b|\bусловн", re.I)


def sentence(t: str) -> str:
    """Первое предложение целиком (без обрезки по символам)."""
    t = collapse_ws(t)
    m = re.search(r"(?<=[^\d\s][.!?])\s+(?=[А-ЯA-Z«])", t)
    return t[:m.start()] if m else t


def options_from(text: str) -> list[str]:
    """Варианты решения из формулировки «указать: A или B» / «X ли A или B» (для прогонов без поля options)."""
    t = collapse_ws(text or "")
    m = re.search(r"\bуказать:\s*([^():;]+?)\s+или\s+([^().;]+)", t, re.I)
    if m:
        a, b = m.group(1).strip(" ,«»"), m.group(2).strip(" ,«»")
    else:
        m = re.search(r"\b(\w+)\s+ли\s+(.+?)\s+или\s+(.+?)(?:[.;)]|$)", t, re.I)
        if not m:
            return []
        a, b = f"{m.group(1)}: {m.group(2).strip(' ,')}", m.group(3).strip(" ,")
    if not (2 <= len(a) <= 120 and 2 <= len(b) <= 120) or re.match(r"(если|когда|при)\b", a, re.I) \
            or (a[-1:].isdigit() and b[:1].isdigit()):
        return []
    return [a[:1].upper() + a[1:], b[:1].upper() + b[1:]]


def _reqs_of(f: dict) -> list[str]:
    return list(f.get("requirements") or ([f["_req"]] if f.get("_req") else []))


def role(v, f: dict) -> str:
    tag = f.get("tag") or ""
    kinds = {(v.req_by.get(r) or {}).get("kind", "").lower() for r in _reqs_of(f)}
    text = (f.get("problem") or "").lower()
    if tag == "ACCEPTANCE" or (tag == "SECTION-EMPTY" and "тест" in text) or \
            any("приём" in k or "прием" in k for k in kinds):
        return "qa"
    if tag in API_TAGS or any("api" in k or "интерфейс" in k for k in kinds):
        return "api"
    return "analyst"


def _reqs_ext(v, f: dict, gids: set) -> set:
    """Требования дефекта: явно указанные и те, на строки которых дефект ссылается (сторона противоречия)."""
    out = set(_reqs_of(f))
    for r in v.reqs:
        ln = list(r.get("lines") or [])
        if not ln or ln[0] is None:
            continue
        a, b = ln[0], ln[-1]
        if any(a <= g <= (b or a) for g in gids):
            out.add(r["id"])
    return out


def _compatible(a: dict, b: dict, A: set | None = None, B: set | None = None) -> bool:
    """Противоречия между РАЗНЫМИ парами правил — разные решения (разные переменные состояния)."""
    if "CONTRADICTION" in (a.get("tag"), b.get("tag")):
        A = A if A is not None else set(_reqs_of(a))
        B = B if B is not None else set(_reqs_of(b))
        if A and B and not (A & B):
            return False
    ta, tb = a.get("tag") or "", b.get("tag") or ""
    return not ta or not tb or ta == tb


def decision_groups(v, crit: list[dict]) -> list[list[dict]]:
    """Блокеры одного решения: модель назвала одно и то же решение, либо одна первопричина (близкий текст, общие
    строки) при совместимом составе требований. Цепочки A~B~C без проверки всей группы не склеиваются."""
    groups: list[list[dict]] = []
    txt = {f["id"]: f["problem"] + " " + (f.get("recommendation") or "") for f in crit}
    g = {f["id"]: v.gids(f) | {int(x) for x in re.findall(r"\bL(\d+)\b", txt[f["id"]])} for f in crit}
    rx = {f["id"]: _reqs_ext(v, f, g[f["id"]]) for f in crit}

    def dkey(f):
        return collapse_ws(((f.get("why_critical") or {}).get("decision") or "").lower()).rstrip("?")

    def same(a, b):
        if dkey(a) and dkey(a) == dkey(b):
            return True
        sim = similarity(txt[a["id"]], txt[b["id"]])
        return _compatible(a, b, rx[a["id"]], rx[b["id"]]) and \
            (sim >= 0.5 or (sim >= 0.35 and bool(g[a["id"]] & g[b["id"]])))
    for f in sorted(crit, key=lambda x: (-v.places(x), x["id"])):
        host = next((grp for grp in groups if all(same(f, x) for x in grp)), None)   # вся группа, не цепочка
        if host is None:
            groups.append([f])
        else:
            host.append(f)
    return groups


# ====================================================================== построение записей
def build_decisions(v) -> list[dict]:
    from .present import where_short
    out = []
    crit = [f for f in v.all_f if f["severity"] == "CRITICAL"]
    for n, grp in enumerate(decision_groups(v, crit), 1):
        f = grp[0]
        why = next((x.get("why_critical") for x in grp if (x.get("why_critical") or {}).get("decision")), None) \
            or f.get("why_critical") or {}
        rec = why.get("resolution") or f.get("recommendation") or ""
        opts = why.get("options") or options_from(rec) or options_from(f.get("recommendation") or "")
        business = (f.get("tag") or "") in BUSINESS_TAGS or len(opts) >= 2
        if len(opts) < 2:
            opts = [f"{sentence(rec).rstrip('.')} (рекомендуется)" if rec else "Исправить документ (рекомендуется)",
                    "Отложить: зафиксировать допущение и риск в документе"]
        owner = why.get("owner") or OWNER_BY_TAG.get(f.get("tag") or "") or \
            {"api": "Владелец API", "qa": "QA-лид"}.get(role(v, f), "Аналитик")
        out.append({"id": f"D-{n:02d}", "question": why.get("decision") or sentence(v.clean(f["problem"])),
                    "options": [v.clean(x) for x in opts], "kind": "business" if business else "technical",
                    "impact": why.get("impact") or why.get("business") or sentence(v.clean(f["problem"])),
                    "owner": owner, "findings": [x["id"] for x in grp], "status": "open", "authority": None,
                    "requirements": sorted({r for x in grp for r in _reqs_of(x)}), "where": where_short(f)})
    return out


def _depends(v, f: dict, decisions: list[dict]) -> list[str]:
    """Действие зависит от открытого бизнес-решения, если касается тех же требований и того же вопроса."""
    out = []
    for d in decisions:
        if d["kind"] != "business" or f["id"] in d["findings"]:
            continue
        if set(_reqs_of(f)) & set(d["requirements"]):
            ref = " ".join(v.by_fid[x]["problem"] for x in d["findings"] if x in v.by_fid)
            if similarity(f["problem"] + " " + (f.get("recommendation") or ""), ref) >= 0.25:
                out.append(d["id"])
    return out


def build_plan(v, decisions: list[dict]) -> list[dict]:
    """Одно действие на канонический дефект (кроме вынесенных в решения). Строки плана группируют действия по
    требованию только для чтения: у каждого действия своя ссылка на дефект и на решение, от которого оно зависит."""
    in_dec = {x for d in decisions for x in d["findings"]}
    actions = []
    for f in v.all_f:
        if f["id"] in in_dec:
            continue
        dep = _depends(v, f, decisions)
        text = v.clean(f.get("recommendation") or "") or "Устранить: " + v.clean(f["problem"])
        if dep:
            text = f"После решения {', '.join(dep)}: {text}"
        actions.append({"id": "", "finding": f["id"], "severity": f["severity"], "role": role(v, f),
                        "text": text, "depends_on": dep, "closes": "partial" if dep else "full",
                        "requirements": _reqs_of(f)})
    actions.sort(key=lambda a: ([k for k, _, _ in ROLES].index(a["role"]), SEV_ORDER.get(a["severity"], 3),
                                a["finding"]))
    for n, a in enumerate(actions, 1):
        a["id"] = f"A-{n:02d}"
    return actions


def build_questions(v, decisions: list[dict]) -> list[dict]:
    qs = [(r["id"], q) for r in v.reqs for q in r.get("questions") or []] + \
         [(f"EC-{i:02d}", e["question"]) for i, e in enumerate(v.d.get("edge_cases") or [], 1) if e.get("question")]
    out: list[dict] = []
    for src, q in qs:
        host = next((u for u in out if similarity(u["text"], q) >= 0.6), None)
        if host is None:
            out.append({"id": "", "text": q, "sources": [src], "decisions": []})
        elif src not in host["sources"]:
            host["sources"].append(src)
    for n, q in enumerate(out, 1):
        q["id"] = f"Q-{n:02d}"
        reqs = {s for s in q["sources"] if s in v.req_by}
        for d in decisions:
            if d["kind"] == "business" and reqs & set(d["requirements"]) and similarity(
                    q["text"], d["question"] + " " + " ".join(d["options"])) >= 0.25:
                q["decisions"].append(d["id"])
    return out


def build_tests(v, decisions: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in v.reqs:
        for t in r.get("test_ideas") or []:
            host = next((u for u in out if similarity(u["text"], t) >= 0.7), None)
            if host is None:
                out.append({"id": "", "text": t, "requirements": [r["id"]], "depends_on": []})
            elif r["id"] not in host["requirements"]:
                host["requirements"].append(r["id"])
    for n, t in enumerate(out, 1):
        t["id"] = f"TC-{n:02d}"
        t["depends_on"] = [d["id"] for d in decisions
                           if d["kind"] == "business" and set(t["requirements"]) & set(d["requirements"])]
        t["conditional"] = bool(t["depends_on"])
        t["ready"] = not t["depends_on"] or bool(_CONDITIONAL.search(t["text"]))
    return out


def recompute_stats(v) -> dict:
    s = dict(v.d.get("stats") or {})
    defects = v.all_f
    s["findings"] = len(defects)
    s["req_findings"] = len(v.req_f)
    s["set_findings"] = len(v.set_f)
    s["by_severity"] = {k.lower(): sum(1 for f in defects if f["severity"] == k) for k in SEV_ORDER}
    s["occurrences"] = sum(1 for _ in v.req_f) + sum(max(1, int(f.get("places") or 1)) for f in v.set_f)
    affected = {r["id"] for r in v.reqs if r.get("findings") or r.get("systemic_defects")}
    s["with_issues"] = len(affected)
    s["without_issues"] = len(v.reqs) - len(affected)
    s["affected_reqs"] = len(affected)
    s["clarifications"] = len(v.d.get("questions") or [])
    s["recommendations"] = len(v.d.get("recommendations") or [])
    s["decisions"] = len(v.d.get("decisions") or [])
    s["actions"] = len(v.d.get("plan") or [])
    s["observations"] = len(defects) + sum(int(f.get("merged_duplicates") or 0) for f in defects if
                                          int(f.get("merged_duplicates") or 0) > 0) + len(v.d.get("dropped") or [])
    s["rejected"] = len(v.d.get("dropped") or [])
    return s


# ====================================================================== проверки перед выводом
def validate(v, doc=None) -> dict:
    checks = []

    def check(cid, title, bad: list, note=""):
        checks.append({"id": cid, "type": "детерминированная", "title": title, "ok": not bad,
                       "problems": bad[:8], "count": len(bad), "note": note})
    ids = [f["id"] for f in v.all_f]
    check("V01", "Уникальные ID дефектов, допустимая критичность и классификация",
          [i for i in set(ids) if ids.count(i) > 1] +
          [f["id"] for f in v.all_f if f["severity"] not in SEV_ORDER] +
          [f["id"] for f in v.all_f if (f.get("classification") or "defect") != "defect"])
    known = set(ids)
    dangling = [f"{r['id']}→{s}" for r in v.reqs for s in r.get("systemic_defects") or [] if s not in known]
    dangling += [f"{r['id']}→{fid}" for r in v.reqs for fid, _ in r.get("related") or [] if fid not in known]
    dangling += [f"{d['id']}→{x}" for d in v.d.get("decisions") or [] for x in d["findings"] if x not in known]
    dangling += [f"{a['id']}→{a['finding']}" for a in v.d.get("plan") or [] if a["finding"] not in known]
    check("V02", "Ссылки на дефекты из требований, решений и плана существуют", dangling)
    if doc is not None:
        bad_q = []
        for f in v.all_f:
            for q in f.get("quotes") or []:
                if q and len(q) >= 3 and doc.find(q.strip("«» …")) is None and f.get("quote_verified", True):
                    bad_q.append(f["id"])
                    break
        check("V03", "Цитаты дефектов совпадают с текстом документа", bad_q)
        check("V04", "Отчёт построен по тому же документу (SHA-256)",
              [] if (v.d.get("document") or {}).get("sha256") in (None, doc.sha256) else ["sha256"])
    st = v.stats
    bs = st.get("by_severity") or {}
    mism = []
    if st.get("findings") != len(v.all_f):
        mism.append(f"всего {st.get('findings')} ≠ {len(v.all_f)}")
    if sum(bs.values()) != len(v.all_f):
        mism.append("сумма по критичности ≠ всего")
    check("V05", "Счётчики сводки совпадают с реестром дефектов", mism)
    in_dec = {x for d in v.d.get("decisions") or [] for x in d["findings"]}
    in_plan = {a["finding"] for a in v.d.get("plan") or []}
    check("V06", "У каждого дефекта есть путь исправления (решение или действие)",
          [f["id"] for f in v.all_f if f["id"] not in in_dec and f["id"] not in in_plan])
    check("V07", "Открытые решения не помечены решёнными без источника",
          [d["id"] for d in v.d.get("decisions") or [] if d.get("status") == "resolved" and not d.get("authority")])
    check("V08", "Тест, зависящий от открытого решения, помечен условным",
          [t["id"] for t in v.d.get("tests") or [] if t["depends_on"] and not t.get("conditional")])
    marks = []
    for r in v.reqs:
        cr, cm = r.get("criteria") or {}, r.get("criteria_comments") or {}
        own = {c for f in r.get("findings") or [] for c in (f.get("criteria") or [f.get("criterion")])}
        for c, stt in cr.items():
            if stt in ("warn", "fail") and _POSITIVE.search(cm.get(c, "")) and c not in own:
                marks.append(f"{r['id']}:{c}")
        if r.get("related") and cr.get("CON") == "ok":
            marks.append(f"{r['id']}:CON")
    check("V09", "Отметки критериев не противоречат комментариям и связанным противоречиям", marks)
    check("V10", "Действие, зависящее от открытого решения, не закрывает дефект полностью",
          [a["id"] for a in v.d.get("plan") or [] if a["depends_on"] and a["closes"] == "full"])
    failed = [c for c in checks if not c["ok"]]
    return {"status": "passed" if not failed else "failed", "checks": checks,
            "failed": [c["id"] for c in failed],
            "semantic": "Смысловая проверка (ложные срабатывания, составные дефекты, шкала критичности, рекомендации, "
                        "выбирающие сторону в открытом решении) выполняется вторым проходом той же модели — это "
                        "дополнительная проверка, а не независимая валидация."}


def _repair(v) -> None:
    """Ограниченный ремонт: висячие ссылки убираются, отметки с положительным комментарием возвращаются в «ok»."""
    known = {f["id"] for f in v.all_f}
    for r in v.reqs:
        r["systemic_defects"] = [s for s in r.get("systemic_defects") or [] if s in known]
        r["related"] = [x for x in r.get("related") or [] if x[0] in known]
        cr, cm = r.get("criteria") or {}, r.get("criteria_comments") or {}
        own = {c for f in r.get("findings") or [] for c in (f.get("criteria") or [f.get("criterion")])}
        for c, stt in list(cr.items()):
            if stt in ("warn", "fail") and _POSITIVE.search(cm.get(c, "")) and c not in own:
                cr[c] = "ok"
        if r.get("related"):
            cr["CON"] = "fail" if cr.get("CON") in ("ok", "na", None) else cr["CON"]


def finalize(data: dict, doc=None, repair_budget: int = 2) -> dict:
    """Строит записи для всех представлений, пересчитывает статистику и проверяет согласованность.
    При ошибках — ограниченный ремонт и повторная проверка; если не помогло — отчёт помечается как
    не прошедший проверку (без подавления дефектов и без выдуманных решений)."""
    from .present import View
    if data.get("findings") is None and not data.get("requirements"):
        return data                                       # диагностический отчёт
    from .verify import FORM_TAGS
    for f in [f for r in data.get("requirements") or [] for f in r.get("findings") or []] + \
            list(data.get("set_findings") or []):
        if f.get("severity") == "CRITICAL" and f.get("tag") in FORM_TAGS:      # шкала: пустой раздел — MAJOR
            f["severity"] = "MAJOR"
            f["caveat"] = ("Критичность приведена к MAJOR по шкале: незаполненный раздел или оформление — MAJOR, "
                           "отдельный блокирующий эффект не установлен.")
    for _ in range(repair_budget + 1):
        v = View(data)
        data["decisions"] = build_decisions(v)
        v = View(data)
        data["plan"] = build_plan(v, data["decisions"])
        data["questions"] = build_questions(v, data["decisions"])
        data["tests"] = build_tests(v, data["decisions"])
        data["stats"] = recompute_stats(View(data))
        val = validate(View(data), doc)
        if val["status"] == "passed":
            break
        _repair(View(data))
    data["validation"] = val
    rs = data.get("run_status") or {}
    if val["status"] != "passed" and rs.get("code") == "completed":
        rs["code"], rs["label"] = "partial", "Частичная"
        rs.setdefault("reasons", []).append(
            "Отчёт не прошёл проверки согласованности: " + ", ".join(val["failed"]) + " (см. журнал проверки).")
        data["run_status"] = rs
    return data
