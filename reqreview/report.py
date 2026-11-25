"""Формирование отчёта на русском языке (Markdown) и машиночитаемой копии (JSON)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from .agent import Finding, Result
from .criteria import INDIVIDUAL, SET, SEVERITY, SEVERITY_ORDER
from .util import md_cell, short

MARK = {"ok": "✅", "fail": "❌", "na": "—"}
SET_MARK = {"ok": "✅ выполнен", "partial": "⚠️ частично", "fail": "❌ не выполнен", "n/a": "— не оценивался"}
SEV_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}

# критичные дефекты по брифу: непроверяемые, противоречивые, неполные
CRITICAL_CLASSES = {"VER": "непроверяемость", "COM": "неполнота", "S-CONS": "противоречие/несогласованность",
                    "S-COMPL": "неполнота набора", "COR": "некорректность", "FEA": "невыполнимость"}


def _loc(res: Result, f: Finding) -> str:
    if f.g1 is None:
        return "—"
    locs = getattr(f, "_locs", None)
    if locs and len(locs) > 1:
        return "; ".join(res.doc.location(a, b) for a, b in locs)
    return res.doc.location(f.g1, f.g2)


def _req_ref(res: Result, rid: str) -> str:
    r = next((x for x in res.requirements if x.id == rid), None)
    if not r:
        return rid
    return f"{rid}" + (f" ({r.doc_id})" if r.doc_id else "")


def _is_critical(f: Finding) -> bool:
    return f.severity == "critical"


def render(res: Result, agent_version: str, input_display: str) -> str:
    d, s = res.doc, res.stats
    now = dt.datetime.fromtimestamp(res.finished).astimezone()
    out: list[str] = []
    w = out.append

    w(f"# Отчёт о статической проверке требований\n")
    w(f"**Документ:** «{d.title}»  ")
    w(f"**Дата проверки:** {now:%d.%m.%Y %H:%M} ({now:%Z})\n")
    if res.stats.get("mock"):
        w("> ⚠️ **ДЕМО-РЕЖИМ.** Отчёт сформирован без языковой модели, только по правилам. "
          "Он показывает, что агент работает, но не является полноценной проверкой.\n")

    # ---------------------------------------------------------------- 1
    w("## 1. Паспорт прогона\n")
    w("| Параметр | Значение |\n|---|---|")
    rows = [
        ("Название документа", d.title),
        ("Файл", input_display),
        ("Формат", d.fmt),
        ("Контрольная сумма SHA-256", f"`{d.sha256[:16]}…`"),
        ("Дата и время прогона", f"{now:%Y-%m-%d %H:%M:%S %z}"),
        ("Длительность", f"{res.finished - res.started:.0f} с"),
        ("Модель", res.model_name),
        ("Обращений к модели", str(res.llm_calls)),
        ("Версия агента", agent_version),
        ("Критерии", "ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6; IEEE Std 830-1998 п. 4.3"),
    ]
    for k, v in rows:
        w(f"| {k} | {md_cell(v)} |")
    w("")
    w("Обозначения мест: «строка N» — номер строки в исходном файле"
      + (" (для PDF — «с. N, строка M» — страница и строка на странице)" if d.fmt == "PDF" else "")
      + "; R-xxx — идентификатор требования в отчёте; F-xxx — замечание к требованию; S-xx — замечание к набору.\n")

    # ---------------------------------------------------------------- 2
    w("## 2. Критерии проверки\n")
    w("**Каждое требование** (ISO/IEC/IEEE 29148 §5.2.5, IEEE 830 §4.3):\n")
    w("| Код | Критерий | Источник |\n|---|---|---|")
    for c, (name, src, _) in INDIVIDUAL.items():
        w(f"| {c} | {name} | {src} |")
    w("\n**Набор требований** (ISO/IEC/IEEE 29148 §5.2.6, IEEE 830 §4.3):\n")
    w("| Код | Критерий | Источник |\n|---|---|---|")
    for c, (name, src, _) in SET.items():
        w(f"| {c} | {name} | {src} |")
    w("\nСерьёзность: 🔴 критический — непроверяемое, противоречивое или неполное требование, по которому нельзя "
      "разрабатывать/тестировать; 🟠 существенный; 🟡 незначительный.\n")

    # ---------------------------------------------------------------- 3
    w("## 3. Как агент выделил требования\n")
    no_ids = sum(1 for r in res.requirements if not r.doc_id)
    w(f"Выделено требований: **{len(res.requirements)}**"
      + (f"; у {no_ids} нет идентификатора в документе — агент присвоил свой (R-xxx), цитата сохранена в разделе 4." if no_ids else ".")
      + "\n")
    w("| ID в отчёте | ID в документе | Место в документе | Краткая формулировка |\n|---|---|---|---|")
    for r in res.requirements:
        w(f"| {r.id} | {md_cell(r.doc_id or '— (нет)')} | {md_cell(d.location(r.g1, r.g2))} | {md_cell(short(r.summary, 100))} |")
    w("")

    # ---------------------------------------------------------------- 4
    w("## 4. Проверка каждого требования\n")
    w("### 4.1. Сводная матрица\n")
    w("✅ — критерий выполнен, ❌ — нарушен, — — неприменим/не оценён.\n")
    w("| Требование | " + " | ".join(INDIVIDUAL) + " | Замечаний |")
    w("|---|" + "---|" * (len(INDIVIDUAL) + 1))
    for r in res.requirements:
        w(f"| {r.id} | " + " | ".join(MARK.get(r.criteria.get(c, "na"), "—") for c in INDIVIDUAL)
          + f" | {len(r.findings)} |")
    w("")
    w("### 4.2. Замечания по требованиям\n")
    for r in res.requirements:
        title = f"#### {r.id}" + (f" ({r.doc_id})" if r.doc_id else "") + f" — {md_cell(short(r.summary, 80))}"
        w(title + "\n")
        w(f"*Место:* {d.location(r.g1, r.g2)}  ")
        w(f"*Тип:* {r.kind}  ")
        w(f"*Цитата:* «{r.text}»\n")
        set_only = [c for c in INDIVIDUAL if r.criteria.get(c) == "fail" and not any(f.criterion == c for f in r.findings)]
        if set_only:
            refs = [f.fid for f in res.set_findings if r.id in f.req_ids]
            w(f"- Нарушен критерий {', '.join(set_only)} — общий дефект оформления, описан в замечании к набору"
              + (f" {', '.join(refs)}" if refs else " (раздел 5.2)") + ".")
        if not r.findings:
            if not set_only:
                w("Нарушений не выявлено.\n" if r.reviewed else "⚠️ Модель не вернула результат; нарушений по правилам не выявлено.\n")
            else:
                w("")
            continue
        for f in r.findings:
            name = INDIVIDUAL[f.criterion][0]
            w(f"- {SEV_ICON[f.severity]} **{f.fid}** · {f.criterion} {name} · {SEVERITY[f.severity]}"
              + ("" if f.source == "LLM" else " · *автоматическое правило*"))
            if f.quote and f.quote != r.text:
                w(f"  - Где: {_loc(res, f)} — «{f.quote}»")
            else:
                w(f"  - Где: {_loc(res, f)} — вся формулировка требования")
            if not f.verified:
                w("  - *Цитата модели не найдена в тексте дословно; замечание отнесено ко всей формулировке.*")
            w(f"  - Проблема: {f.problem}")
            if f.recommendation:
                w(f"  - Рекомендация: {f.recommendation}")
        w("")

    # ---------------------------------------------------------------- 5
    w("## 5. Проверка набора требований\n")
    w("### 5.1. Оценка по критериям набора\n")
    w("| Код | Критерий | Оценка | Комментарий |\n|---|---|---|---|")
    for c, v in res.set_criteria.items():
        w(f"| {c} | {SET[c][0]} | {SET_MARK.get(v['status'], v['status'])} | {md_cell(v['comment'] or '—')} |")
    w("")
    w("### 5.2. Замечания к набору\n")
    if not res.set_findings:
        w("Замечаний к набору не выявлено.\n")
    for f in res.set_findings:
        w(f"- {SEV_ICON[f.severity]} **{f.fid}** · {f.type or '—'} · {f.criterion} {SET[f.criterion][0]} · "
          f"{SEVERITY[f.severity]}" + ("" if f.source == "LLM" else " · *автоматическое правило*"))
        if f.req_ids:
            w(f"  - Требования: {', '.join(_req_ref(res, i) for i in f.req_ids)}")
        w(f"  - Где: {_loc(res, f)}" + (f" — {f.quote}" if f.quote else ""))
        w(f"  - Проблема: {f.problem}")
        if f.recommendation:
            w(f"  - Рекомендация: {f.recommendation}")
    w("")

    # ---------------------------------------------------------------- 6
    all_f = [f for r in res.requirements for f in r.findings] + res.set_findings
    crit = sorted([f for f in all_f if _is_critical(f)], key=lambda f: (f.fid[0] != "S", f.fid))
    w("## 6. Сводка\n")
    w("| Показатель | Значение |\n|---|---|")
    w(f"| Требований всего | {s['requirements']} |")
    w(f"| Требований без нарушений | {s['without_issues']} |")
    w(f"| Требований с нарушениями | {s['with_issues']} |")
    w(f"| Нарушений всего | {s['findings']} (к требованиям — {s['req_findings']}, к набору — {s['set_findings']}) |")
    for sev, label in SEVERITY.items():
        w(f"| — {SEV_ICON[sev]} {label.lower()} | {s['by_severity'][sev]} |")
    w("")
    w("**Нарушения по критериям (отдельные требования):**\n")
    w("| Критерий | Нарушений |\n|---|---|")
    for c, n in sorted(s["by_criterion"].items(), key=lambda x: -x[1]):
        if n:
            w(f"| {c} {INDIVIDUAL[c][0]} | {n} |")
    w("")
    w(f"**Критичные дефекты ({len(crit)})** — непроверяемые, противоречивые, неполные формулировки:\n")
    if not crit:
        w("Критичных дефектов не выявлено.\n")
    else:
        w("| № | Требование | Место | Вид | Суть |\n|---|---|---|---|---|")
        for f in crit:
            reqs = ", ".join(_req_ref(res, i) for i in f.req_ids) or "—"
            kind = CRITICAL_CLASSES.get(f.criterion) or (INDIVIDUAL.get(f.criterion) or SET.get(f.criterion))[0]
            w(f"| {f.fid} | {md_cell(reqs)} | {md_cell(_loc(res, f))} | {md_cell(kind)} | {md_cell(short(f.problem, 180))} |")
        w("")

    # ---------------------------------------------------------------- 7
    w("## 7. Итог\n")
    n_crit, n_major = s["by_severity"]["critical"], s["by_severity"]["major"]
    if not res.requirements:
        verdict = "❌ **Нет.** В документе не выделено ни одного требования — начинать разработку по нему нельзя."
    elif n_crit:
        verdict = (f"❌ **Нет, начинать разработку и тестирование по документу в текущем виде не рекомендуется.** "
                   f"Найдено {n_crit} критичных дефектов: по таким требованиям нельзя однозначно реализовать "
                   f"функциональность и составить тесты с объективным критерием «пройден/не пройден».")
    elif n_major:
        verdict = (f"⚠️ **Условно.** Критичных дефектов нет, но {n_major} существенных замечаний нужно устранить "
                   "до начала разработки затронутых функций; остальное можно начинать.")
    else:
        verdict = "✅ **Да.** Существенных дефектов не выявлено; незначительные замечания можно исправить по ходу."
    w(verdict + "\n")
    w("**Что исправить в первую очередь:**\n")
    prio = sorted(all_f, key=lambda f: (SEVERITY_ORDER[f.severity], f.fid[0] != "S", f.fid))
    prio = [f for f in prio if f.severity in ("critical", "major")]
    groups: dict[tuple, list[Finding]] = {}
    for f in prio:
        key = tuple(f.req_ids) if f.fid.startswith("F") else (f.fid,)
        groups.setdefault(key, []).append(f)
    if not groups:
        w("Обязательных исправлений нет.\n")
    for i, fs in enumerate(list(groups.values())[:12], 1):
        f = fs[0]
        reqs = ", ".join(_req_ref(res, x) for x in f.req_ids)
        more = (f" Также: {', '.join(x.fid for x in fs[1:])}." if len(fs) > 1 else "")
        w(f"{i}. {SEV_ICON[f.severity]} **{reqs or f.fid}** ({_loc(res, f)}; {f.fid}) — {f.problem}"
          + (f" → {f.recommendation}" if f.recommendation else "") + more)
    if len(groups) > 12:
        w(f"\nОстальные {len(groups) - 12} позиций — см. разделы 4.2 и 5.2.")
    w("")

    # ---------------------------------------------------------------- прил.
    w("## Приложение. Ограничения проверки\n")
    w("- Проверка статическая: оценивается текст требований, программа не запускается.")
    w("- Все цитаты автоматически сверены с текстом документа; замечания модели без подтверждаемой цитаты "
      "или ссылки на требование в отчёт не включаются.")
    w(f"- Замечаний модели, привязанных ко всей формулировке из-за неточной цитаты: {s['unverified']}.")
    for wn in dict.fromkeys(res.warnings):
        w(f"- {wn}")
    w("")
    w(f"*Отчёт сформирован автоматически агентом req-review-agent {agent_version}.*")
    return "\n".join(out) + "\n"


def to_json(res: Result, agent_version: str) -> str:
    def fd(f: Finding):
        return {"id": f.fid, "criterion": f.criterion, "severity": f.severity, "type": f.type,
                "requirements": f.req_ids, "location": _loc(res, f), "quote": f.quote,
                "quote_verified": f.verified, "problem": f.problem, "recommendation": f.recommendation,
                "source": f.source}
    data = {
        "document": {"title": res.doc.title, "file": res.doc.path.name, "sha256": res.doc.sha256},
        "model": res.model_name, "agent_version": agent_version,
        "date": dt.datetime.fromtimestamp(res.finished).astimezone().isoformat(timespec="seconds"),
        "stats": res.stats,
        "requirements": [{"id": r.id, "doc_id": r.doc_id, "location": res.doc.location(r.g1, r.g2),
                          "text": r.text, "summary": r.summary, "kind": r.kind, "criteria": r.criteria,
                          "findings": [fd(f) for f in r.findings]} for r in res.requirements],
        "set_criteria": {c: {"status": v["status"], "comment": v["comment"]} for c, v in res.set_criteria.items()},
        "set_findings": [fd(f) for f in res.set_findings],
        "warnings": list(dict.fromkeys(res.warnings)),
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def write(res: Result, out_path: str | Path, agent_version: str, input_display: str) -> list[Path]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res, agent_version, input_display), encoding="utf-8")
    js = out.with_suffix(".data.json" if out.suffix.lower() == ".json" else ".json")
    js.write_text(to_json(res, agent_version), encoding="utf-8")
    return [out, js]
