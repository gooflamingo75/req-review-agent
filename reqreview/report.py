"""Формирование отчёта на русском языке (Markdown) и машиночитаемой копии (JSON).

Структура отчёта соответствует шаблону организаторов (REPORT_TEMPLATE):
1. Паспорт прогона · 2. Нарезка и инвентаризация требований · 3. Детальная проверка каждого требования ·
4. Проверка набора требований в целом · 5. Сводная статистика · 6. Итоговый вердикт и рекомендации.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

from .agent import Finding, Requirement, Result
from .criteria import GUIDE_CORE, INDIVIDUAL, SET, SEVERITY, SEVERITY_ORDER, SEVERITY_RU, CRITICAL_HINT
from .util import md_cell, short

MARK = {"ok": "✅", "warn": "⚠️", "fail": "❌", "na": "—"}
STATUS_TEXT = {"ok": "✅ Без нарушений.", "warn": "⚠️ *Замечание.*", "fail": "❌ *Нарушение.*", "na": "— Неприменимо."}
SET_MARK = {"ok": "✅ выполнен", "partial": "⚠️ частично", "fail": "❌ не выполнен", "n/a": "— не оценивался"}
SEV_ICON = {"critical": "🔴", "major": "🟡", "minor": "🟢"}

SET_SECTIONS = [
    ("4.1. Системная полнота (Completeness of the Set)", ["S-COMPL", "S-RANK"],
     "Описан ли полный жизненный цикл функции от триггера до финального состояния, позитивные и негативные "
     "сценарии; все ли разделы шаблона заполнены?", "Найденные пробелы"),
    ("4.2. Взаимная непротиворечивость (Internal Consistency)", ["S-CONS", "S-COMP"],
     "Нет ли взаимоисключающих условий между разделами и связанными частями документа, расхождений единиц, "
     "типов данных и терминов?", "Найденные противоречия"),
    ("4.3. Архитектурная осуществимость и стандарты (Feasibility & Standards)", ["S-FEAS", "S-BND"],
     "Соответствие протоколам (HTTP-методы и коды ответов), паттернам безопасности, идемпотентности; "
     "границы системы.", "Найденные нарушения"),
    ("4.4. Прослеживаемость и модифицируемость (Traceability)", ["S-TRC", "S-MOD"],
     "Есть ли уникальные идентификаторы, версионность; прослеживается ли цепочка Бизнес-цель → UI/UX → "
     "Функциональное требование → API/Модель данных → Критерии приёмки?", "Найденные нарушения"),
]


def plural(n: int, one: str, few: str, many: str) -> str:
    n10, n100 = n % 10, n % 100
    if n10 == 1 and n100 != 11:
        return one
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return few
    return many


class _R:
    """Вспомогательные функции форматирования, привязанные к результату."""

    def __init__(self, res: Result):
        self.res = res
        self.doc = res.doc
        self.by_id = {r.id: r for r in res.requirements}

    def loc(self, f: Finding) -> str:
        if f.g1 is None:
            return "—"
        if f.locs and len(f.locs) > 1:
            return "; ".join(self.doc.location(a, b) for a, b in f.locs)
        return self.doc.location(f.g1, f.g2)

    def ref(self, rid: str) -> str:
        r = self.by_id.get(rid)
        return f"{rid}" + (f" ({r.doc_id})" if r and r.doc_id else "")

    def refs(self, ids) -> str:
        """Список требований; длинные последовательности сворачиваются в диапазоны (REQ-01–REQ-04)."""
        ids = list(ids)
        if len(ids) <= 6:
            return ", ".join(self.ref(i) for i in ids) or "—"
        order = {r.id: n for n, r in enumerate(self.res.requirements)}
        known = sorted((i for i in ids if i in order), key=order.get)
        parts, run = [], []
        for i in known:
            if run and order[i] == order[run[-1]] + 1:
                run.append(i)
            else:
                if run:
                    parts.append(run)
                run = [i]
        if run:
            parts.append(run)
        out = [f"{p[0]}–{p[-1]}" if len(p) > 2 else ", ".join(p) for p in parts]
        return ", ".join(out + [i for i in ids if i not in order]) + f" (всего {len(ids)})"

    def label(self, f: Finding) -> str:
        part = self.doc.part_of(f.g1) if f.g1 is not None else ""
        bits = [b for b in (part, f.tag) if b]
        return f"[{' / '.join(bits)}]" if bits else ""

    def where(self, f: Finding) -> list[str]:
        """Список «место — цитата» для замечания."""
        if f.locs and f.quotes and len(f.locs) == len(f.quotes):
            return [f"{self.doc.location(a, b)} — «{q}»" for (a, b), q in zip(f.locs, f.quotes)]
        if f.quote:
            return [f"{self.loc(f)} — {f.quote if f.quote.startswith('«') else '«' + f.quote + '»'}"]
        return [self.loc(f)]


def _finding_block(R: _R, f: Finding, w, with_reqs: bool = True) -> None:
    crit = INDIVIDUAL.get(f.criterion) or SET.get(f.criterion)
    head = (f"- `[{SEVERITY[f.severity]}]` {SEV_ICON[f.severity]} **{f.fid}** {R.label(f)} · "
            f"{f.type + ' · ' if f.type else ''}{f.criterion} {crit[0]}"
            + ("" if f.source == "LLM" else " · *автоматическая проверка*")
            + (f" · **мест: {len(f.locs)}**" if len(f.locs) > 1 else ""))
    w(head)
    if with_reqs and f.req_ids:
        w(f"  - **Требования:** {R.refs(f.req_ids)}")
    places = R.where(f)
    if len(places) == 1:
        w(f"  - **Где:** {places[0]}")
    else:
        w("  - **Где:**")
        for p in places:
            w(f"    - {p}")
    if not f.verified and f.source == "LLM":
        w("  - *Цитата модели не найдена в тексте дословно; замечание отнесено ко всей формулировке требования.*")
    w(f"  - **Проблема:** {f.problem}")
    if f.also:
        w("  - **Также проявляется как:**")
        for a in f.also[:6]:
            w(f"    - {short(a, 220)}")
    if f.recommendation:
        w(f"  - **Рекомендация:** {f.recommendation}")


def render(res: Result, agent_version: str, input_display: str) -> str:
    R = _R(res)
    d, s = res.doc, res.stats
    now = dt.datetime.fromtimestamp(res.finished).astimezone()
    out: list[str] = []
    w = out.append
    reqs = res.requirements
    all_f = [f for r in reqs for f in r.findings] + res.set_findings

    w("# Отчёт о статической проверке требований\n")
    w(f"**Документ:** «{d.title}» · **Дата проверки:** {now:%d.%m.%Y %H:%M}\n")
    if s.get("mock"):
        w("> ⚠️ **ДЕМО-РЕЖИМ.** Отчёт сформирован без языковой модели, только автоматическими правилами. "
          "Он показывает, что агент работает, но не является полноценной проверкой.\n")

    # ------------------------------------------------------------ 1
    w("## 1. Паспорт прогона\n")
    parts = sorted({r.part for r in reqs if r.part})
    rows = [
        ("Название проверяемого документа", f"`{Path(input_display).name}` — «{d.title}»"),
        ("Путь к файлу / формат", f"`{input_display}` · {d.fmt}"),
        ("Дата и время прогона", f"{now:%Y-%m-%d %H:%M:%S} (UTC{now:%z})"),
        ("Используемая LLM модель", res.model_name),
        ("Версия агента", agent_version),
        ("Время анализа", f"{res.finished - res.started:.1f} сек · обращений к модели: {res.llm_calls}"),
        ("Контрольная сумма документа (SHA-256)", f"`{d.sha256}`"),
        ("Структура документа", (", ".join(parts) + "; " if parts else "") + f"{len(d.nonempty())} непустых строк"),
        ("Стандарты и критерии", "ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6; IEEE Std 830-1998 п. 4.3; "
                                 "руководство организаторов по критериям (см. Приложение А)"),
    ]
    for k, v in rows:
        w(f"* **{k}:** {v}")
    w("\n*Обозначения:* REQ-xx — требование в отчёте; F-xxx — дефект отдельного требования; S-xx — дефект набора; "
      "«строка N» — номер строки в исходном файле" + (" (для PDF — «с. N, строка M»)" if d.fmt == "PDF" else "")
      + "; `[CRITICAL]` 🔴 блокер, `[MAJOR]` 🟡 существенный, `[MINOR]` 🟢 замечание.\n")
    w("---\n")

    # ------------------------------------------------------------ 2
    w("## 2. Нарезка и инвентаризация требований\n")
    no_ids = sum(1 for r in reqs if not r.doc_id)
    kinds = Counter(r.kind for r in reqs)
    w(f"Агент извлёк **{len(reqs)}** элементарных требований"
      + (f"; у {no_ids} нет идентификатора в документе — агент присвоил свой (REQ-xx) и сохранил точную цитату"
         if no_ids else "") + ".")
    if kinds:
        w("По типам: " + ", ".join(f"{k} — {n}" for k, n in kinds.most_common()) + ".\n")
    w("| ID в отчёте | ID в документе | Исходный раздел документа | Место | Краткая формулировка / Точная цитата |")
    w("| :--- | :--- | :--- | :--- | :--- |")
    for r in reqs:
        sec = " / ".join(x for x in (r.part, r.section) if x and x != r.part) or r.part or "—"
        if r.part and r.section and r.section != r.part:
            sec = f"{r.part} / {r.section}"
        loc = d.location(r.g1, r.g2).split(", раздел")[0]
        w(f"| `{r.id}` | `{md_cell(r.doc_id or 'Без номера')}` | {md_cell(sec)} | {md_cell(loc)} | "
          f"**{md_cell(short(r.summary, 90))}** — «{md_cell(r.text)}» |")
    w("\n---\n")

    # ------------------------------------------------------------ 3
    w("## 3. Детальная проверка каждого требования\n")
    w("### 3.0. Сводная матрица по критериям\n")
    w("Первые шесть столбцов — основные критерии руководства (Полнота, Однозначность, Непротиворечивость, "
      "Тестируемость, Реализуемость, Необходимость), далее — дополнительные характеристики ISO/IEC/IEEE 29148. "
      "✅ — выполнен, ⚠️ — замечание, ❌ — нарушение, — — неприменимо.\n")
    codes = list(INDIVIDUAL)
    w("| Требование | " + " | ".join(codes) + " | Дефектов (+ системные) | Макс. критичность |")
    w("| :--- |" + " :---: |" * (len(codes) + 2))
    for r in reqs:
        worst = min((f.severity for f in list(r.findings) + list(r.linked)), key=lambda x: SEVERITY_ORDER[x], default=None)
        cnt = str(len(r.findings)) + (f" + {', '.join(m.fid for m in r.linked)}" if r.linked else "")
        w(f"| {r.id} | " + " | ".join(MARK.get(r.criteria.get(c, "na"), "—") for c in codes)
          + f" | {cnt} | {SEV_ICON[worst] + ' ' + SEVERITY[worst] if worst else '—'} |")
    w("\nРасшифровка кодов: " + "; ".join(f"**{c}** — {INDIVIDUAL[c][0]}" for c in codes) + ".\n")

    for r in reqs:
        _req_block(R, r, w)
    w("---\n")

    # ------------------------------------------------------------ 4
    w("## 4. Проверка набора требований в целом (спецификация как система)\n")
    for title, codes_, question, found_title in SET_SECTIONS:
        w(f"### {title}\n")
        w(f"* **Анализ:** {question}")
        for c in codes_:
            v = res.set_criteria.get(c, {})
            w(f"* **{SET[c][0]}** ({SET[c][1]}): {SET_MARK.get(v.get('status', 'n/a'), v.get('status'))}"
              + (f" — {v['comment']}" if v.get("comment") else ""))
        fs = [f for f in res.set_findings if f.criterion in codes_]
        w(f"* **{found_title}:**" + (" не выявлено.\n" if not fs else "\n"))
        for f in fs:
            _finding_block(R, f, w)
        if fs:
            w("")
        if title.startswith("4.4") and res.traceability:
            w("**Матрица прослеживаемости** (Бизнес-цель → UI/UX → Требование → API/Данные → Критерии приёмки):\n")
            w("| Требование | Бизнес-цель | UI/UX | API / модель данных | Критерии приёмки | Разрывы цепочки |")
            w("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in res.traceability:
                cell = lambda k: ", ".join(t[k]) if t[k] else "❌ нет"  # noqa: E731
                w(f"| {R.ref(t['req'])} | {cell('goal')} | {cell('ui')} | {cell('api_data')} | {cell('acceptance')} | "
                  f"{md_cell(t['gap'] or '—')} |")
            w("")
    w("### 4.5. Непокрытые пограничные сценарии (Edge cases)\n")
    if not res.edge_cases:
        w("Агент не выделил дополнительных пограничных сценариев.\n")
    else:
        w("Сценарии, поведение в которых документ не определяет. Каждый — вопрос к аналитику до начала спринта.\n")
        w("| № | Сценарий | Связанные требования | Риск | Вопрос к аналитику |")
        w("| :--- | :--- | :--- | :--- | :--- |")
        for i, e in enumerate(res.edge_cases, 1):
            w(f"| EC-{i:02d} | {md_cell(e['scenario'])} | {md_cell(R.refs(e['req_ids']))} | {md_cell(e['risk'] or '—')} | "
              f"{md_cell(e['question'] or '—')} |")
        w("")
    w("### 4.6. Оценка по критериям набора (ISO/IEC/IEEE 29148 §5.2.6, IEEE 830 §4.3)\n")
    w("| Код | Критерий | Источник | Оценка | Дефекты |")
    w("| :--- | :--- | :--- | :--- | :--- |")
    for c, v in res.set_criteria.items():
        w(f"| {c} | {SET[c][0]} | {SET[c][1]} | {SET_MARK.get(v['status'], v['status'])} | "
          f"{', '.join(f.fid for f in v['findings']) or '—'} |")
    w("\n---\n")

    # ------------------------------------------------------------ 5
    w("## 5. Сводная статистика проведённого аудита\n")
    w("| Метрика | Значение |\n| :--- | :--- |")
    w(f"| **Всего выделено требований:** | {s['requirements']} |")
    w(f"| **Требований без дефектов:** | {s['without_issues']} |")
    w(f"| **Требований с дефектами:** | {s['with_issues']} |")
    w(f"| **Всего уникальных дефектов:** | {s['findings']} (в отдельных требованиях — {s['req_findings']}, "
      f"системных/в наборе — {s['set_findings']}) |")
    w(f"| **Мест проявления в документе:** | {s['occurrences']} |")
    for sev in SEVERITY:
        n_occ = sum(max(1, len(f.locs)) if f.fid.startswith("S") else 1 for f in all_f if f.severity == sev)
        w(f"| — {SEV_ICON[sev]} {SEVERITY_RU[sev]} ({SEVERITY[sev]}): | {s['by_severity'][sev]}"
          + (f" (в {n_occ} {plural(n_occ, 'месте', 'местах', 'местах')})" if n_occ != s['by_severity'][sev] else "") + " |")
    if s.get("merged"):
        w(f"| Объединено повторов одной первопричины: | {s['merged']} |")
    w(f"| Дефектов, найденных автоматическими проверками (без LLM): | {s.get('rule_findings', 0)} |")
    if res.dropped:
        w(f"| Отклонено при самопроверке (ложные срабатывания): | {len(res.dropped)} |")
    w(f"| Непокрытых пограничных сценариев (вопросы к аналитику, не дефекты): | {len(res.edge_cases)} |")
    w("")
    w("**Распределение дефектов отдельных требований по критериям:**\n")
    w("| Критерий | Дефектов |\n| :--- | :---: |")
    for c, n in sorted(s["by_criterion"].items(), key=lambda x: -x[1]):
        if n:
            w(f"| {c} — {INDIVIDUAL[c][0]} | {n} |")
    if len(s.get("by_part", {})) > 1:
        w("\n**По частям документа:** " + "; ".join(f"{p} — {n}" for p, n in s["by_part"].items()) + ".")
    tags = Counter(f.tag for f in all_f if f.tag)
    if tags:
        w("\n**Частые типы дефектов (теги):** " + ", ".join(f"`{t}` — {n}" for t, n in tags.most_common(10)) + ".")
    crit = sorted([f for f in all_f if f.severity == "critical"], key=lambda f: (f.fid[0] != "S", f.fid))
    w(f"\n### Список критических дефектов (блокеры разработки) — {len(crit)}\n")
    if not crit:
        w("Критических дефектов не выявлено.\n")
    for i, f in enumerate(crit, 1):
        places = f" — **мест: {len(f.locs)}**" if len(f.locs) > 1 else ""
        w(f"{i}. **{R.label(f) or f.fid}** {f.problem}{places} — {R.refs(f.req_ids)}; {R.loc(f)} ({f.fid}).")
    w("\n---\n")

    # ------------------------------------------------------------ 6
    w("## 6. Итоговый вердикт и первоочередные рекомендации\n")
    n_crit, n_major = s["by_severity"]["critical"], s["by_severity"]["major"]
    w("### Вердикт:\n")
    if not reqs:
        w("⛔ **РАЗРАБОТКУ НАЧИНАТЬ НЕЛЬЗЯ** — в документе не выделено ни одного требования.\n")
    elif n_crit:
        w("⛔ **РАЗРАБОТКУ НАЧИНАТЬ НЕ РЕКОМЕНДУЕТСЯ ДО УСТРАНЕНИЯ БЛОКЕРОВ.**\n")
        n_occ = sum(max(1, len(f.locs)) if f.fid.startswith("S") else 1 for f in all_f if f.severity == "critical")
        w(f"Найдено {n_crit} {plural(n_crit, 'уникальный блокер', 'уникальных блокера', 'уникальных блокеров')} (CRITICAL"
          + (f", {plural(n_crit, 'проявляется', 'проявляются', 'проявляются')} в {n_occ} "
             f"{plural(n_occ, 'месте', 'местах', 'местах')} документа" if n_occ != n_crit else "")
          + f") и {n_major} {plural(n_major, 'существенный дефект', 'существенных дефекта', 'существенных дефектов')}"
          " (MAJOR). По документу в "
          "текущем виде нельзя однозначно реализовать функциональность и составить тест-кейсы с объективным "
          "ожидаемым результатом; тестирование также начинать нельзя, пока не устранены блокеры и не заполнен "
          "раздел приёмочного тестирования.\n")
    elif n_major:
        w("⚠️ **УСЛОВНО ДОПУСКАЕТСЯ С ОГОВОРКАМИ.**\n")
        w(f"Блокеров нет, но {n_major} {plural(n_major, 'существенное замечание', 'существенных замечания', 'существенных замечаний')} (MAJOR) нужно закрыть до начала спринта по затронутым функциям.\n")
    else:
        w("✅ **ДОПУСКАЕТСЯ.** Существенных дефектов нет; замечания MINOR можно исправить в рабочем порядке.\n")

    w("### Что исправить в первую очередь (Action Items):\n")
    prio = sorted(all_f, key=lambda f: (SEVERITY_ORDER[f.severity], f.fid[0] != "S", f.fid))
    prio = [f for f in prio if f.severity in ("critical", "major")]
    groups: dict[tuple, list[Finding]] = {}
    for f in prio:
        key = (f.fid,) if f.fid.startswith("S") else ("F",) + tuple(f.req_ids)
        groups.setdefault(key, []).append(f)
    if not groups:
        w("Обязательных исправлений нет.\n")
    items = list(groups.values())
    for i, fs in enumerate(items[:15], 1):
        f = fs[0]
        more = f" Связанные дефекты: {', '.join(x.fid for x in fs[1:])}." if len(fs) > 1 else ""
        where = f"мест: {len(f.locs)}" if len(f.locs) > 1 else R.loc(f)
        w(f"{i}. `[{SEVERITY[f.severity]}]` **{R.label(f) or f.fid}** {R.refs(f.req_ids)} ({where}; {f.fid}) — "
          f"{f.recommendation or f.problem}{more}")
    if len(items) > 15:
        w(f"\nОстальные {len(items) - 15} позиций — см. разделы 3 и 4.")
    qs = [(r.id, q) for r in reqs for q in r.questions] + [("EC", e["question"]) for e in res.edge_cases if e["question"]]
    if qs:
        w("\n### Вопросы к аналитику и бизнесу (закрыть до начала разработки)\n")
        for i, (rid, q) in enumerate(qs[:25], 1):
            w(f"{i}. {'' if rid == 'EC' else f'({R.ref(rid)}) '}{q}")
    w("\n---\n")

    # ------------------------------------------------------------ приложения
    w("## Приложение А. Критерии и методика\n")
    w("**Критерии отдельного требования:**\n")
    w("| Код | Критерий | Источник | Что проверяется |\n| :--- | :--- | :--- | :--- |")
    for c, (name, src, desc) in INDIVIDUAL.items():
        w(f"| {c} | {name} | {src} | {md_cell(desc)} |")
    w("\n**Критерии набора требований:**\n")
    w("| Код | Критерий | Источник | Что проверяется |\n| :--- | :--- | :--- | :--- |")
    for c, (name, src, desc) in SET.items():
        w(f"| {c} | {name} | {src} | {md_cell(desc)} |")
    w("\n**Классификация критичности:**\n")
    for line in CRITICAL_HINT.split("\n"):
        w(f"- {line}")
    w("\n**Как работает агент:** (1) чтение документа и нумерация строк; (2) LLM выделяет элементарные требования, "
      "каждая формулировка сверяется с текстом документа; (3) LLM проверяет каждое требование по 10 критериям с "
      "контекстом всего документа, словарь размытых формулировок дополняет проверку; (4) проверка набора в два "
      "прохода (непротиворечивость/стандарты; полнота/пограничные сценарии/прослеживаемость) + детерминированные "
      "проверки документа (единицы измерения, HTTP-семантика, заглушки шаблона); (5) каждая цитата верифицируется "
      "по тексту документа, неподтверждённые замечания отбрасываются.\n")
    w("## Приложение Б. Ограничения и журнал проверки\n")
    w("- Проверка статическая: оценивается текст требований, программа не запускается.")
    w("- Все цитаты сверены с текстом документа автоматически; замечания модели без подтверждаемой цитаты "
      "или ссылки на требование в отчёт не включаются.")
    w(f"- Замечаний модели, привязанных ко всей формулировке из-за неточной цитаты: {s['unverified']}.")
    ch = res.checks or {}
    pre, post = ch.get("pre", {}), ch.get("post", {})
    if ch:
        w(f"- Самопроверка согласованности отчёта: оценок без подтверждающего дефекта оформлено как дефект — "
          f"{pre.get('created', 0)}, снято — {pre.get('cleared', 0) + post.get('cleared', 0)}; противоречий, "
          f"отражённых у второй стороны, — {pre.get('linked', 0) + post.get('linked', 0)}; исправлено комментариев — "
          f"{pre.get('comments', 0) + post.get('comments', 0)}.")
    for wn in dict.fromkeys(res.warnings):
        w(f"- {wn}")
    if res.dropped:
        w("\n### Отклонено при самопроверке (ложные срабатывания)\n")
        w("LLM-ревьюер второго уровня отклонил эти кандидаты в дефекты; для каждого приведено место документа, "
          "которое их опровергает.\n")
        w("| Где | Кандидат в дефект | Почему отклонён | Доказательство |\n| :--- | :--- | :--- | :--- |")
        for d_ in res.dropped:
            w(f"| {md_cell(d_['where'])} | {md_cell(short(d_['problem'], 200))} | {md_cell(d_['reason'] or '—')} | "
              f"{md_cell(d_['evidence'] or 'стандартное поле шаблона')} |")
    w(f"\n*Отчёт сформирован автоматически агентом req-review-agent {agent_version}.*")
    return "\n".join(out) + "\n"


def _req_block(R: _R, r: Requirement, w) -> None:
    title = f"### [{r.id}] {(r.kind[:1].upper() + r.kind[1:]) if r.kind and r.kind != '—' else 'Требование'}" \
            + (f" {r.doc_id}" if r.doc_id else "") + f": {short(r.summary, 90)}"
    w(title + "\n")
    w(f"* **Место:** {R.doc.location(r.g1, r.g2)}")
    w(f"* **Цитата из документа:** *«{r.text}»*")
    w("* **Оценка по критериям:**")
    for c in INDIVIDUAL:
        st = r.criteria.get(c, "na")
        if c not in GUIDE_CORE and st in ("ok", "na"):
            continue
        name = INDIVIDUAL[c][0]
        cm = r.comments.get(c, "")
        if st in ("warn", "fail") and not cm:
            fs = [f for f in r.findings if f.criterion == c]
            cm = fs[0].problem if fs else ("см. системный дефект " + ", ".join(m.fid for m in r.linked) + "."
                                            if r.linked else "см. замечания к набору (раздел 4).")
        text = STATUS_TEXT[st] + (f" {cm}" if cm and st != "na" else "")
        w(f"  * **{name}:** {text}")
    if r.linked:
        w("* **Системные дефекты, затрагивающие требование** (описаны один раз в разделе 4):")
        for m in r.linked:
            w(f"  * `[{SEVERITY[m.severity]}]` {SEV_ICON[m.severity]} **{m.fid}** {R.label(m)} — {short(m.problem, 160)}")
    if getattr(r, "related", None):
        w("* **Противоречия, найденные у других требований:**")
        for f, rid in r.related:
            w(f"  * `[{SEVERITY[f.severity]}]` {SEV_ICON[f.severity]} **{f.fid}** у {R.ref(rid)} — {short(f.problem, 200)}")
    if r.findings:
        w("* **Выявленные дефекты:**")
        for f in r.findings:
            _finding_block(R, f, lambda x: w("  " + x), with_reqs=False)
        recs = [f for f in r.findings if f.recommendation]
        if recs:
            w("* **Рекомендация по исправлению:**")
            for f in recs:
                w(f"  * ({f.fid}) {f.recommendation}")
    elif not r.linked and not getattr(r, "related", None):
        w("* **Выявленные дефекты:** " + ("не выявлено." if r.reviewed else "⚠️ модель не вернула результат; "
                                                                             "правила дефектов не выявили."))
    if r.test_ideas:
        w("* **Идеи тест-кейсов / граничные значения:** " + "; ".join(r.test_ideas))
    if r.questions:
        w("* **Вопросы к аналитику:** " + " ".join(r.questions))
    w("")


def to_json(res: Result, agent_version: str) -> str:
    R = _R(res)

    def fd(f: Finding):
        return {"id": f.fid, "severity": SEVERITY[f.severity], "criterion": f.criterion, "tag": f.tag,
                "type": f.type, "part": res.doc.part_of(f.g1) if f.g1 is not None else "",
                "requirements": f.req_ids, "location": R.loc(f), "quotes": f.quotes or ([f.quote] if f.quote else []),
                "quote_verified": f.verified, "problem": f.problem, "recommendation": f.recommendation,
                "source": f.source, "places": max(1, len(f.locs)), "merged_duplicates": f.merged}
    data = {
        "document": {"title": res.doc.title, "file": res.doc.path.name, "sha256": res.doc.sha256},
        "model": res.model_name, "agent_version": agent_version,
        "date": dt.datetime.fromtimestamp(res.finished).astimezone().isoformat(timespec="seconds"),
        "duration_sec": round(res.finished - res.started, 1),
        "stats": res.stats,
        "requirements": [{"id": r.id, "doc_id": r.doc_id, "part": r.part, "section": r.section,
                          "location": res.doc.location(r.g1, r.g2), "text": r.text, "summary": r.summary,
                          "kind": r.kind, "criteria": r.criteria, "criteria_comments": r.comments,
                          "findings": [fd(f) for f in r.findings], "test_ideas": r.test_ideas,
                          "questions": r.questions, "systemic_defects": [m.fid for m in r.linked]}
                         for r in res.requirements],
        "set_criteria": {c: {"status": v["status"], "comment": v["comment"]} for c, v in res.set_criteria.items()},
        "set_findings": [fd(f) for f in res.set_findings],
        "edge_cases": res.edge_cases,
        "traceability": res.traceability,
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
