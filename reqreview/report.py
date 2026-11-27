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
from .coverage import STATUS_RU as RUN_STATUS_RU
from .ingest import METHOD_RU, PAGE_RU, STATUS_RU as ING_STATUS_RU, human_size
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
            return "; ".join(self.doc.location(a, b, link=True) for a, b in f.locs)
        return self.doc.location(f.g1, f.g2, link=True)

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
            return [f"{self.doc.location(a, b, link=True)} — «{q}»" for (a, b), q in zip(f.locs, f.quotes)]
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
    if f.caveat:
        w(f"  - ⚠️ **Оговорка:** {f.caveat}")
    if f.severity == "critical":
        _why_block(f, w)


WHY_TITLES = [("trigger", "Когда проявляется"), ("development", "Влияние на разработку"),
              ("testing", "Влияние на тестирование"), ("business", "Влияние на бизнес и клиента"),
              ("exploitation", "Возможность злоупотребления (риск уровня требований)"),
              ("scenario", "Сценарий"), ("resolution", "Какое решение нужно"),
              ("closure", "Условие закрытия дефекта")]


def _why_block(f: Finding, w) -> None:
    if not f.why:
        w("  - **Почему это критично:** обоснование не сформировано (см. журнал проверки).")
        return
    w("  - **Почему это критично** (дефект → поведение → последствие → решение):")
    for k, title in WHY_TITLES:
        v = f.why.get(k)
        if k == "exploitation" and not v:
            continue
        w(f"    - *{title}:* {v if v else 'не установлено документом'}")


STATUS_ICON = {"completed": "✅", "partial": "⚠️", "failed": "⛔", "na": "ℹ️"}


def _fmt_n(n) -> str:
    return "неизвестно" if n is None else f"{n:,}".replace(",", " ")


def _integrity_rows(ing, cov: dict | None, doc=None) -> list[tuple[str, str]]:
    """Блок A панели: идентичность документа, обработка и фактический объём проверки (всё посчитано кодом)."""
    rows = [("Файл", f"`{ing.filename}` · обнаруженный формат: {ing.detected or '—'}"
             + (f" · кодировка: {ing.encoding}" if ing.encoding else "")),
            ("Размер исходного файла", f"{_fmt_n(ing.size)} байт ({human_size(ing.size)})"),
            ("SHA-256 исходного файла", f"`{ing.sha256[:16]}…` (полностью — в паспорте)" if ing.sha256 else "—"),
            ("Статус приёма документа", ING_STATUS_RU.get(ing.status, ing.status))]
    if ing.fmt == "PDF" or ing.expected_pages is not None or len(ing.pages) > 1:
        pc = ing.page_counts()
        rows.append(("Страницы исходного PDF",
                     f"всего: {_fmt_n(ing.expected_pages)}; учтено в журнале: {len(ing.pages)}; прочитано: "
                     f"{pc.get('readable', 0)}; частично: {pc.get('partial', 0)}; под сомнением: {pc.get('uncertain', 0)}; "
                     f"не прочитано: {pc.get('unreadable', 0)}; пустых (проверено): {pc.get('blank', 0)}"
                     + (f"; не обработано (лимит): {pc['skipped']}" if pc.get("skipped") else "")))
    else:
        rows.append(("Страницы", "не применимо (документ без страниц)"))
    rows.append(("Способ извлечения текста", ", ".join(ing.methods()) or "—"))
    rows.append(("OCR", ing.ocr_state + (f" — {ing.ocr_engine}" if ing.ocr_engine else "")))
    if doc is not None:
        rows.append(("Символов извлечено в итоговый текст", _fmt_n(doc.canon_len)
                     + (f" (кандидаты до сведения: текстовый слой — {_fmt_n(ing.native_chars)}, "
                        f"OCR — {_fmt_n(ing.ocr_chars)})" if ing.ocr_chars else "")))
    if cov:
        pct = f" ({cov['completed_pct']:.1f}% извлечённого текста)".replace(".", ",") \
            if cov.get("completed_pct") is not None else " (не применимо)"
        rows += [("Уникальных символов передано на проверку", _fmt_n(cov["submitted_chars"])),
                 ("Символов в завершённых проверках фрагментов", _fmt_n(cov["completed_chars"]) + pct),
                 ("Непроверенных символов извлечённого текста", _fmt_n(cov["unreviewed_chars"]))]
        done = [c for c in cov["checks"] if c["status"] == "выполнена"]
        rows.append(("Проверки документа целиком", f"выполнено {len(done)} из {len(cov['checks'])}"
                     + ("" if len(done) == len(cov["checks"]) else
                        " — не выполнены: " + "; ".join(c["title"] for c in cov["checks"] if c["status"] != "выполнена"))))
    elif doc is None:
        rows.append(("Символов извлечено в итоговый текст", "0 или неизвестно — текст не получен"))
    return rows


def _integrity_alerts(ing, w) -> None:
    if not ing.issues:
        w("Инцидентов целостности не выявлено: документ прочитан полностью, восстановление не потребовалось.\n")
        return
    w("| ID | Где | Что наблюдается | Достоверность | Что предпринято → результат | Влияние на выводы | Что сделать |")
    w("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for i in ing.issues:
        w(f"| {i.id} | {md_cell(i.location)} | {md_cell(i.observation)} | {i.certainty} | "
          f"{md_cell(i.recovery)} → {md_cell(i.result)} | {md_cell(i.effect or '—')} | {md_cell(i.next_action or '—')} |")
    w("\n*Инциденты ING-… — это проблемы чтения входного файла, а не дефекты спецификации; они не входят в счётчики "
      "дефектов и не используют шкалу CRITICAL/MAJOR/MINOR.*\n")


def _dashboard(res: Result, R: "_R", w) -> None:
    d, s, ing, cov = res.doc, res.stats, res.doc.ingest, res.coverage
    reqs = res.requirements
    all_f = [f for r in reqs for f in r.findings] + res.set_findings
    w("## Панель: дефекты, целостность и фактический объём проверки\n")
    w(f"**Статус проверки:** {STATUS_ICON[res.status]} **{RUN_STATUS_RU[res.status]}**"
      + (" — " + " ".join(res.status_reasons) if res.status_reasons else
         " — все обязательные этапы выполнены, документ учтён полностью.") + "\n")
    w("### A. Документ и обработка\n")
    w("| Показатель | Значение |\n| :--- | :--- |")
    if ing is not None:
        for k, v in _integrity_rows(ing, cov, d):
            w(f"| {k} | {v} |")
    w("")
    w("### B. Целостность и полнота исходных данных\n")
    if ing is not None:
        _integrity_alerts(ing, w)
    w("### C. Дефекты спецификации\n")
    n_q = sum(len(r.questions) for r in reqs) + len(res.edge_cases)
    incomplete = sum(1 for r in reqs if not r.reviewed)
    w("| Показатель | Значение |\n| :--- | :--- |")
    w(f"| Уникальных подтверждённых дефектов | **{s['findings']}** — 🔴 CRITICAL: {s['by_severity']['critical']}, "
      f"🟡 MAJOR: {s['by_severity']['major']}, 🟢 MINOR: {s['by_severity']['minor']} |")
    w(f"| Вопросов к аналитику (уточнения, не дефекты) | {n_q} |")
    w(f"| Требований выделено / проверено моделью / проверено не полностью | {len(reqs)} / {s.get('reviewed_reqs', 0)} / "
      f"{incomplete} |")
    w(f"| Требований, затронутых дефектами | {s['with_issues']} |")
    if res.dropped:
        w(f"| Отклонено самопроверкой как ложные срабатывания | {len(res.dropped)} |")
    w("")
    crit = sorted([f for f in all_f if f.severity == "critical"], key=lambda f: (f.fid[0] != "S", f.fid))
    top = crit or sorted([f for f in all_f if f.severity == "major"], key=lambda f: -len(f.locs))[:5]
    if top:
        w(("**Блокеры (CRITICAL):**" if crit else "**Главные существенные дефекты (MAJOR):**") + "\n")
        for f in top[:8]:
            first = R.where(f)[0].split(" — «")[0]
            rq = R.refs(f.req_ids) if f.req_ids else ""
            w(f"- **{f.fid}** {R.label(f)} {short(f.problem, 180)} — {rq + '; ' if rq else ''}{first}"
              + (f" (+ ещё {len(f.locs) - 1} мест)" if len(f.locs) > 1 else "")
              + (f". Решение: {short(f.why['resolution'], 160)}" if f.why.get("resolution") else ""))
        w("")
    w("### D. Вывод о готовности\n")
    w(_verdict_line(res) + "\n")
    w("---\n")


def _verdict_line(res: Result) -> str:
    s = res.stats
    n_crit, n_major = s["by_severity"]["critical"], s["by_severity"]["major"]
    scope = "" if res.status == "completed" else \
        " Вывод ограничен проверенной частью документа: общий вердикт «готов к разработке» не выносится."
    if res.status == "failed":
        return "⛔ **Оценка качества не выставляется** — проверка по существу не выполнена." + scope
    if res.status == "na":
        return "ℹ️ **Оценка качества не выставляется** — в документе нет оцениваемых требований."
    if n_crit:
        return (f"⛔ **Разработку начинать не рекомендуется до устранения блокеров** ({n_crit} CRITICAL, "
                f"{n_major} MAJOR)." + scope)
    if n_major:
        return f"⚠️ **Условно допускается с оговорками** — блокеров нет, {n_major} MAJOR закрыть до спринта." + scope
    if res.status != "completed":
        return ("ℹ️ **В пределах выполненных проверок подтверждённых существенных дефектов не выявлено.**" + scope)
    return "✅ **Допускается** — существенных дефектов нет; замечания MINOR исправить в рабочем порядке."


def render(res: Result, agent_version: str, input_display: str, include_source: bool = True) -> str:
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

    _dashboard(res, R, w)

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
        ("Размер файла / обнаруженный формат", (f"{d.ingest.size} байт ({human_size(d.ingest.size)}) · {d.ingest.detected}"
                                                if d.ingest else "—")),
        ("Чтение документа", (f"{d.ingest.parser}; способ: {', '.join(d.ingest.methods())}; OCR: {d.ingest.ocr_state}"
                              if d.ingest else "—")),
        ("Статус проверки", f"{RUN_STATUS_RU[res.status]}" + (f" — {' '.join(res.status_reasons)}" if res.status_reasons else "")),
        ("Структура документа", (", ".join(parts) + "; " if parts else "") + f"{len(d.nonempty())} непустых строк"),
        ("Стандарты и критерии", "ISO/IEC/IEEE 29148:2018 п. 5.2.5, 5.2.6; IEEE Std 830-1998 п. 4.3; "
                                 "руководство организаторов по критериям (см. Приложение А)"),
    ]
    for k, v in rows:
        w(f"* **{k}:** {v}")
    w("\n*Обозначения:* REQ-xx — требование в отчёте; F-xxx — дефект отдельного требования; S-xx — дефект набора; "
      "ING-xxx — инцидент чтения входного файла (не дефект); "
      + ("«строка N» — номер строки в исходном файле" if d.fmt in ("Markdown", "TXT") else
         "«L0042» — номер строки канонического транскрипта (не номер строки в исходном PDF/DOCX)"
         + (", «с. N» — страница исходного PDF" if d.fmt == "PDF" else ""))
      + "; ссылки на места ведут в приложение «Исходный текст»; "
      "`[CRITICAL]` 🔴 блокер, `[MAJOR]` 🟡 существенный, `[MINOR]` 🟢 замечание.\n")
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
        loc = d.location(r.g1, r.g2, link=True).split(", раздел")[0]
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
    if res.status != "completed":
        w(f"> {STATUS_ICON[res.status]} **Статус проверки: {RUN_STATUS_RU[res.status]}.** "
          + " ".join(res.status_reasons) + " Вывод ниже относится только к проверенной части; общий вердикт "
          "«готов к разработке» для всего документа не выносится.\n")
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
    elif res.status == "completed":
        w("✅ **ДОПУСКАЕТСЯ.** Существенных дефектов нет; замечания MINOR можно исправить в рабочем порядке.\n")
    else:
        w("ℹ️ **В пределах выполненных проверок подтверждённых существенных дефектов не выявлено.**\n")

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
    w("\n**Как работает агент:** (1) приём файла: снимок, SHA-256, размер, проверка фактического формата, "
      "читаемости, шифрования и лимитов; (2) извлечение текста с контролем качества каждой страницы; сканы и "
      "искажённые страницы восстанавливаются OCR (если он установлен), всё фиксируется в журнале страниц; "
      "(3) фиксация канонического транскрипта с номерами строк — на него ссылаются все цитаты; (4) LLM выделяет "
      "элементарные требования, каждая формулировка сверяется с текстом; (5) LLM проверяет каждое требование по "
      "10 критериям с контекстом всего документа, словарь размытых формулировок дополняет проверку; (6) проверка "
      "набора в три прохода (непротиворечивость/стандарты; полнота/пограничные сценарии/прослеживаемость; сценарный "
      "анализ противоречий) + детерминированные проверки документа; (7) объединение дефектов по первопричине; "
      "(8) самопроверка: ревьюер второго уровня, запрет выдуманных значений, согласованность отметок и дефектов, "
      "защита от критичных выводов по ненадёжному OCR; (9) обоснование «Почему это критично» для каждого блокера; "
      "(10) все счётчики, статусы и панель считаются кодом из одной модели данных.\n")
    w("**Как считаются символы:** символ — кодовая точка Unicode канонического транскрипта, включая пробелы и "
      "переводы строк; метки строк, текст инструкций модели, текст отчёта и диагностические вставки не считаются. "
      "«Передано на проверку» — объединение участков текста, отправленных модели (повторы и перекрытия считаются "
      "один раз). «В завершённых проверках» — участки, по которым модель вернула валидный результат выделения и "
      "проверки требований. 100% означает покрытие извлечённого текста, а не всего исходного файла, если в "
      "разделе B есть нечитаемые страницы или изображения.\n")
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
    _appendix_integrity(res.doc.ingest, res.coverage, w)
    if include_source:
        source_appendix_md(res.doc, w)
    w(f"\n*Отчёт сформирован автоматически агентом req-review-agent {agent_version}.*")
    return "\n".join(out) + "\n"


SOURCE_TITLE = "Приложение Г. Исходный текст (канонический транскрипт)"


def _appendix_integrity(ing, cov: dict | None, w) -> None:
    if ing is None:
        return
    w("\n## Приложение В. Журнал приёма документа и покрытия проверки\n")
    w(f"- Парсер: {ing.parser or '—'}; время чтения: {ing.seconds:.1f} сек; лимиты: размер ≤ "
      f"{human_size(ing.limits.get('max_input_bytes', 0))}, страниц ≤ {ing.limits.get('max_pages', '—')}.")
    w(f"- OCR: {ing.ocr_state}" + (f" — {ing.ocr_engine}" if ing.ocr_engine else "") + ". OCR распознаёт только "
      "то, что видно на изображении страницы: он не восстанавливает удалённые страницы, недостающие байты или "
      "зашифрованное содержимое.")
    for n in ing.notes:
        w(f"- {n}")
    if ing.fmt == "PDF" and ing.pages:
        w("\n**Журнал страниц:**\n")
        w("| Стр. | Статус | Способ | Символов (текстовый слой) | Символов (OCR) | Уверенность OCR | Примечание |")
        w("| :--- | :--- | :--- | ---: | ---: | ---: | :--- |")
        for p in ing.pages:
            w(f"| {p.page} | {PAGE_RU.get(p.status, p.status)} | {METHOD_RU.get(p.method, p.method)} | {p.native_chars} | "
              f"{p.ocr_chars or '—'} | {f'{p.ocr_conf:.0f}' if p.ocr_conf is not None else '—'} | {md_cell(p.note or '—')} |")
        w("\n*Уверенность OCR — диагностическое значение движка Tesseract (0–100), а не вероятность правильности.*")
    if cov:
        w("\n**Обязательные проверки документа целиком:**\n")
        w("| Проверка | Статус | Примечание |\n| :--- | :--- | :--- |")
        for c in cov["checks"]:
            w(f"| {c['title']} | {c['status']} | {md_cell(c['note'] or '—')} |")
        ext = [c for c in cov["manifest"] if c["kind"] == "extract"]
        rev = [c for c in cov["manifest"] if c["kind"] == "review"]
        w(f"\n**Манифест фрагментов:** выделение требований — {len(ext)} фрагм. (успешно "
          f"{sum(1 for c in ext if c['status'] == 'completed')}); проверка требований — {len(rev)} "
          f"(успешно {sum(1 for c in rev if c['status'] == 'completed')}).")
        if cov["chunks_failed"]:
            w("\n| Фрагмент | Требования | Причина |\n| :--- | :--- | :--- |")
            for c in cov["chunks_failed"]:
                w(f"| {c['id']} | {', '.join(c['req_ids']) or '—'} | {md_cell(c['note'] or '—')} |")
        w(f"\nИнвариант: проверено ({cov['completed_chars']}) ≤ передано ({cov['submitted_chars']}) ≤ извлечено "
          f"({cov['canonical_chars']}) — соблюдён.")


def source_appendix_md(doc, w) -> None:
    """Исходный текст строится программой из канонического транскрипта (не моделью)."""
    w(f"\n## {SOURCE_TITLE}\n")
    ing = doc.ingest
    partial = ing is not None and any(p.status in ("unreadable", "partial", "uncertain", "skipped") for p in ing.pages)
    w(("⚠️ **Транскрипт неполный:** часть содержимого не прочитана (см. раздел B панели). " if partial else "")
      + ("Номера L…. — строки канонического транскрипта; " if doc.fmt not in ("Markdown", "TXT") else
         "Номера — строки исходного файла (L0001 = строка 1); ")
      + "строки с пометкой «OCR» распознаны с изображения. Перенос длинных строк не меняет нумерацию.\n")
    w("```text")
    cur = None
    pages = {p.page: p for p in ing.pages} if ing else {}
    for ln in doc.lines:
        if ln.page is not None and ln.page != cur:
            cur = ln.page
            p = pages.get(cur)
            w(f"===== с. {cur}" + (f" · {METHOD_RU.get(p.method, p.method)}" if p else "") + " =====")
        w(f"L{ln.gid:04d}{' OCR' if ln.src == 'ocr' else '    '} │ {ln.text}")
    w("```")
    gaps = [p for p in (ing.pages if ing else []) if p.status in ("unreadable", "skipped", "blank")]
    if gaps:
        w("\n*Страницы без текста в транскрипте:* " + "; ".join(
            f"с. {p.page} — {PAGE_RU.get(p.status, p.status)}" for p in gaps) + ".")


def render_diagnostic(ing, status: str, reasons: list[str], agent_version: str, input_display: str,
                      model_name: str = "—", doc=None, started: float | None = None) -> str:
    """Диагностический отчёт, когда проверка по существу невозможна: без выдуманной оценки качества."""
    now = dt.datetime.now().astimezone()
    out: list[str] = []
    w = out.append
    w("# Отчёт о статической проверке требований — диагностика\n")
    w(f"**Документ:** `{Path(input_display).name}` · **Дата проверки:** {now:%d.%m.%Y %H:%M}\n")
    w("## Панель: дефекты, целостность и фактический объём проверки\n")
    w(f"**Статус проверки:** {STATUS_ICON.get(status, '⛔')} **{RUN_STATUS_RU.get(status, status)}** — "
      + " ".join(reasons) + "\n")
    w("### A. Документ и обработка\n")
    w("| Показатель | Значение |\n| :--- | :--- |")
    for k, v in _integrity_rows(ing, None, doc):
        w(f"| {k} | {v} |")
    w("\n### B. Целостность и полнота исходных данных\n")
    _integrity_alerts(ing, w)
    w("### C. Дефекты спецификации\n")
    w("Проверка требований не выполнялась: дефекты не подсчитываются, оценка качества **не выставляется** "
      "(это не означает, что документ хороший или плохой).\n")
    w("### D. Что сделать дальше\n")
    acts = list(dict.fromkeys(i.next_action for i in ing.issues if i.next_action)) or \
        ["Устраните причину, указанную выше, и запустите проверку снова."]
    for a in acts:
        w(f"- {a}")
    w("\n## 1. Паспорт прогона\n")
    for k, v in [("Путь к файлу", f"`{input_display}`"), ("Дата и время прогона", f"{now:%Y-%m-%d %H:%M:%S} (UTC{now:%z})"),
                 ("Используемая LLM модель", model_name), ("Версия агента", agent_version),
                 ("SHA-256", f"`{ing.sha256 or '—'}`"), ("Размер", f"{ing.size} байт ({human_size(ing.size)})")]:
        w(f"* **{k}:** {v}")
    _appendix_integrity(ing, None, w)
    if doc is not None:
        source_appendix_md(doc, w)
    w(f"\n*Отчёт сформирован автоматически агентом req-review-agent {agent_version}.*")
    return "\n".join(out) + "\n"


def _req_block(R: _R, r: Requirement, w) -> None:
    title = f"### [{r.id}] {(r.kind[:1].upper() + r.kind[1:]) if r.kind and r.kind != '—' else 'Требование'}" \
            + (f" {r.doc_id}" if r.doc_id else "") + f": {short(r.summary, 90)}"
    w(title + "\n")
    w(f"* **Место:** {R.doc.location(r.g1, r.g2, link=True)}")
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
                "requirements": f.req_ids, "location": plain(R.loc(f)), "quotes": f.quotes or ([f.quote] if f.quote else []),
                "quote_verified": f.verified, "problem": f.problem, "recommendation": f.recommendation,
                "source": f.source, "places": max(1, len(f.locs)), "merged_duplicates": f.merged,
                "why_critical": f.why or None, "caveat": f.caveat or None, "also": f.also,
                "criteria": f.criteria or [f.criterion], "classification": f.kind, "observations": f.obs,
                "lines": [list(x) for x in (f.locs or ([(f.g1, f.g2)] if f.g1 is not None else []))]}
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
                          "questions": r.questions, "systemic_defects": [m.fid for m in r.linked],
                          "related": [[f.fid, rid] for f, rid in (r.related or [])], "lines": [r.g1, r.g2]}
                         for r in res.requirements],
        "set_criteria": {c: {"status": v["status"], "comment": v["comment"]} for c, v in res.set_criteria.items()},
        "set_findings": [fd(f) for f in res.set_findings],
        "recommendations": res.recommendations,
        "edge_cases": res.edge_cases,
        "traceability": res.traceability,
        "warnings": list(dict.fromkeys(res.warnings)),
        "run_status": {"code": res.status, "label": RUN_STATUS_RU[res.status], "reasons": res.status_reasons},
        "dropped": res.dropped,
        "self_check": {k: v for k, v in (res.checks or {}).items()},
        "ingestion": ingestion_json(res.doc.ingest),
        "coverage": res.coverage,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


def ingestion_json(ing) -> dict | None:
    if ing is None:
        return None
    from dataclasses import asdict
    d = asdict(ing)
    d["size_human"] = human_size(ing.size)
    return d


_LINK = __import__("re").compile(r"\[([^\]]+)\]\(#L\d+\)")


def plain(t: str) -> str:
    """Убирает Markdown-ссылки на строки (для JSON)."""
    return _LINK.sub(r"\1", t or "")


def output_paths(out_path: str | Path) -> tuple[Path, Path, Path]:
    """report.md | report.pdf → (report.md, report.pdf, report.json)."""
    out = Path(out_path)
    base = out.with_suffix("") if out.suffix.lower() in (".md", ".pdf", ".json", ".markdown") else out
    return base.with_suffix(".md"), base.with_suffix(".pdf"), base.with_suffix(".json")


def _pdf(data: dict, pdf_path: Path, doc, agent_version: str) -> tuple[Path | None, str]:
    try:
        from . import present
        from .pdf import validate_pdf
        present.write_pdf(data, doc, pdf_path, footer=f"req-review-agent {agent_version}")
        v = validate_pdf(pdf_path)
        if not v["ok"]:
            return None, f"PDF создан, но не прошёл проверку: {v}"
        return pdf_path, ""
    except Exception as e:  # noqa: BLE001
        return None, f"PDF не создан: {type(e).__name__}: {e}"


def write(res: Result, out_path: str | Path, agent_version: str, input_display: str, compare_with=None) -> dict:
    data = json.loads(to_json(res, agent_version))
    if compare_with:
        from .compare import compare
        data["comparison"] = compare(compare_with, data)
    return write_data(data, res.doc, out_path, agent_version)


def write_data(data: dict, doc, out_path: str | Path, agent_version: str) -> dict:
    """Все три файла строятся из одной модели данных (data) — её же можно загрузить из report.json."""
    from . import present
    from .records import finalize
    finalize(data, doc)                 # решения, план, вопросы, тесты, статистика и проверки — из одних записей
    md_p, pdf_p, js_p = output_paths(out_path)
    md_p.parent.mkdir(parents=True, exist_ok=True)
    md_p.write_text(present.render_markdown(data, doc), encoding="utf-8")
    pdf_ok, pdf_err = _pdf(data, pdf_p, doc, agent_version)
    data["outputs"] = {"markdown": str(md_p), "pdf": str(pdf_ok) if pdf_ok else None, "pdf_error": pdf_err or None}
    js_p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"md": md_p, "pdf": pdf_ok, "json": js_p, "pdf_error": pdf_err}


def write_diagnostic(ing, status: str, reasons: list[str], out_path: str | Path, agent_version: str,
                     input_display: str, model_name: str = "—", doc=None) -> dict:
    data = {"document": {"title": (doc.title if doc is not None else Path(input_display).name),
                         "file": Path(input_display).name, "sha256": ing.sha256},
            "run_status": {"code": status, "label": RUN_STATUS_RU.get(status, status), "reasons": reasons},
            "agent_version": agent_version, "model": model_name,
            "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "ingestion": ingestion_json(ing), "findings": None, "requirements": []}
    return write_data(data, doc, out_path, agent_version)
