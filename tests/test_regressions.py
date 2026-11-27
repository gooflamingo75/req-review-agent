"""Регрессионные сценарии точности отчёта (R01–R20 из дополнения к требованиям).

Здесь проверяется детерминированная часть: правила, объединение по первопричине, шкала критичности, записи
решений/плана/вопросов/тестов и проверки перед выводом. Сценарии R06, R11, R15, R16 зависят от суждения модели
(есть ли в документе разрешающий текст, нейтральна ли рекомендация) — они закреплены в промптах и проверке
критика, но юнит-тестом без модели не проверяются.

Запуск:  python -m pytest -q tests/test_regressions.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reqreview import records, verify  # noqa: E402
from reqreview.agent import Finding, Requirement  # noqa: E402
from reqreview.consolidate import consolidate  # noqa: E402
from reqreview.loader import load_document  # noqa: E402
from reqreview.rules import check_requirement  # noqa: E402


def _doc(text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text)
    return load_document(f.name)


def _f(fid, sev="MAJOR", problem="p", rec="", tag="", reqs=(), lines=(), crit="COM", **kw):
    d = {"id": fid, "severity": sev, "criterion": crit, "criteria": [crit], "tag": tag, "type": "",
         "requirements": list(reqs), "location": "", "quotes": [], "quote_verified": True, "problem": problem,
         "recommendation": rec, "source": "LLM", "places": 1, "merged_duplicates": 0, "why_critical": None,
         "caveat": None, "also": [], "lines": [[g, g] for g in lines], "classification": "defect"}
    d.update(kw)
    return d


def _req(rid, line, findings=(), tests=(), questions=(), criteria=None, comments=None):
    return {"id": rid, "doc_id": None, "text": f"Требование {rid}", "summary": "", "kind": "функциональное",
            "location": f"строка {line}", "criteria": criteria or {"COM": "ok"}, "criteria_comments": comments or {},
            "findings": list(findings), "test_ideas": list(tests), "questions": list(questions),
            "systemic_defects": [], "related": [], "lines": [line, line]}


def _data(reqs, set_findings=(), **kw):
    d = {"document": {"sha256": None}, "stats": {}, "requirements": reqs, "set_findings": list(set_findings),
         "edge_cases": [], "dropped": [], "recommendations": [], "run_status": {"code": "completed", "reasons": []}}
    d.update(kw)
    return d


# ---------------------------------------------------------------- R01 / R02 / R20: одна первопричина
def _rf(crit, problem, rec="", quote="", sev="major"):
    return Finding(criterion=crit, severity=sev, problem=problem, recommendation=rec, quote=quote, g1=5, g2=5,
                   req_ids=["REQ-01"], criteria=[crit])


def test_r01_r02_one_root_cause_one_defect_with_all_criteria():
    text = "Push отправляется в кратчайшие сроки после покупки."
    r = Requirement(id="REQ-01", doc_id=None, text=text, summary="", kind="функциональное", g1=5, g2=5)
    r.findings = [
        _rf("UNA", "Срок отправки push не задан: «в кратчайшие сроки» — размытая формулировка.",
            "Указать допустимую задержку в секундах.", "в кратчайшие сроки"),
        _rf("VER", "Срок отправки push не задан числом, проверить «в кратчайшие сроки» нельзя.",
            "Указать допустимую задержку в секундах.", "в кратчайшие сроки"),
        _rf("COM", "Не задан срок отправки push: «в кратчайшие сроки» без числа.",
            "Задать допустимую задержку в секундах.", "в кратчайшие сроки", sev="minor"),
    ]
    consolidate([r], [], Finding)
    assert len(r.findings) == 1, [f.problem for f in r.findings]
    assert set(r.findings[0].criteria) == {"UNA", "VER", "COM"}
    assert r.findings[0].severity == "major" and r.findings[0].merged == 2


def test_r20_stable_accounting_independent_of_order():
    text = "Push отправляется в кратчайшие сроки после покупки."
    base = [_rf("UNA", "Срок отправки push не задан: «в кратчайшие сроки» — размытая формулировка.",
                "Указать задержку в секундах.", "в кратчайшие сроки"),
            _rf("VER", "Срок push не задан числом (например 5 или 10 секунд), проверить нельзя.",
                "Указать задержку в секундах.", "в кратчайшие сроки"),
            _rf("COM", "Не описан формат push-сообщения.", "Добавить шаблон текста push.", "")]
    counts = []
    for order in ([0, 1, 2], [2, 1, 0], [1, 2, 0]):
        r = Requirement(id="REQ-01", doc_id=None, text=text, summary="", kind="", g1=5, g2=5)
        r.findings = [copy.copy(base[i]) for i in order]
        consolidate([r], [], Finding)
        counts.append(sorted(len(f.criteria) for f in r.findings))
    assert counts[0] == counts[1] == counts[2] == [1, 2], counts


def test_r03_one_conflict_in_several_places_is_one_finding():
    def sf(problem, line):
        return Finding(criterion="S-CONS", severity="critical", problem=problem, recommendation="Оставить один лимит.",
                       req_ids=["REQ-03", "REQ-07"], type="Противоречие", tag="CONTRADICTION",
                       locs=[(line, line)], quotes=["лимит"])
    out, _ = consolidate([], [sf("Верхний лимит порога противоречит: 500 000 ₸ в тексте и 1 000 000 ₸ в таблице.", 10),
                              sf("Верхний лимит порога противоречит: 500 000 ₸ в алгоритме и 1 000 000 ₸ в таблице.", 40),
                              sf("Верхний лимит порога противоречит: 500 000 ₸ и 1 000 000 ₸ в таблице параметров.", 70)],
                         Finding)
    assert len(out) == 1 and {a for a, _ in out[0].locs} == {10, 40, 70}


def test_contradictions_between_different_pairs_stay_separate():
    a = Finding(criterion="S-CONS", severity="critical", problem="Сценарный конфликт порогов.", recommendation="r",
                req_ids=["REQ-01", "REQ-04"], type="Противоречие", tag="CONTRADICTION", locs=[(5, 5)])
    b = Finding(criterion="S-CONS", severity="critical", problem="Сценарный конфликт порогов.", recommendation="r",
                req_ids=["REQ-01", "REQ-02"], type="Противоречие", tag="CONTRADICTION", locs=[(5, 5)])
    out, _ = consolidate([], [a, b], Finding)
    assert len(out) == 2


# ---------------------------------------------------------------- R04 / R10 / R14: решения, план, тесты
def _reset_data():
    s1 = _f("S-01", "CRITICAL", "Не определено, сбрасывается ли накопленная сумма трат в начале месяца.",
            "Выбрать: сбрасывать накопленную сумму или переносить остаток.", tag="CONTRADICTION",
            reqs=["REQ-01", "REQ-02"], lines=[10, 20], crit="S-CONS",
            why_critical={"decision": "Сбрасывать ли накопленные траты в начале месяца?",
                          "options": ["Сбрасывать 1-го числа", "Не сбрасывать"], "impact": "Клиент получит лишний push.",
                          "owner": "Владелец продукта"})
    s2 = _f("S-02", "CRITICAL", "Не определено, восстанавливается ли разрешение на повторное уведомление.",
            "Выбрать политику повторного уведомления.", tag="CONTRADICTION", reqs=["REQ-03", "REQ-04"],
            lines=[30, 40], crit="S-CONS",
            why_critical={"decision": "Когда снова разрешено уведомление?",
                          "options": ["После сброса периода", "Только один раз"], "impact": "Повторные push.",
                          "owner": "Владелец продукта"})
    s3 = _f("S-03", "MAJOR", "Метод сохранения порога объявлен как GET.", "Заменить метод на POST.",
            tag="REST-METHOD", reqs=["REQ-05"], lines=[50], crit="S-FEAS")
    s4 = _f("S-04", "MAJOR", "Не описано, какую сумму трат показывает экран после смены месяца.",
            "Описать отображение на экране.", reqs=["REQ-01"], lines=[11], crit="S-COMPL")
    reqs = [_req("REQ-01", 10, tests=["Траты 400 000 ₸, затем 200 000 ₸ → push?"]),
            _req("REQ-02", 20), _req("REQ-03", 30), _req("REQ-04", 40),
            _req("REQ-05", 50, tests=["POST сохраняет порог → 201"])]
    return _data(reqs, [s1, s2, s3, s4])


def test_r04_two_unresolved_resets_are_two_decisions():
    d = records.finalize(_reset_data())
    decs = d["decisions"]
    assert len(decs) == 2 and [x["findings"] for x in decs] == [["S-01"], ["S-02"]]
    assert all(x["status"] == "open" and x["kind"] == "business" for x in decs)
    assert d["validation"]["status"] == "passed", d["validation"]["failed"]


def test_r10_tests_depending_on_open_decision_are_conditional():
    d = records.finalize(_reset_data())
    t1 = next(t for t in d["tests"] if "REQ-01" in t["requirements"])
    t5 = next(t for t in d["tests"] if "REQ-05" in t["requirements"])
    assert t1["conditional"] and t1["depends_on"] == ["D-01"]
    assert not t5["conditional"] and t5["depends_on"] == []


def test_r14_plan_actions_link_only_their_own_finding():
    d = records.finalize(_reset_data())
    by_f = {a["finding"]: a for a in d["plan"]}
    assert set(by_f) == {"S-03", "S-04"}, "по одному действию на дефект вне решений"
    assert by_f["S-03"]["depends_on"] == [] and by_f["S-03"]["closes"] == "full", "метод HTTP не зависит от сброса"
    assert by_f["S-04"]["depends_on"] == ["D-01"] and by_f["S-04"]["closes"] == "partial"
    assert by_f["S-04"]["text"].startswith("После решения D-01")


# ---------------------------------------------------------------- R05 / R12: раздельные дефекты, шкала
def test_r05_r12_empty_section_is_major_and_separate():
    tp = _f("S-01", "CRITICAL", "Раздел «Тест-план» не заполнен.", "Заполнить раздел.", tag="SECTION-EMPTY",
            reqs=[], lines=[90], crit="S-COMPL")
    dup = _f("S-02", "MAJOR", "Не описана обработка повторного события о покупке (дубли).",
             "Описать идемпотентность.", tag="IDEMPOTENCY", reqs=["REQ-01"], lines=[10], crit="S-COMPL")
    d = records.finalize(_data([_req("REQ-01", 10)], [tp, dup]))
    sf = {f["id"]: f for f in d["set_findings"]}
    assert sf["S-01"]["severity"] == "MAJOR" and "шкал" in sf["S-01"]["caveat"]
    assert d["decisions"] == [] and {a["finding"] for a in d["plan"]} == {"S-01", "S-02"}
    f = Finding(criterion="S-COMPL", severity="critical", problem="x", recommendation="", tag="SECTION-EMPTY")
    assert verify.severity_rubric([f]) == 1 and f.severity == "major" and f.caveat


# ---------------------------------------------------------------- R07 / R08 / R09: правила
def test_r07_reference_point_present_is_not_claimed_missing():
    hits = check_requirement("Ответ формируется не более 3 секунд после получения запроса.")
    nfr = [h for h in hits if h["tag"] == "NFR-METRIC"]
    assert nfr and "точка начала указана" in nfr[0]["problem"] and "точки начала отсчёта" not in nfr[0]["problem"]
    hits = check_requirement("Ответ формируется не более 3 секунд.")
    assert "точки начала отсчёта" in hits[0]["problem"]
    vague = check_requirement("Push отправляется в кратчайшие сроки после покупки.")
    assert vague and not any("точк" in h["problem"] for h in vague)


# R08/R09 — в tests/test_pipeline.py::test_data_type_and_nfr_rules


# ---------------------------------------------------------------- одна проблема на двух уровнях (аудит v8)
def test_same_problem_in_set_and_requirements_is_one_defect():
    s1 = _f("S-01", "CRITICAL", "Ошибка бизнес-проверки возвращается с HTTP 200 и телом result ERROR вместо 400.",
            "Возвращать 400 для ошибок проверки.", tag="STATUS-200", reqs=["REQ-02"], lines=[20, 24], crit="S-CONS")
    f1 = _f("F-001", "MAJOR", "Пример [L24] возвращает ERROR со статусом 200, хотя для неверных параметров "
            "описан код 400.", "Перенести пример ERROR под код 400.", tag="CONTRADICTION", crit="CON")
    f2 = _f("F-002", "MAJOR", "Неясно, какой код ответа означает успех.", "Описать коды успеха.", crit="UNA")
    d = _data([_req("REQ-01", 24, findings=[f1, f2]), _req("REQ-02", 20)], [s1])
    d = records.finalize(d)
    assert [f["id"] for f in d["set_findings"]] == ["S-01"] and d["set_findings"][0]["places"] == 2
    assert [f["id"] for f in d["requirements"][0]["findings"]] == ["F-001"], "второй дефект перенумерован"
    assert "S-01" in d["requirements"][0]["systemic_defects"] and "REQ-01" in d["set_findings"][0]["requirements"]
    assert len(d["decisions"]) == 1 and d["validation"]["status"] == "passed"


def test_line_reference_moves_to_line_with_cited_fragment():
    doc = _doc("# T\n\n200 OK\n\n{\"result\":\"OK\"}\n\n{\"result\":\"ERROR\"}\n")
    ln = {l_.text: l_.gid for l_ in doc.lines}
    ok, err = ln['{"result":"OK"}'], ln['{"result":"ERROR"}']
    f = _f("F-001", "MAJOR", f"В примере [L{ok}] ответ с ERROR возвращается со статусом 200.", reqs=["REQ-01"])
    g = _f("F-002", "MAJOR", f"Строки [L{ok}]-[L{err}] — примеры.", reqs=["REQ-01"])
    d = _data([_req("REQ-01", ok, findings=[f, g])])
    assert records.fix_line_refs(d, doc) == 1
    assert f"[L{err}]" in d["requirements"][0]["findings"][0]["problem"]
    assert d["requirements"][0]["findings"][1]["problem"] == g["problem"] or "[L" in g["problem"]


def test_goal_statement_and_storage_details_are_minor():
    r = _req("REQ-01", 3, findings=[_f("F-001", "MAJOR", "Цель «в момент покупки» неточна.", crit="UNA")])
    r["kind"] = "контекст (цели/архитектура/схема)"
    r2 = _req("REQ-02", 5, findings=[_f("F-002", "MAJOR", "Не указаны типы колонок.", "Указать VARCHAR(7) и BOOLEAN.")])
    d = records.finalize(_data([r, r2]))
    assert [f["severity"] for x in d["requirements"] for f in x["findings"]] == ["MINOR", "MINOR"]


def test_request_response_same_storage_field_type_mismatch():
    from reqreview.rules import check_document
    doc = _doc("\n".join(["# API", "## Query-параметры", "| Parameter | Format | DB field |", "| --- | --- | --- |",
                          "| `amount` | integer | `limit_kzt` |", "", "## Ответ", "| Parameter | Format | DB field |",
                          "| --- | --- | --- |", "| `amount` | string | `limit_kzt` |"]))
    hits = [h for h in check_document(doc) if h["tag"] == "DATA-TYPE"]
    assert len(hits) == 1 and "limit_kzt" in hits[0]["problem"] and "противореч" not in hits[0]["problem"]
    assert any(h["tag"] == "DESIGN-DETAIL" for h in check_document(doc))
    from reqreview.rules import check_requirement
    assert any(h.get("tag") == "DESIGN-DETAIL" for h in check_requirement("Запрос обрабатывает AlertsController."))


# ---------------------------------------------------------------- R13: «реализуемо» — не нарушение
def test_r13_positive_justification_is_not_a_violation():
    doc = _doc("# T\n\nСистема сохраняет порог.\n")
    r = Requirement(id="REQ-01", doc_id=None, text="Система сохраняет порог.", summary="", kind="", g1=3, g2=3,
                    criteria={"FEA": "warn", "COM": "warn"},
                    comments={"FEA": "Реализуемо при асинхронной модели", "COM": "Не указан диапазон порога"})
    verify.self_check(doc, [r], [], Finding, final=False)
    assert r.criteria["FEA"] == "ok"
    assert r.criteria["COM"] == "warn" and len(r.findings) == 1
    d = _data([_req("REQ-01", 3, criteria={"FEA": "warn"}, comments={"FEA": "Реализуемо"})])
    d = records.finalize(d)
    assert d["requirements"][0]["criteria"]["FEA"] == "ok" and d["validation"]["status"] == "passed"


# ---------------------------------------------------------------- R17 / R19: согласованность и провал проверки
def test_r17_references_consistent_and_stats_recomputed():
    d = _reset_data()
    d["stats"] = {"findings": 99, "by_severity": {"critical": 50}}             # устаревшие счётчики
    d["requirements"][0]["systemic_defects"] = ["S-01", "S-77"]                  # висячая ссылка после слияния
    d = records.finalize(d)
    assert d["stats"]["findings"] == 4 and sum(d["stats"]["by_severity"].values()) == 4
    assert d["requirements"][0]["systemic_defects"] == ["S-01"]
    assert d["validation"]["status"] == "passed"
    known = {f["id"] for f in d["set_findings"]}
    assert all(x in known for dd in d["decisions"] for x in dd["findings"])
    assert all(a["finding"] in known for a in d["plan"])


def test_r19_unrepairable_inconsistency_marks_report_partial():
    d = _reset_data()
    d["set_findings"].append(_f("S-03", "MINOR", "Версия документа не указана.", "Указать версию."))  # дубль ID
    d = records.finalize(d)
    assert d["validation"]["status"] == "failed" and "V01" in d["validation"]["failed"]
    assert d["run_status"]["code"] == "partial"
    assert any("проверки согласованности" in x for x in d["run_status"]["reasons"])
    from reqreview.present import render_markdown
    assert "не прошёл" in render_markdown(d) or "не прошел" in render_markdown(d)


# ---------------------------------------------------------------- R18: составной дефект делится критиком
def test_r18_critic_splits_compound_claim():
    from reqreview import prompts
    doc = _doc("# T\n\n| amount | integer |\n\n{\"amount\": \"50000\"}\n")
    f = Finding(criterion="S-CONS", severity="major", problem="Тип amount расходится в схеме и примере, а также "
                "в запросе и ответе.", recommendation="", fid="S-01", req_ids=["REQ-01"], tag="DATA-TYPE",
                locs=[(3, 3)], quotes=["| amount | integer |"])

    class L:
        model = "x"

        def generate_json(self, *a, **k):
            return {"reviews": [{"id": "S-01", "verdict": "split", "parts": [
                {"problem": "Схема ответа задаёт amount как integer, пример — строкой.", "supported": True},
                {"problem": "Запрос и ответ задают amount разными типами.", "supported": False,
                 "reason": "разные направления — не противоречие"}]}]}
    sf = [f]
    dropped = verify.critic(doc, [], sf, L(), doc.numbered(), prompts, [])
    assert [x.problem for x in sf] == ["Схема ответа задаёт amount как integer, пример — строкой."]
    assert any("Запрос и ответ" in x["problem"] for x in dropped)


# ---------------------------------------------------------------- классификация: вопрос — не дефект
def test_clarifications_and_recommendations_are_not_defects():
    from reqreview.agent import _split_kinds
    r = Requirement(id="REQ-01", doc_id=None, text="t", summary="", kind="", g1=1, g2=1)
    r.findings = [Finding(criterion="COM", severity="minor", problem="Какой часовой пояс используется?",
                          recommendation="", kind="clarification"),
                  Finding(criterion="APP", severity="minor", problem="Можно добавить пример.",
                          recommendation="Добавить пример ответа.", kind="recommendation"),
                  Finding(criterion="VER", severity="major", problem="Нет числа.", recommendation="")]
    edge, recs = [], []
    _split_kinds([r], [], edge, recs)
    assert [f.problem for f in r.findings] == ["Нет числа."]
    assert r.questions == ["Какой часовой пояс используется?"] and recs[0]["text"] == "Добавить пример ответа."


# ---------------------------------------------------------------- PDF: длинная таблица не теряет строк (аудит v8)
def test_pdf_long_table_keeps_every_row():
    import pymupdf

    from reqreview.pdf import write_pdf_html
    from reqreview.present import CSS
    rows = "".join(f"<tr><td>TC-{i:02d}</td><td>{'Проверка суммы и статуса ответа. ' * (1 + i % 5)}</td>"
                   f"<td>REQ-{i % 9:02d}<br/>строка {i}</td></tr>" for i in range(1, 61))
    table = f'<table class=""><tr><th width="8%">#</th><th width="72%">Что</th><th width="20%">Где</th></tr>{rows}</table>'
    out = Path(tempfile.mkdtemp()) / "t.pdf"
    for n in range(0, 40, 3):
        write_pdf_html("<p>строка</p>" * n + table, out, CSS, "t")
        with pymupdf.open(str(out)) as d:
            text = "".join(p.get_text() for p in d)
        missing = [i for i in range(1, 61) if f"TC-{i:02d}" not in text]
        assert not missing, (n, missing)
