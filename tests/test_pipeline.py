"""Тесты без ключа: имитация ответов модели (ScriptedLLM) и демо-режим.

Запуск:  python -m pytest -q   (или  python tests/test_pipeline.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reqreview.agent import run  # noqa: E402
from reqreview.llm import MockLLM, parse_json  # noqa: E402
from reqreview.loader import load_document  # noqa: E402
from reqreview.report import render  # noqa: E402

SAMPLE = ROOT / "examples" / "sample_requirements.md"


class ScriptedLLM:
    """Имитирует ответы Gemini: в т. ч. неточную цитату и выдуманную цитату."""
    name = "scripted"
    model = "scripted"

    def __init__(self):
        self.calls = 0

    def generate_json(self, task, system, prompt, payload=None):
        self.calls += 1
        if task == "extract":
            # нормальные строки + одна неточная цитата (ё/кавычки/регистр) + одна выдуманная
            text = "\n".join(t for _, t in payload["lines"])
            out = []
            for gid, t in payload["lines"]:
                if "**КБ-" in t or "**НФ-" in t:
                    doc_id = t.split("**")[1].rstrip(".")
                    body = t.split("**", 2)[2].strip()
                    out.append({"doc_id": doc_id, "text": body, "line_start": gid, "line_end": gid,
                                "summary": body[:50], "kind": "функциональное"})
                elif t.startswith("| Карта активна"):
                    out.append({"doc_id": None, "text": t, "line_start": gid, "line_end": gid,
                                "summary": "Приёмка: 2% от 10 000", "kind": "прочее"})
            out[0]["text"] = out[0]["text"].upper()  # регистр не важен
            out.append({"doc_id": "X-99", "text": "Система должна летать в космос", "line_start": 9999,
                        "line_end": 9999, "summary": "выдумка", "kind": "прочее"})
            assert "КБ-1" in text
            return json.dumps({"requirements": out}) and {"requirements": out}
        if task == "review":
            res = []
            for r in payload["requirements"]:
                f = []
                if r["doc_id"] == "КБ-2":
                    f.append({"criterion": "VER", "severity": "critical", "quote": "начисляется  БЫСТРО",
                              "problem": "Нет срока начисления.", "recommendation": "Указать срок, напр. 1 час."})
                if r["doc_id"] == "КБ-7":
                    f.append({"criterion": "UNA", "severity": "major", "quote": "цитата, которой нет",
                              "problem": "Субъективные оценки.", "recommendation": "Задать критерии."})
                res.append({"id": r["id"], "criteria": {"NEC": {"status": "ok", "comment": ""},
                                                        "VER": "fail" if f else "ok"}, "findings": f,
                            "test_ideas": ["10 000 → 200"], "questions": []})
            return {"results": res}
        if task == "set_completeness":
            ids = {r["doc_id"]: r["id"] for r in payload["requirements"]}
            return {"findings": [], "edge_cases": [{"scenario": "Несколько карт", "requirement_ids": [ids["КБ-1"]],
                                                    "risk": "r", "question": "q?"}],
                    "traceability": [{"requirement": ids["КБ-1"], "goal": [], "ui": [], "api_data": [],
                                      "acceptance": [], "gap": "нет критерия приёмки"}]}
        if task == "critic":
            out = []
            for it in payload["items"]:
                if it["problem"] == "Субъективные оценки.":
                    out.append({"id": it["id"], "verdict": "severity", "severity": "minor", "reason": "стиль"})
                if "КБ-5" in it["problem"]:
                    out.append({"id": it["id"], "verdict": "drop", "reason": "тест",
                                "evidence_quote": "Кэшбэк начисляется быстро"})
                if it["problem"] == "Сценарный конфликт.":
                    out.append({"id": it["id"], "verdict": "drop", "reason": "выдумка",
                                "evidence_quote": "этого текста нет в документе"})
                if it["problem"] == "Нет срока начисления.":
                    out.append({"id": it["id"], "verdict": "rewrite", "problem": "Нет срока начисления.",
                                "recommendation": "Перейти на https://api.example.com/v2/cashback и /api/v1/cb/limits"})
            return {"reviews": out}
        if task == "set_conflicts":
            ids = {r["doc_id"]: r["id"] for r in payload["requirements"]}
            return {"conflicts": [{"requirement_ids": [ids["КБ-1"], ids["КБ-4"]],
                                   "quotes": ["получает кэшбэк 2%", "отображается в истории операций"],
                                   "scenario": "Оплата 10 000 ₸", "outcomes": "по A: 200; по B: иначе",
                                   "severity": "major", "problem": "Сценарный конфликт.", "recommendation": "r"}],
                    "pairs": [{"id": p["id"], "verdict": "conflict" if n == 0 else "ok",
                               "problem": "Матричный конфликт.", "scenario": "s", "outcomes": "o"}
                              for n, p in enumerate(payload["pairs"])
                              if set(p["req_ids"]) != {ids["КБ-1"], ids["КБ-4"]}],
                    "bad_test_ideas": [{"requirement_id": ids["КБ-1"], "test_idea": "10 000 → 200",
                                        "violates": [ids["КБ-3"]], "why": "лимит"}]}
        if task == "set_consistency":
            ids = {r["doc_id"]: r["id"] for r in payload["requirements"]}
            return {"findings": [
                {"type": "Противоречие", "criterion": "S-CONS", "severity": "critical",
                 "requirement_ids": [ids["КБ-3"], ids["КБ-6"]],
                 "quotes": ["Максимальная сумма кэшбэка в месяц — 5 000 ₸", "составляет 3 000 ₸"],
                 "problem": "Разные лимиты.", "recommendation": "Оставить один."},
                {"type": "Пропуск", "criterion": "S-COMPL", "severity": "major", "requirement_ids": [],
                 "quotes": ["этого текста нет в документе"], "problem": "Выдумка", "recommendation": "-"},
            ], "set_criteria": {"S-CONS": {"status": "ok", "comment": ""}}}
        if task == "why_critical":
            return {"items": [{"id": it["id"], "trigger": "Оплата 10 000 ₸ в конце месяца",
                               "development": "Нельзя выбрать лимит", "testing": "Два ожидаемых результата",
                               "business": "", "exploitation": "", "scenario": "A: 5 000; B: 3 000",
                               "resolution": "Выбрать один лимит", "closure": "В документе один лимит",
                               "severity_ok": "yes"} for it in payload["items"]]}
        raise AssertionError(task)


def test_loader_and_find():
    doc = load_document(SAMPLE)
    assert doc.title.startswith("Кэшбэк")
    hit = doc.find("кэшбэк начисляется   «быстро»".replace("«", "").replace("»", ""))
    assert hit and "быстро" in hit[0]
    assert doc.find("совсем другой текст") is None


def test_scripted_pipeline():
    doc = load_document(SAMPLE)
    res = run(doc, ScriptedLLM(), batch_size=4)
    ids = [r.doc_id for r in res.requirements]
    assert "X-99" not in ids, "выдуманное требование должно быть отброшено"
    assert "КБ-1" in ids and "НФ-2" in ids and None in ids
    r1 = next(r for r in res.requirements if r.doc_id == "КБ-1")
    assert r1.text.startswith("Клиент с активной"), "цитата восстановлена дословно из документа"
    r2 = next(r for r in res.requirements if r.doc_id == "КБ-2")
    f = next(f for f in r2.findings if f.criterion == "VER")
    assert f.verified and f.quote == "начисляется быстро"
    r7 = next(r for r in res.requirements if r.doc_id == "КБ-7")
    bad = next(f for f in r7.findings if f.source == "LLM")
    assert not bad.verified and bad.quote == r7.text
    assert any(f.source == "правило" for f in r7.findings), "правила дополняют модель"
    types = [f.type for f in res.set_findings]
    assert "Противоречие" in types
    assert not any(f.problem == "Выдумка" for f in res.set_findings), "непроверяемое замечание отброшено"
    assert res.set_criteria["S-CONS"]["status"] == "fail"
    assert any("Сценарий: Оплата" in f.problem for f in res.set_findings), "сценарный конфликт"
    assert any("Матричный конфликт." in f.problem for f in res.set_findings), "вердикт матрицы пар правил"
    assert any("Матрица сценариев" in w for w in res.warnings)
    assert res.stats["model_usage"] == {} and "primary_share" in res.stats
    assert res.dropped and "Кэшбэк начисляется быстро" in res.dropped[0]["evidence"], "критик отклоняет с доказательством"
    assert all(d["id"].startswith("D-") for d in res.dropped) and "_index" not in res.dropped[0]
    assert not any("КБ-5" in f.problem for f in res.set_findings)
    r2c = next(r for r in res.requirements if r.doc_id == "КБ-2")
    rec = next(f for f in r2c.findings if f.problem == "Нет срока начисления.").recommendation
    assert "api.example.com" not in rec and "/api/v1" not in rec and "<указать" in rec, rec
    for r in res.requirements:     # самопроверка: у каждой отметки ❌/⚠️ есть дефект
        for c, st in r.criteria.items():
            if st in ("warn", "fail"):
                assert any(f.criterion == c for f in r.findings) or r.linked or r.related, (r.id, c)
    r1b = next(r for r in res.requirements if r.doc_id == "КБ-1")
    assert r1b.criteria["CON"] == "fail" and any("противоречит" in t for t in r1b.test_ideas)
    md = render(res, "test", "examples/sample_requirements.md")
    for section in ("Паспорт прогона", "Нарезка и инвентаризация требований", "Детальная проверка каждого требования",
                    "Проверка набора требований в целом", "Сводная статистика", "Итоговый вердикт",
                    "Матрица прослеживаемости", "EC-01", "Идеи тест-кейсов", "`[CRITICAL]`"):
        assert section in md
    assert "РАЗРАБОТКУ НАЧИНАТЬ НЕ РЕКОМЕНДУЕТСЯ" in md
    assert res.requirements[0].id == "REQ-01"
    assert "строка" in md


def test_mock_mode():
    doc = load_document(SAMPLE)
    res = run(doc, MockLLM())
    assert len(res.requirements) >= 10
    assert res.stats["by_severity"]["critical"] >= 1  # TBD в КБ-9


def test_parse_json():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Вот ответ: {"a": [1]} спасибо') == {"a": [1]}


def test_document_rules_generic(tmp_path=None):
    """Детерминированные проверки универсальны: другой домен, другие валюты и URL."""
    import tempfile
    from reqreview.rules import check_document
    text = "\n".join([
        "# Перевод между счетами", "## Данные", "Сумма хранится в копейках.",
        "## API", "| Method | GET |", "| URL | /accounts/updateLimit |", "Лимит задаётся в рублях.",
        "#### 200. Успешно", "```json", '{"result":"ERROR"}', "```",
        "## Тесты", "Сценарии будут определены позже.",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text)
    tags = {h["tag"] for h in check_document(load_document(f.name))}
    assert {"UNITS", "REST-METHOD", "STATUS-200", "SECTION-EMPTY"} <= tags, tags


def _tmpdoc(text):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write(text)
    return load_document(f.name)


def test_data_type_and_nfr_rules():
    """R08: разные типы в запросе и ответе — не дефект; R09: схема ответа и пример того же ответа расходятся;
    R07: точка отсчёта указана — не утверждаем, что её нет; срок без перцентиля — универсальные правила."""
    from reqreview.rules import check_document, check_requirement
    doc = _tmpdoc("\n".join([
        "# Заказ", "## Запрос", "| Поле | Тип | Описание |", "| --- | --- | --- |",
        "| `qty` | integer | Количество |", "| `note` | string | Комментарий |", "",
        "## Ответ", "| Name | Format |", "| :--- | :--- |", "| qty | string |", "| note | string |"]))
    assert [h for h in check_document(doc) if h["tag"] == "DATA-TYPE"] == [], "R08: запрос ≠ ответ — не противоречие"
    doc = _tmpdoc("\n".join([
        "# Заказ", "## Ответ", "| Поле | Тип | Описание |", "| --- | --- | --- |",
        "| `amount` | integer | Сумма |", "| `note` | string | Комментарий |", "",
        "### 200 OK", "```json", '{"amount": "50000", "note": "ok"}', "```"]))
    hits = [h for h in check_document(doc) if h["tag"] == "DATA-TYPE"]
    assert len(hits) == 1 and "amount" in hits[0]["problem"] and "note" not in hits[0]["problem"], hits
    assert [h["tag"] for h in check_requirement("Отчёт формируется не более 3 секунд.")] == ["NFR-METRIC"]
    assert check_requirement("Отчёт формируется не более 3 секунд для 95% запросов.") == []


def test_critic_needs_other_line_evidence_and_answers_edge_cases():
    from reqreview import prompts, verify
    from reqreview.agent import Finding, Requirement
    doc = _tmpdoc("# T\n\nСистема отправляет письмо.\n\nПисьмо отправляется на адрес из профиля клиента.\n")
    r = Requirement(id="REQ-01", doc_id=None, text="Система отправляет письмо.", summary="", kind="функциональное",
                    g1=3, g2=3, questions=["На какой адрес отправляется письмо?"])
    f1 = Finding(criterion="COM", severity="major", problem="Не указан адрес.", recommendation="",
                 quote="Система отправляет письмо.", g1=3, g2=3, req_ids=["REQ-01"], fid="F-001")
    f2 = Finding(criterion="UNA", severity="major", problem="Неясно что за письмо.", recommendation="",
                 quote="Система отправляет письмо.", g1=3, g2=3, req_ids=["REQ-01"], fid="F-002")
    r.findings = [f1, f2]
    ec = [{"scenario": "Адрес не указан", "req_ids": ["REQ-01"], "risk": "", "question": ""}]

    class L:
        model = "x"

        def generate_json(self, *a, **k):
            return {"reviews": [
                {"id": "F-001", "verdict": "drop", "reason": "см. F-002",
                 "evidence_quote": "Письмо отправляется на адрес из профиля клиента"},
                {"id": "F-002", "verdict": "drop", "reason": "своя строка",
                 "evidence_quote": "Система отправляет письмо"},
                {"id": "Q-01", "verdict": "drop", "evidence_quote": "на адрес из профиля клиента"},
                {"id": "EC-01", "verdict": "drop", "evidence_quote": "на адрес из профиля клиента"}]}
    dropped = verify.critic(doc, [r], [], L(), doc.numbered(), prompts, [], edge_cases=ec)
    assert [f.fid for f in r.findings] == ["F-002"], "цитата из строки самого дефекта не опровергает его"
    assert ec == [] and r.questions == []
    r.findings[0].fid = "F-001"                  # перенумерация после самопроверки
    verify.restamp_dropped(dropped)
    assert [d["id"] for d in dropped] == ["D-01", "D-02", "D-03"]
    assert "F-001" in dropped[0]["reason"], "ссылка на живой дефект — его итоговый номер"


def test_fix_references_and_marks():
    from reqreview import verify
    from reqreview.agent import Finding, Requirement
    doc = _tmpdoc("# T\n\nСумма передаётся в тенге.\n\nСрок хранения — пять лет.\n")
    f = Finding(criterion="S-CONS", severity="critical", problem="В [L3] сказано «Срок хранения — пять лет»; "
                "пустая строка [L4]. Цитата с опечаткой «Сумма передаеться в тенге».", recommendation="",
                g1=3, g2=3, req_ids=["REQ-01"], fid="S-01", locs=[(3, 3)], quotes=[""])
    st = verify.fix_references(doc, [], [f], [])
    assert "[L5]" in f.problem and "[L4]" not in f.problem and "«Сумма передаётся в тенге»" in f.problem, f.problem
    assert st["refs"] == 1 and st["refs_removed"] == 1 and st["quotes"] == 1
    assert verify.fill_quotes(doc, [f]) == 1 and f.quotes == ["Сумма передаётся в тенге."]
    r = Requirement(id="REQ-01", doc_id=None, text="x", summary="", kind="", g1=3, g2=3,
                    criteria={"CON": "ok", "COM": "ok"})
    r.linked = [f]
    verify.align_marks([r], [{"scenario": "s", "req_ids": ["REQ-01"]}])
    assert r.criteria == {"CON": "fail", "COM": "warn"}


def test_gemini_fast_failover():
    """503 у основной модели: запрос сразу уходит на резервную (без длинных пауз), основная пропускается
    на время cooldown; 404 исключает модель; параллельные запросы безопасны."""
    import threading
    import time as _t
    from concurrent.futures import ThreadPoolExecutor
    from reqreview import llm as L

    class Err(Exception):
        def __init__(self, code):
            super().__init__(f"err {code}")
            self.code = code

    class Models:
        def __init__(self):
            self.log, self.lock = [], threading.Lock()

        def generate_content(self, model, contents, config):
            with self.lock:
                self.log.append(model)
            if model == "main":
                raise Err(503)
            if model == "gone":
                raise Err(404)
            return type("R", (), {"text": '{"ok": 1}'})()

    g = L.GeminiLLM("k", model="main", min_interval=0.0, fallbacks=["gone", "backup"])
    g.client = type("C", (), {"models": Models()})()
    t0 = _t.monotonic()
    with ThreadPoolExecutor(4) as ex:
        out = list(ex.map(lambda _: g.generate_json("t", "s", "p"), range(6)))
    assert all(o == {"ok": 1} for o in out)
    assert _t.monotonic() - t0 < 3, "перегрузка не должна вызывать длинных пауз"
    assert "gone" in g.dead and g.used == {"backup": 6}
    assert g.client.models.log.count("main") <= 4, "перегруженная основная модель пропускается"
    assert g.primary_share() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
