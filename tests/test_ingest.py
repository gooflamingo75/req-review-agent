"""Тесты приёма документа, OCR, покрытия, статусов и PDF-отчёта (фикстуры генерируются в тесте).

Запуск:  python tests/test_ingest.py   (или  python -m pytest -q)
Тесты OCR пропускаются с пометкой SKIP, если Tesseract не установлен.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fixtures as fx  # noqa: E402
from reqreview.agent import run  # noqa: E402
from reqreview.coverage import unique_covered_characters  # noqa: E402
from reqreview.ingest import InputError  # noqa: E402
from reqreview.llm import LLMError, MockLLM  # noqa: E402
from reqreview.loader import load_document  # noqa: E402
from reqreview.ocr import Ocr  # noqa: E402
from reqreview.report import write, write_diagnostic  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="rra-test-"))
SAMPLE = ROOT / "examples" / "sample_requirements.md"
_OCR = Ocr()
HAS_OCR = _OCR.available


class Skip(Exception):
    pass


def _fails(path: Path, status: str):
    try:
        load_document(path)
    except InputError as e:
        assert e.ingestion.status == status, (e.ingestion.status, status)
        assert e.ingestion.issues and e.ingestion.issues[-1].next_action
        return e.ingestion
    raise AssertionError(f"{path.name}: ожидалась ошибка приёма {status}")


def test_unique_covered_characters():
    assert unique_covered_characters([(0, 10), (5, 15), (20, 25), (25, 25)], 30) == 20
    assert unique_covered_characters([], 0) == 0
    try:
        unique_covered_characters([(0, 40)], 30)
    except ValueError:
        pass
    else:
        raise AssertionError("интервал вне текста должен отклоняться")


def test_clean_text_document():
    doc = load_document(SAMPLE)
    ing = doc.ingest
    assert ing.status == "ok" and ing.ocr_state == "не требовался"
    data = SAMPLE.read_bytes()
    assert ing.size == len(data)
    import hashlib
    assert ing.sha256 == hashlib.sha256(data).hexdigest()
    assert doc.canon_len == len(data.decode("utf-8-sig").replace("\r\n", "\n").rstrip("\n")) or doc.canon_len > 0


def test_empty_inputs():
    (TMP / "zero.md").write_bytes(b"")
    (TMP / "spaces.txt").write_text("  \n\t\n ", encoding="utf-8")
    _fails(TMP / "zero.md", "empty_file")
    _fails(TMP / "spaces.txt", "empty_content")


def test_bad_pdfs():
    _fails(fx.truncated_pdf(TMP / "trunc.pdf"), "unreadable")
    _fails(fx.encrypted_pdf(TMP / "enc.pdf"), "encrypted")
    _fails(fx.fake_pdf(TMP / "fake.pdf"), "unsupported")


def test_native_pdf_no_ocr():
    doc = load_document(fx.native_pdf(TMP / "native.pdf"))
    assert doc.ingest.status == "ok" and doc.ingest.ocr_state == "не требовался"
    assert all(ln.src == "native" for ln in doc.lines)
    assert doc.location(3).startswith("с. 1, L0003")


def test_scanned_pdf():
    path = fx.scanned_pdf(TMP / "scan.pdf")
    if not HAS_OCR:
        ing = _fails(path, "unreadable")
        assert ing.ocr_state in ("недоступен", "отключён")
        raise Skip("Tesseract не установлен — проверен только диагностический путь")
    doc = load_document(path)
    p = doc.ingest.pages[0]
    assert p.method == "ocr" and p.status in ("readable", "uncertain") and p.ocr_chars > 0
    assert any("50000" in ln.text and ln.src == "ocr" for ln in doc.lines)


def test_mixed_pdf_and_blank_page():
    path = fx.mixed_pdf(TMP / "mixed.pdf")
    if not HAS_OCR:
        doc = load_document(path)
        assert doc.ingest.status == "partial" and doc.ingest.pages[0].status == "partial"
        assert doc.ingest.pages[1].status == "blank"
        raise Skip("Tesseract не установлен — проверено: страница помечена как частично прочитанная")
    doc = load_document(path)
    p1, p2 = doc.ingest.pages
    assert p1.method == "mixed", p1
    assert p2.status == "blank"
    heads = [ln for ln in doc.lines if "смешанное" in ln.text]
    assert len(heads) == 1, "текстовый заголовок не должен дублироваться OCR"


def test_ocr_disabled_is_explicit():
    os.environ["OCR_ENABLED"] = "off"
    try:
        ing = _fails(fx.scanned_pdf(TMP / "scan2.pdf"), "unreadable")
        assert ing.ocr_state == "отключён"
        assert any("не прочитана" in i.effect or "не проверено" in i.effect for i in ing.issues)
    finally:
        os.environ.pop("OCR_ENABLED", None)


def test_missing_appendix_reference():
    p = TMP / "ref.md"
    p.write_text("# ТЗ\n\n**Ф-1.** Система должна считать лимит по правилам, см. Приложение Б.\n", encoding="utf-8")
    doc = load_document(p)
    assert doc.ingest.status == "partial"
    assert any("Приложение Б" in i.observation for i in doc.ingest.issues)


class FlakyLLM(MockLLM):
    """Модель, у которой падает первый пакет проверки требований."""

    def __init__(self):
        super().__init__()
        self.model = "flaky"
        self.name = "flaky"
        self.n = 0

    def generate_json(self, task, system, prompt, payload=None):
        if task == "review":
            self.n += 1
            if self.n == 1:
                raise LLMError("503 перегрузка (тест)")
        if task in ("critic", "why_critical"):
            return {}
        return super().generate_json(task, system, prompt, payload)


def test_partial_coverage_and_status():
    doc = load_document(SAMPLE)
    res = run(doc, FlakyLLM(), batch_size=3)
    cov = res.coverage
    assert 0 <= cov["completed_chars"] <= cov["submitted_chars"] <= cov["canonical_chars"]
    assert cov["unreviewed_chars"] > 0 and cov["chunks_failed"]
    assert res.status == "partial", res.status
    files = write(res, TMP / "partial.md", "test", "sample.md")
    md = files["md"].read_text(encoding="utf-8")
    assert "Частичная" in md and "общий вердикт «готов к разработке»" in md
    assert "✅ **ДОПУСКАЕТСЯ" not in md


def test_full_report_pdf_and_links():
    doc = load_document(SAMPLE)
    res = run(doc, MockLLM(), batch_size=8)
    files = write(res, TMP / "full.pdf", "test", "sample.md")
    assert files["pdf"] and files["pdf"].exists(), files.get("pdf_error")
    import pymupdf
    with pymupdf.open(str(files["pdf"])) as d:
        text = "".join(p.get_text() for p in d)
        assert "Вердикт" in d[0].get_text() and "Исходный текст" in text and "₸" in text
        assert "1. Паспорт прогона" in d[1].get_text(), "после обложки — разделы шаблона"
        links = [lk for p in d for lk in p.get_links() if lk.get("kind") == pymupdf.LINK_GOTO]
        assert links and all(0 <= lk["page"] < d.page_count for lk in links)
        assert d.get_toc()
    md = files["md"].read_text(encoding="utf-8")
    order = ["## Вердикт", "## 1. Паспорт прогона", "## 2. Нарезка и инвентаризация", "## 3. Детальная проверка",
             "## 4. Проверка набора", "## 5. Сводная статистика", "## 6. Итоговый вердикт", "План исправлений по ролям",
             "Вопросы к аналитику", "Тест-кейсы", "# Приложения", "Приложение А. Полная матрица",
             "Приложение Б. Критерии и методика", "Приложение Д. Исходный текст"]
    pos = [md.index(x) for x in order]
    assert pos == sorted(pos), list(zip(order, pos))
    # каждая строка исходного файла есть в приложении
    assert md.count("\nL0") >= len(doc.lines)
    # требования без замечаний не получают отдельной карточки
    clean = [r for r in res.requirements if not r.findings and not r.linked and not r.questions
             and not any(st in ("warn", "fail") for st in r.criteria.values())]
    for r in clean:
        assert f"#### {r.id}" not in md


def test_rerender_from_json():
    """Отчёт перестраивается из сохранённого JSON без модели и совпадает с исходным."""
    import json
    from reqreview.report import write_data
    doc = load_document(SAMPLE)
    res = run(doc, MockLLM(), batch_size=8)
    f1 = write(res, TMP / "a.md", "test", "sample.md")
    data = json.loads(f1["json"].read_text(encoding="utf-8"))
    f2 = write_data(data, load_document(SAMPLE), TMP / "b.md", "test")
    assert f1["md"].read_text(encoding="utf-8") == f2["md"].read_text(encoding="utf-8")
    assert f2["pdf"] and f2["pdf"].exists()


def test_compare_runs():
    """Повторный прогон после исправления: закрытые, новые и оставшиеся дефекты; блок на первой странице."""
    import json
    from reqreview.compare import compare
    from reqreview.report import write_data
    doc = load_document(SAMPLE)
    res = run(doc, MockLLM(), batch_size=8)
    f1 = write(res, TMP / "before.md", "test", "sample.md")
    old = json.loads(f1["json"].read_text(encoding="utf-8"))
    new = json.loads(json.dumps(old))
    gone = new["set_findings"].pop(0)
    c = compare(old, new)
    assert [x["id"] for x in c["resolved"]] == [gone["id"]] and not c["new"]
    assert c["same_document"] and len(c["still_open"]) == len(old["set_findings"]) - 1 + sum(
        len(r["findings"]) for r in old["requirements"])
    new["comparison"] = c
    f2 = write_data(new, doc, TMP / "after.md", "test")
    md = f2["md"].read_text(encoding="utf-8")
    assert "Сравнение с прогоном" in md.split("План исправлений")[0] and "Что изменилось с прошлого прогона" in md


def test_ocr_uncertain_token_not_critical():
    """Критичный вывод, держащийся только на строке OCR с невысокой уверенностью, понижается до MAJOR."""
    from reqreview.agent import Finding
    from reqreview.verify import ocr_guard
    path = fx.scanned_pdf(TMP / "scan3.pdf")
    if not HAS_OCR:
        raise Skip("Tesseract не установлен")
    doc = load_document(path)
    ln = next(x for x in doc.lines if "50000" in x.text)
    f = Finding(criterion="CON", severity="critical", problem="p", recommendation="r", quote=ln.text, g1=ln.gid, g2=ln.gid)
    assert ocr_guard(doc, [f], min_conf=101) == 1 and f.severity == "major" and "OCR" in f.caveat


def test_diagnostic_report():
    (TMP / "zero2.md").write_bytes(b"")
    try:
        load_document(TMP / "zero2.md")
    except InputError as e:
        files = write_diagnostic(e.ingestion, "failed", [str(e)], TMP / "diag.md", "test", "zero2.md")
        md = files["md"].read_text(encoding="utf-8")
        assert "не выставляется" in md and "ING-001" in md
        assert files["pdf"] and files["pdf"].exists(), files.get("pdf_error")
        return
    raise AssertionError("ожидалась ошибка приёма")


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except Skip as s:
                print(f"SKIP {name}: {s}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
