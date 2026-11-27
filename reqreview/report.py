"""Запись отчёта: одна модель данных (JSON) → Markdown, PDF и report.json.

Отчёт строится модулем present.py (обложка с решениями, затем разделы 1–6 по шаблону организаторов и
приложения); records.py до вывода строит решения, план, вопросы, тест-кейсы, статистику и проверяет согласованность.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import __team__
from .agent import Finding, Result
from .coverage import STATUS_RU as RUN_STATUS_RU
from .criteria import SEVERITY
from .ingest import human_size


def _loc(res: Result, f: Finding) -> str:
    if f.g1 is None:
        return "—"
    if f.locs and len(f.locs) > 1:
        return "; ".join(res.doc.location(a, b, link=True) for a, b in f.locs)
    return res.doc.location(f.g1, f.g2, link=True)


def to_json(res: Result, agent_version: str) -> str:
    def fd(f: Finding):
        return {"id": f.fid, "severity": SEVERITY[f.severity], "criterion": f.criterion, "tag": f.tag,
                "type": f.type, "part": res.doc.part_of(f.g1) if f.g1 is not None else "",
                "requirements": f.req_ids, "location": plain(_loc(res, f)), "quotes": f.quotes or ([f.quote] if f.quote else []),
                "quote_verified": f.verified, "problem": f.problem, "recommendation": f.recommendation,
                "source": f.source, "places": max(1, len(f.locs)), "merged_duplicates": f.merged,
                "why_critical": f.why or None, "caveat": f.caveat or None, "also": f.also,
                "criteria": f.criteria or [f.criterion], "classification": f.kind, "observations": f.obs,
                "lines": [list(x) for x in (f.locs or ([(f.g1, f.g2)] if f.g1 is not None else []))]}
    data = {
        "document": {"title": res.doc.title, "file": res.doc.path.name, "sha256": res.doc.sha256},
        "model": res.model_name, "agent_version": agent_version, "team": __team__,
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
        from . import __team__
        present.write_pdf(data, doc, pdf_path, footer=f"req-review-agent {agent_version} · {__team__}")
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
