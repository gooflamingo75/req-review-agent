"""Сравнение двух прогонов («до» и «после» исправления документа).

Номера дефектов между прогонами не совпадают (F-/S- присваиваются заново), поэтому дефекты сопоставляются
по смыслу: одинаковый тег первопричины и близкий текст, либо очень близкий текст. Результат — словарь
data["comparison"], который отчёт показывает на первой странице и в приложении.
"""
from __future__ import annotations

from .consolidate import similarity

SEVS = ("CRITICAL", "MAJOR", "MINOR")


def _findings(data: dict) -> list[dict]:
    out = [dict(f, _req=r["id"]) for r in data.get("requirements") or [] for f in r.get("findings") or []]
    return out + list(data.get("set_findings") or [])


def _match(a: dict, b: dict) -> float:
    sim = similarity(a.get("problem", ""), b.get("problem", ""))
    if a.get("tag") and a.get("tag") == b.get("tag") and a.get("tag") not in ("AMBIGUITY", "INCOMPLETE", "UNTESTABLE"):
        sim += 0.25
    return sim


def compare(old: dict, new: dict) -> dict:
    old_f, new_f = _findings(old), _findings(new)
    pairs = sorted(((_match(a, b), i, j) for i, a in enumerate(old_f) for j, b in enumerate(new_f)), reverse=True)
    used_o, used_n, matched = set(), set(), []
    for score, i, j in pairs:
        if score < 0.5:
            break
        if i in used_o or j in used_n:
            continue
        used_o.add(i)
        used_n.add(j)
        matched.append((old_f[i], new_f[j]))

    def brief(f):
        return {"id": f["id"], "severity": f["severity"], "problem": f.get("problem", ""), "tag": f.get("tag") or ""}

    def counts(fs):
        return {s: sum(1 for f in fs if f["severity"] == s) for s in SEVS}
    return {
        "previous": {"date": old.get("date"), "file": (old.get("document") or {}).get("file"),
                     "sha256": (old.get("document") or {}).get("sha256"), "agent_version": old.get("agent_version"),
                     "counts": counts(old_f), "requirements": len(old.get("requirements") or [])},
        "current": {"counts": counts(new_f), "requirements": len(new.get("requirements") or [])},
        "same_document": (old.get("document") or {}).get("sha256") == (new.get("document") or {}).get("sha256"),
        "resolved": [brief(f) for i, f in enumerate(old_f) if i not in used_o],
        "new": [brief(f) for j, f in enumerate(new_f) if j not in used_n],
        "still_open": [dict(brief(b), was=a["id"]) for a, b in matched],
    }
