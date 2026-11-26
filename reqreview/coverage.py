"""Честный учёт покрытия: какие символы канонического транскрипта переданы модели и какие реально проверены,
какие обязательные проверки документа выполнены, и итоговый статус запуска.

Все числа считаются кодом (не моделью). Символ — кодовая точка Unicode канонического транскрипта, включая
пробелы и переводы строк; служебные метки строк, текст промптов и диагностические вставки не считаются.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STATUS_RU = {"completed": "Завершена", "partial": "Частичная", "failed": "Не выполнена",
             "na": "Нет оцениваемых требований"}


def unique_covered_characters(intervals: list[tuple[int, int]], total_characters: int) -> int:
    """Длина объединения полуоткрытых интервалов [start, end) канонического текста."""
    if total_characters < 0:
        raise ValueError("Negative transcript length.")
    for start, end in intervals:
        if not 0 <= start <= end <= total_characters:
            raise ValueError("Invalid canonical source span.")
    covered = 0
    ms = me = None
    for start, end in sorted(intervals):
        if start == end:
            continue
        if ms is None:
            ms, me = start, end
        elif start <= me:
            me = max(me, end)
        else:
            covered += me - ms
            ms, me = start, end
    if ms is not None:
        covered += me - ms
    return covered


@dataclass
class Chunk:
    id: str
    kind: str                              # extract | review
    intervals: list[tuple[int, int]]
    req_ids: list[str] = field(default_factory=list)
    status: str = "pending"                # pending | sent | completed | failed
    note: str = ""


class Coverage:
    def __init__(self, doc):
        self.doc = doc
        self.total = doc.canon_len
        self.chunks: list[Chunk] = []
        self.submitted: list[tuple[int, int]] = []
        self.checks: list[dict] = []        # обязательные проверки уровня документа

    def chunk(self, cid: str, kind: str, intervals, req_ids=None) -> Chunk:
        c = Chunk(cid, kind, [(max(0, a), min(self.total, b)) for a, b in intervals], list(req_ids or []))
        self.chunks.append(c)
        return c

    def sent(self, c: Chunk, extra_context: list[tuple[int, int]] | None = None):
        c.status = "sent"
        self.submitted += c.intervals + [(max(0, a), min(self.total, b)) for a, b in (extra_context or [])]

    def done(self, c: Chunk, ok: bool, note: str = ""):
        c.status = "completed" if ok else "failed"
        c.note = note

    def check(self, key: str, title: str, status: str, note: str = ""):
        """status: выполнена | не выполнена | только правила"""
        self.checks = [x for x in self.checks if x["key"] != key]
        self.checks.append({"key": key, "title": title, "status": status, "note": note})

    def whole(self) -> list[tuple[int, int]]:
        return [(0, self.total)]

    def summary(self) -> dict:
        n = self.total
        submitted = unique_covered_characters(self.submitted, n)
        mark = bytearray(n)
        for c in self.chunks:
            if c.kind == "extract" and c.status == "completed":
                for a, b in c.intervals:
                    mark[a:b] = b"\x01" * (b - a)
        for c in self.chunks:                     # требования, чья проверка не удалась, вычитаются
            if c.kind == "review" and c.status != "completed":
                for a, b in c.intervals:
                    mark[a:b] = bytes(b - a)
        sub_mark = bytearray(n)
        for a, b in self.submitted:
            sub_mark[a:b] = b"\x01" * (b - a)
        completed = sum(1 for i in range(n) if mark[i] and sub_mark[i])
        assert 0 <= completed <= submitted <= n, "нарушен инвариант покрытия"
        failed = [c for c in self.chunks if c.status != "completed"]
        return {
            "canonical_chars": n,
            "submitted_chars": submitted,
            "completed_chars": completed,
            "unreviewed_chars": n - completed,
            "completed_pct": (round(100 * completed / n, 1) if n else None),
            "chunks_total": len(self.chunks),
            "chunks_failed": [{"id": c.id, "kind": c.kind, "req_ids": c.req_ids, "note": c.note} for c in failed],
            "checks": self.checks,
            "manifest": [{"id": c.id, "kind": c.kind, "intervals": c.intervals, "req_ids": c.req_ids,
                          "status": c.status, "note": c.note} for c in self.chunks],
        }


def run_status(ing, cov_summary: dict | None, n_reqs: int, n_reviewed: int, mock: bool,
               fatal_reason: str = "") -> tuple[str, list[str]]:
    """Итоговый статус проверки и причины (по-русски)."""
    reasons: list[str] = []
    if fatal_reason:
        return "failed", [fatal_reason]
    if ing is not None and ing.fatal:
        return "failed", ["Входной документ нельзя проверить по существу (см. «Целостность исходных данных»)."]
    if cov_summary is None:
        return "failed", ["Проверка не выполнялась."]
    ext_failed = [c for c in cov_summary["chunks_failed"] if c["kind"] == "extract"]
    ext_total = sum(1 for c in cov_summary["manifest"] if c["kind"] == "extract")
    if ext_total and len(ext_failed) == ext_total:
        return "failed", ["Ни один фрагмент документа не обработан моделью (выделение требований не выполнено)."]
    if n_reqs == 0:
        return "na", ["Документ прочитан, но оцениваемых требований в нём не найдено."]
    if n_reviewed == 0 and not mock:
        return "failed", ["Ни одно требование не проверено моделью."]
    if mock:
        reasons.append("Демо-режим: языковая модель не использовалась, работали только автоматические правила.")
    if ing is not None:
        lim = [i for i in ing.issues if i.limits_review]
        if lim:
            reasons.append(f"Есть ограничения входных данных ({len(lim)}): " + ", ".join(i.id for i in lim) + ".")
    if cov_summary["unreviewed_chars"] > 0 and cov_summary["chunks_failed"]:
        reasons.append(f"Не проверено моделью {cov_summary['unreviewed_chars']} символов извлечённого текста "
                       f"(неудачных фрагментов/пакетов: {len(cov_summary['chunks_failed'])}).")
    bad = [c for c in cov_summary["checks"] if c["status"] != "выполнена"]
    if bad:
        reasons.append("Не выполнены обязательные проверки документа: " + "; ".join(c["title"] for c in bad) + ".")
    return ("partial" if reasons else "completed"), reasons
