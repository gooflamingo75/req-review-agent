"""Мелкие вспомогательные функции."""
from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_CHARMAP = str.maketrans({
    "ё": "е", "Ё": "е",
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "`": "'",
    "–": "-", "—": "-", "−": "-", " ": " ",
})


def collapse_ws(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def norm_chars(s: str) -> str:
    """Посимвольная нормализация (длина строки сохраняется)."""
    s = s.translate(_CHARMAP)
    out = []
    for ch in s:
        lo = ch.lower()
        out.append(lo if len(lo) == 1 else ch)
    return "".join(out)


def norm(s: str) -> str:
    return norm_chars(collapse_ws(s))


def short(s: str, n: int = 110) -> str:
    s = collapse_ws(s)
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def md_cell(s: str) -> str:
    """Безопасный текст для ячейки Markdown-таблицы."""
    return collapse_ws(str(s)).replace("|", "\\|")
