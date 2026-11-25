"""Обращение к LLM (Google Gemini) с повторами при лимитах бесплатного тарифа.

Также содержит MockLLM — демо-режим без ключа, который отвечает на основе
правил. Он нужен только чтобы проверить, что агент установлен и работает.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> Any:
    """Достаёт JSON из ответа модели (с учётом ```json ... ``` и мусора вокруг)."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
    if not starts:
        raise ValueError("в ответе модели нет JSON")
    s = min(starts)
    e = max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[s:e + 1])


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class GeminiLLM:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 min_interval: float = 6.0, max_retries: int = 6, temperature: float = 0.1):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("Не установлен пакет google-genai. Выполните: pip install -r requirements.txt") from e
        if not api_key:
            raise LLMError("Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и вставьте ключ.")
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"Google Gemini · {model}"
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.temperature = temperature
        self._last = 0.0
        self.calls = 0

    def _wait_turn(self):
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()

    def _raw(self, system: str, prompt: str) -> str:
        types = self._types
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        delay = 10.0
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                self.calls += 1
                text = resp.text
                if not text:
                    raise LLMError("пустой ответ модели")
                return text
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                msg = str(e)
                if code in (400, 401, 403) or "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
                    raise LLMError(f"Ключ отклонён или модель недоступна ({code}): {msg[:300]}") from e
                if code == 404:
                    raise LLMError(f"Модель «{self.model}» не найдена. Проверьте GEMINI_MODEL в .env. {msg[:200]}") from e
                if attempt == self.max_retries:
                    raise LLMError(f"Не удалось получить ответ после {attempt} попыток: {msg[:300]}") from e
                wait = delay
                m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", msg, re.I)
                if m:
                    wait = max(wait, float(m.group(1)) + 1)
                if code == 429 and "PerDay" in msg:
                    raise LLMError("Исчерпан дневной лимит бесплатного тарифа Gemini. "
                                   "Попробуйте завтра или укажите другую модель в GEMINI_MODEL.") from e
                log(f"   ! ошибка API ({code or type(e).__name__}), попытка {attempt}/{self.max_retries}, "
                    f"жду {wait:.0f} с…")
                time.sleep(wait)
                delay = min(delay * 2, 90)
        raise LLMError("unreachable")

    def generate_json(self, task: str, system: str, prompt: str, payload: Any = None) -> Any:
        text = self._raw(system, prompt)
        try:
            return parse_json(text)
        except (ValueError, json.JSONDecodeError):
            log("   ! модель вернула невалидный JSON, повторяю запрос…")
            text = self._raw(system, prompt + "\n\nВАЖНО: верни ТОЛЬКО валидный JSON без пояснений.")
            try:
                return parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                raise LLMError(f"Модель дважды вернула невалидный JSON: {text[:200]}") from e


class OpenAICompatLLM:
    """Любая модель с OpenAI-совместимым API: OpenAI, Ollama (локально), OpenRouter, vLLM и т.п.
    Работает без дополнительных библиотек (через urllib)."""

    def __init__(self, base_url: str, api_key: str, model: str, label: str,
                 min_interval: float = 0.0, max_retries: int = 5, temperature: float = 0.1):
        if not model:
            raise LLMError("Не задана модель (OPENAI_MODEL / OLLAMA_MODEL) в .env.")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.name = f"{label} · {model}"
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.temperature = temperature
        self._last = 0.0
        self.calls = 0

    def _raw(self, system: str, prompt: str) -> str:
        import urllib.error
        import urllib.request
        body = json.dumps({"model": self.model, "temperature": self.temperature,
                           "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": prompt}]}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 5.0
        for attempt in range(1, self.max_retries + 1):
            dt = time.monotonic() - self._last
            if dt < self.min_interval:
                time.sleep(self.min_interval - dt)
            self._last = time.monotonic()
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self.calls += 1
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                if e.code in (400, 401, 403, 404):
                    raise LLMError(f"Ошибка API {e.code}: {msg}") from e
                if attempt == self.max_retries:
                    raise LLMError(f"Ошибка API {e.code} после {attempt} попыток: {msg}") from e
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                if attempt == self.max_retries:
                    raise LLMError(f"Нет ответа от {self.url}: {e}") from e
            log(f"   ! ошибка API, попытка {attempt}/{self.max_retries}, жду {delay:.0f} с…")
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise LLMError("unreachable")

    generate_json = GeminiLLM.generate_json


class MockLLM:
    """Демо-режим без ключа: имитирует ответы модели с помощью эвристик и правил.
    Отчёт в этом режиме НЕ является полноценной экспертной проверкой."""

    name = "ДЕМО-режим (без LLM, только правила)"
    model = "mock"

    def __init__(self):
        self.calls = 0

    def generate_json(self, task: str, system: str, prompt: str, payload: Any = None) -> Any:
        self.calls += 1
        from . import rules
        if task == "extract":
            out = []
            id_re = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?\[?([A-Za-zА-Яа-яЁё]{1,6}[-_.]?\d+(?:\.\d+)*)\]?\.?(?:\*\*)?[\s.:)\-–—]+(.*)$")
            for gid, text in payload["lines"]:
                if text.lstrip().startswith("#") or text.lstrip().startswith("|---"):
                    continue
                cells = [c.strip() for c in text.strip().strip("|").split("|")] if text.strip().startswith("|") else None
                doc_id, body = None, text.strip()
                if cells and len(cells) >= 2:
                    m = re.match(r"^[A-Za-zА-Яа-яЁё]{1,6}[-_.]?\d+(?:\.\d+)*$", cells[0])
                    if not m:
                        continue
                    doc_id, body = cells[0], cells[1]
                else:
                    m = id_re.match(body)
                    if m:
                        doc_id, body = m.group(1), m.group(2).strip()
                if not body:
                    continue
                if doc_id or rules.MODAL_RE.search(body):
                    out.append({"doc_id": doc_id, "text": body, "line_start": gid, "line_end": gid,
                                "summary": body[:80], "kind": "не определено (демо)"})
            return {"requirements": out}
        if task == "review":
            res = []
            for r in payload["requirements"]:
                hits = rules.check_requirement(r["text"])
                crit = {c: "na" for c in ("NEC", "APP", "UNA", "COM", "SIN", "FEA", "VER", "COR", "CNF")}
                for h in hits:
                    crit[h["criterion"]] = "fail"
                res.append({"id": r["id"], "criteria": crit, "findings": [
                    {k: h[k] for k in ("criterion", "severity", "quote", "problem", "recommendation")} for h in hits]})
            return {"results": res}
        if task == "set":
            return {"findings": [], "set_criteria": {}}
        raise LLMError(f"MockLLM: неизвестная задача {task}")
