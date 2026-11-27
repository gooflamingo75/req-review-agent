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


# Порядок важен: сначала полноценные flash-модели, облегчённые «lite» — только в крайнем случае
DEFAULT_GEMINI_FALLBACKS = ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
OVERLOAD_COOLDOWN_SEC = 60.0     # перегруженная (503) модель пропускается столько секунд, затем пробуем снова
OVERLOAD_COOLDOWN_MAX = 300.0
PRIMARY_RETRY_SEC = 5.0          # перегрузка основной модели часто кратковременна: одна короткая пауза и повтор    # при повторных перегрузках пауза растёт до этого предела
REQUEST_TIMEOUT_SEC = 240        # один запрос не может «висеть» дольше


def _reason(code, e) -> str:
    name = type(e).__name__.lower()
    if code in (500, 502, 503, 504) or "unavailable" in str(e).lower() or "overloaded" in str(e).lower():
        return f"{code or 503} перегрузка сервиса"
    if "timeout" in name or "timed out" in str(e).lower() or "deadline" in str(e).lower():
        return "тайм-аут ответа"
    if isinstance(e, LLMError):
        return "пустой ответ"
    return f"{code or type(e).__name__} ошибка соединения"


class GeminiLLM:
    """Gemini с быстрым переключением на резервные модели.

    Перегрузка (503) у модели не пережидается длинными паузами: запрос сразу уходит на следующую модель,
    а перегруженная модель пропускается следующими запросами в течение короткой паузы (cooldown), после чего
    основная модель снова получает запросы первой. Недоступные модели (404, дневной лимит) выбывают до конца
    прогона. Класс потокобезопасен: агент отправляет независимые запросы параллельно."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash",
                 min_interval: float = 6.0, max_retries: int = 6, temperature: float = 0.1,
                 fallbacks: list[str] | None = None):
        import threading
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("Не установлен пакет google-genai. Выполните: pip install -r requirements.txt") from e
        if not api_key:
            raise LLMError("Не задан GEMINI_API_KEY. Скопируйте .env.example в .env и вставьте ключ.")
        self._types = types
        try:
            self.client = genai.Client(api_key=api_key,
                                       http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SEC * 1000))
        except Exception:  # noqa: BLE001 — старые версии SDK без HttpOptions
            self.client = genai.Client(api_key=api_key)
        chain = [model] + list(DEFAULT_GEMINI_FALLBACKS if fallbacks is None else fallbacks)
        self.models = list(dict.fromkeys(m.strip() for m in chain if m and m.strip()))
        self.dead: set[str] = set()                 # недоступны до конца прогона (404, дневной лимит)
        self.cool: dict[str, float] = {}            # перегружена до момента времени (monotonic)
        self.strikes: dict[str, int] = {}           # сколько раз подряд модель была перегружена
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.temperature = temperature
        self._last: dict[str, float] = {}           # пауза между запросами — у каждой модели свой лимит
        self._lock = threading.Lock()
        self.calls = 0
        self.used: dict[str, int] = {}
        self.failures: dict[str, dict[str, int]] = {}   # модель → причина отказа → сколько раз (для паспорта)
        self.primary_only = False                         # True — не уходить на резервные модели
        self.probe_note: str | None = None
        self._retrying = False

    def _fail(self, model: str, reason: str) -> None:
        with self._lock:
            d = self.failures.setdefault(model, {})
            d[reason] = d.get(reason, 0) + 1

    def failure_summary(self, model: str | None = None) -> str:
        """«503 перегрузка ×4, тайм-аут ×2» — почему модель не ответила."""
        d = self.failures.get(model or self.primary) or {}
        return ", ".join(f"{k} ×{v}" for k, v in sorted(d.items(), key=lambda x: -x[1]))

    def probe(self) -> str | None:
        """Короткий пробный запрос к основной модели до начала проверки: недоступную модель (404, нет доступа
        у ключа, дневной лимит) видно сразу, а не после прогона. Возвращает текст проблемы или None."""
        types = self._types
        cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
        last = ""
        for wait in (0, 10, 25):
            if wait:
                log(f"   ! основная модель {self.primary} не ответила ({last}) — повтор через {wait} с…")
                time.sleep(wait)
            try:
                self.client.models.generate_content(model=self.primary, contents='Ответь JSON: {"ok": true}',
                                                    config=cfg)
                return None
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                msg = str(e)
                if code in (401, 403) or "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
                    return None                    # ключ проверит первый настоящий запрос с понятной ошибкой
                if code == 404 or (code == 400 and "model" in msg.lower()):
                    self._fail(self.primary, "404 модель недоступна для ключа")
                    self._kill(self.primary, "недоступна (404)")
                    return (f"основная модель {self.primary} недоступна для этого ключа (404) — укажите в .env "
                            "GEMINI_MODEL, которая есть в вашем аккаунте (список: https://ai.google.dev/gemini-api/docs/models)")
                if code == 429 and "PerDay" in msg:
                    self._fail(self.primary, "дневной лимит исчерпан")
                    self._kill(self.primary, "исчерпала дневной лимит бесплатного тарифа")
                    return f"у основной модели {self.primary} исчерпан дневной лимит бесплатного тарифа"
                last = f"{code or type(e).__name__}"
                self._fail(self.primary, _reason(code, e))
        return f"основная модель {self.primary} перегружена ({last}) — сейчас будут использоваться резервные модели"

    @property
    def model(self) -> str:
        return self.models[0]

    @property
    def name(self) -> str:
        if not self.used:
            return f"Google Gemini · {self.models[0]}"
        if len(self.used) == 1:
            return f"Google Gemini · {next(iter(self.used))}"
        return "Google Gemini · " + ", ".join(f"{m} ({n} запр.)" for m, n in self.used.items())

    @property
    def primary(self) -> str:
        return self.models[0]

    def primary_share(self) -> float | None:
        total = sum(self.used.values())
        return (self.used.get(self.primary, 0) / total) if total else None

    def _pick(self, skip: set[str]) -> tuple[str | None, float]:
        """Первая доступная модель по приоритету. Если все живые модели «остывают» — та, что освободится
        раньше всех, и сколько её ждать."""
        now = time.monotonic()
        with self._lock:
            alive = [m for m in (self.models[:1] if self.primary_only else self.models)
                     if m not in self.dead and m not in skip]
            if not alive:
                return None, 0.0
            for m in alive:
                if self.cool.get(m, 0) <= now:
                    return m, 0.0
            m = min(alive, key=lambda x: self.cool.get(x, 0))
            return m, self.cool[m] - now

    def _wait_turn(self, model: str):
        with self._lock:
            now = time.monotonic()
            start = max(now, self._last.get(model, 0) + self.min_interval)
            self._last[model] = start
        if start > now:
            time.sleep(start - now)

    def _overloaded(self, model: str, code) -> None:
        with self._lock:
            n = self.strikes.get(model, 0) + 1
            self.strikes[model] = n
            pause = min(OVERLOAD_COOLDOWN_SEC * (2 ** (n - 1)), OVERLOAD_COOLDOWN_MAX)
            if model == self.models[0]:
                pause = min(pause, OVERLOAD_COOLDOWN_SEC)   # основную модель пробуем снова не позже чем через 60 с
            self.cool[model] = time.monotonic() + pause
        log(f"   ! {model} перегружена ({code}) — запрос сразу ушёл на резервную модель; "
            f"{model} пропускается {pause:.0f} с")

    def _kill(self, model: str, reason: str) -> None:
        with self._lock:
            new = model not in self.dead
            self.dead.add(model)
        if new:
            log(f"   ! модель {model} {reason} — исключена до конца прогона")

    def _raw(self, system: str, prompt: str) -> str:
        types = self._types
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json",
        )
        attempt, last_err, rate_wait = 0, "", 5.0
        skip: set[str] = set()                     # модели, уже отказавшие в ЭТОМ запросе
        primary_retry = True                       # основной модели — один быстрый повтор до резервной
        while attempt < self.max_retries * 3:
            model, wait = self._pick(skip)
            if model is None:
                # все модели отказали в этом запросе — короткая пауза и новый круг с основной модели
                skip.clear()
                model, wait = self._pick(skip)
                if model is None:
                    raise LLMError(f"Ни одна из моделей {', '.join(self.models)} не доступна. {last_err}")
                wait = max(wait, 5.0)
            if wait > 0:
                wait = min(wait, 30.0)
                log(f"   ! все модели перегружены, жду {wait:.0f} с…")
                time.sleep(wait)
            attempt += 1
            self._wait_turn(model)
            try:
                resp = self.client.models.generate_content(model=model, contents=prompt, config=cfg)
                with self._lock:
                    self.calls += 1
                text = resp.text
                if not text:
                    raise LLMError("пустой ответ модели")
                with self._lock:
                    self.used[model] = self.used.get(model, 0) + 1
                    self.strikes[model] = 0
                return text
            except Exception as e:  # noqa: BLE001
                code = getattr(e, "code", None) or getattr(e, "status_code", None)
                msg = str(e)
                last_err = f"{code}: {msg[:300]}"
                if code in (401, 403) or "API_KEY_INVALID" in msg or "PERMISSION_DENIED" in msg:
                    raise LLMError(f"Ключ отклонён ({code}). Проверьте GEMINI_API_KEY в .env. {msg[:300]}") from e
                if code == 404 or (code == 400 and "model" in msg.lower() and "not" in msg.lower()):
                    self._fail(model, "404 модель недоступна для ключа")
                    self._kill(model, "недоступна (404)")
                    continue
                if code == 400:
                    raise LLMError(f"Запрос отклонён (400): {msg[:300]}") from e
                if code == 429 and "PerDay" in msg:
                    self._fail(model, "дневной лимит исчерпан")
                    self._kill(model, "исчерпала дневной лимит бесплатного тарифа")
                    continue
                if code == 429:
                    # поминутный лимит: ждём, сколько просит сервис, но не дольше 60 с
                    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", msg, re.I)
                    w = min(float(m.group(1)) + 1, 60.0) if m else rate_wait
                    rate_wait = min(rate_wait * 2, 60.0)
                    self._fail(model, "429 лимит в минуту")
                    log(f"   ! лимит запросов в минуту у {model} (429), жду {w:.0f} с…")
                    time.sleep(w)
                    continue
                # 5xx, тайм-аут, обрыв соединения, пустой ответ — пробуем следующую модель
                self._fail(model, _reason(code, e))
                with self._lock:                   # быстрый повтор — только одним запросом за раз
                    first = model == self.models[0] and primary_retry and not self._retrying
                    if first:
                        self._retrying = True
                if model == self.models[0] and (first or self.primary_only):
                    primary_retry = False
                    w = PRIMARY_RETRY_SEC if not self.primary_only else min(rate_wait * 2, 60.0)
                    rate_wait = w
                    log(f"   ! {model} не ответила ({code or type(e).__name__}) — повтор через {w:.0f} с…")
                    time.sleep(w)
                    if first:
                        with self._lock:
                            self._retrying = False
                    continue
                self._overloaded(model, code or type(e).__name__)
                skip.add(model)
        raise LLMError(f"Не удалось получить ответ после {attempt} попыток: {last_err}")

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
                crit = {c: "na" for c in ("COM", "UNA", "CON", "VER", "FEA", "NEC", "APP", "SIN", "COR", "CNF")}
                for h in hits:
                    crit[h["criterion"]] = "fail"
                res.append({"id": r["id"], "criteria": crit, "findings": [
                    {k: h[k] for k in ("criterion", "severity", "quote", "problem", "recommendation")} for h in hits]})
            return {"results": res}
        if task.startswith("set"):
            return {"findings": [], "set_criteria": {}}
        raise LLMError(f"MockLLM: неизвестная задача {task}")
