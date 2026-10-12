"""Vision-language model client for the three GenAI hooks.

Supports OpenRouter (OpenAI-compatible) and Google Gemini, chosen by which key is
present in `.env`. Every call is cached on disk and logged (prompt, output,
tokens, cost), so experiments are repeatable and their cost can be reported.
With no key the client reports `available = False` and the hooks stay off.
"""
import base64
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_MODELS = {"openrouter": "anthropic/claude-haiku-5.5", "gemini": "gemini-3.5-flash-lite"}
CACHE_DIR = ROOT / "cache" / "vlm"
LOG_FILE = ROOT / "logs" / "vlm_calls.jsonl"


_LOG_LOCK = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def encode_image(bgr, long_side=768, quality=88):
    """JPEG-encode an image for the API, downsized to keep the token cost low."""
    h, w = bgr.shape[:2]
    s = long_side / max(h, w)
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def parse_json(text):
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


class VLM:
    def __init__(self, model=None, provider=None, budget_usd=None):
        keys = {"openrouter": os.getenv("OPENROUTER_API_KEY", "").strip(),
                "gemini": os.getenv("GEMINI_API_KEY", "").strip()}
        self.provider = provider or next((p for p in ("openrouter", "gemini") if keys[p]), None)
        self.key = keys.get(self.provider, "")
        self.model = model or os.getenv("DIALREADER_VLM_MODEL", "").strip() or DEFAULT_MODELS.get(self.provider, "")
        self.budget = float(budget_usd if budget_usd is not None else os.getenv("DIALREADER_VLM_BUDGET", "2.0"))
        # Reasoning effort: "none" (answer directly), "low", "medium" or "high".
        self.reasoning = os.getenv("DIALREADER_VLM_REASONING", "").strip() or "low"
        self.calls = self.cached_calls = 0
        self.cost = 0.0

    @property
    def available(self):
        return bool(self.provider and self.key)

    # ------------------------------------------------------------ bookkeeping

    def spent_total(self):
        """Money spent on live calls so far, across all runs (from the log)."""
        if not LOG_FILE.exists():
            return 0.0
        total = 0.0
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "call" and not rec.get("cached"):
                    total += rec.get("cost_usd") or 0.0
        return total

    def log(self, record):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------ the call

    def ask(self, prompt, images, schema, tag="", max_tokens=700):
        """Send a prompt with images, get back a JSON object that follows `schema`.

        Returns a dict: result (parsed JSON or None), raw text, tokens, cost_usd,
        cached, seconds, id. Raises BudgetExceeded before a live call that would
        go past the spending limit.
        """
        blobs = [encode_image(im) for im in images]
        digest = hashlib.sha256()
        for part in (self.model, self.reasoning, prompt, json.dumps(schema, sort_keys=True), *blobs):
            digest.update(part if isinstance(part, bytes) else part.encode("utf-8"))
        call_id = digest.hexdigest()[:20]
        cache_file = CACHE_DIR / f"{call_id}.json"
        if cache_file.exists():
            out = json.loads(cache_file.read_text(encoding="utf-8"))
            out["cached"] = True
            self.cached_calls += 1
        else:
            if not self.available:
                raise RuntimeError("no VLM key configured (see .env)")
            if self.spent_total() >= self.budget:
                raise BudgetExceeded(f"VLM spending limit of ${self.budget:.2f} reached")
            t0 = time.perf_counter()
            raw, usage = (self._openrouter if self.provider == "openrouter" else self._gemini)(
                prompt, blobs, schema, max_tokens)
            out = {"id": call_id, "raw": raw, "result": parse_json(raw), "cached": False,
                   "seconds": round(time.perf_counter() - t0, 2), **usage}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(out), encoding="utf-8")
            self.calls += 1
            self.cost += out.get("cost_usd") or 0.0
        self.log({"event": "call", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": tag, "model": self.model,
                  "id": call_id, "prompt": prompt, "n_images": len(images), "output": out["raw"],
                  "tokens_in": out.get("tokens_in"), "tokens_out": out.get("tokens_out"),
                  "cost_usd": out.get("cost_usd"), "cached": out["cached"], "seconds": out.get("seconds")})
        return out

    def _openrouter(self, prompt, blobs, schema, max_tokens):
        content = [{"type": "text", "text": prompt}]
        for blob in blobs:
            url = "data:image/jpeg;base64," + base64.b64encode(blob).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        # Reasoning tokens count against max_tokens, so leave room for them before the answer.
        thinking = {"none": 0, "low": 1500, "medium": 4000, "high": 8000}.get(self.reasoning, 1500)
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens + thinking, "usage": {"include": True},
                "reasoning": {"enabled": False} if thinking == 0 else {"effort": self.reasoning},
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
        if thinking == 0:
            body["temperature"] = 0
        headers = {"Authorization": f"Bearer {self.key}", "X-Title": "DialReader"}
        for attempt in range(4):
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body,
                              timeout=120)
            if r.status_code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            break
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:300]}")
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: {str(data['error'])[:300]}")
        usage = data.get("usage") or {}
        return data["choices"][0]["message"].get("content") or "", {
            "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
            "cost_usd": usage.get("cost")}

    def _gemini(self, prompt, blobs, schema, max_tokens):
        parts = [{"text": prompt + "\nAnswer with one JSON object matching this JSON schema:\n" + json.dumps(schema)}]
        parts += [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(b).decode("ascii")}}
                  for b in blobs]
        body = {"contents": [{"parts": parts}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens,
                                     "responseMimeType": "application/json"}}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        r = requests.post(url, headers={"x-goog-api-key": self.key}, json=body, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:300]}")
        data = r.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"].get("parts", []))
        usage = data.get("usageMetadata") or {}
        return text, {"tokens_in": usage.get("promptTokenCount"), "tokens_out": usage.get("candidatesTokenCount"),
                      "cost_usd": None}
