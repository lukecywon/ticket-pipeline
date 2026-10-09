"""OpenRouter transport with a durable, sanitized audit record for each attempt."""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from .data import DataError, loads_json, stable_hash, write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class ProviderError(RuntimeError):
    def __init__(self, message, output_artifact=None):
        super().__init__(message)
        self.output_artifact = output_artifact


@dataclass
class Completion:
    content: str
    output_artifact: str


class ArtifactStore:
    def __init__(self, output_dir: Path, *, overwrite=False):
        self.root = Path(output_dir)
        if self.root.exists() and not self.root.is_dir():
            raise DataError(f"Output path is not a directory: {self.root}")
        if self.root.exists() and any(self.root.iterdir()) and not overwrite:
            raise DataError(f"Output directory is not empty: {self.root}. Choose another directory or pass --overwrite.")
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid4().hex
        self.call_count = 0
        self.errors = []
        # Fresh log, even on an explicit overwrite. Old raw files remain unreferenced.
        (self.root / "llm_calls.jsonl").write_text("", encoding="utf-8")

    def write(self, name, value):
        write_json(self.root / name, value)

    def error(self, ticket_id, stage, kind, message, **details):
        self.errors.append({"ticket_id": ticket_id, "stage": stage, "kind": kind, "message": message, **details})
        self.write("errors.json", self.errors)

    def save_call(self, *, ticket_id, stage, timestamp, request, response_text, status_code, error, resolved_model, upstream_provider):
        self.call_count += 1
        relative = f"raw_outputs/{self.run_id}/{self.call_count:06d}_{stage}.json"
        artifact = {
            "ticket_id": ticket_id, "stage": stage, "timestamp": timestamp,
            "request": request, "status_code": status_code,
            "response_text": response_text, "error": error,
        }
        self.write(relative, artifact)
        record = {
            "stage": stage, "ticket_id": ticket_id, "timestamp": timestamp,
            "provider": "openrouter", "model": request["model"],
            "resolved_model": resolved_model, "upstream_provider": upstream_provider,
            "prompt_hash": stable_hash({"messages": request["messages"], "response_format": request["response_format"]}),
            "output_artifact": relative, "status": "error" if error else "success",
        }
        with (self.root / "llm_calls.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return relative


class OpenRouterClient:
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key, store, *, client=None, sleep=time.sleep):
        if not isinstance(api_key, str) or not api_key.strip():
            raise DataError("OPENROUTER_API_KEY is blank. Set it in .env or your environment before running the pipeline.")
        self.api_key = api_key.strip()
        self.store = store
        self.client = client if client is not None else httpx.Client(timeout=60.0)
        self.owns_client = client is None
        self.sleep = sleep

    def close(self):
        if self.owns_client:
            self.client.close()

    def _sanitize(self, value):
        return value.replace(self.api_key, "[REDACTED]") if isinstance(value, str) else value

    def complete(self, request, ticket_id, stage):
        for attempt in range(2):
            timestamp = utc_now()
            status_code = None
            response_text = None
            error = None
            retryable = False
            resolved_model = None
            upstream_provider = None
            content = None
            try:
                response = self.client.post(self.ENDPOINT, json=request, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=60.0)
                status_code = response.status_code
                response_text = self._sanitize(response.text)
                if not 200 <= status_code < 300:
                    error = f"OpenRouter HTTP {status_code}."
                    retryable = status_code == 429 or status_code >= 500
                else:
                    try:
                        body = loads_json(response_text)
                        if not isinstance(body, dict) or body.get("error"):
                            raise ValueError("Provider returned an error or an invalid response envelope.")
                        resolved_model = body.get("model")
                        upstream_provider = body.get("provider")
                        choice = body["choices"][0]
                        if choice.get("finish_reason") != "stop":
                            raise ValueError("Model did not finish normally (truncation, refusal, or missing finish reason).")
                        message = choice["message"]
                        content = message.get("content")
                        if message.get("refusal") or not isinstance(content, str) or not content.strip():
                            raise ValueError("Provider returned no usable text or refused the request.")
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                        error = "OpenRouter returned an invalid, refused, or truncated completion envelope."
            except httpx.TransportError as exc:
                error = f"OpenRouter transport failure ({type(exc).__name__})."
                retryable = True

            relative = self.store.save_call(
                ticket_id=ticket_id, stage=stage, timestamp=timestamp,
                request=request, response_text=response_text, status_code=status_code,
                error=error, resolved_model=resolved_model, upstream_provider=upstream_provider,
            )
            if error:
                self.store.error(ticket_id, stage, "provider_error", error, output_artifact=relative, attempt=attempt + 1)
                if retryable and attempt == 0:
                    self.sleep(1)
                    continue
                raise ProviderError(error, relative)
            return Completion(content, relative)
        raise AssertionError("Unreachable retry state.")
