"""Async load runner for SGLang and OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from .datasets import BenchmarkRequest


@dataclass(frozen=True)
class RequestResult:
    uid: str | None
    success: bool
    latency: float
    output_len: int
    error: str = ""


async def run_requests(api_base: str, model: str, requests: list[BenchmarkRequest], concurrency: int, use_chat_api: bool = False) -> list[RequestResult]:
    sem = asyncio.Semaphore(max(concurrency, 1))

    async def one(req: BenchmarkRequest) -> RequestResult:
        async with sem:
            return await asyncio.to_thread(_send_request, api_base, model, req, use_chat_api)

    return await asyncio.gather(*(one(req) for req in requests))


def _send_request(api_base: str, model: str, req: BenchmarkRequest, use_chat_api: bool) -> RequestResult:
    endpoint = _request_endpoint(req, use_chat_api)
    url = api_base.rstrip("/") + endpoint
    if endpoint == "/generate":
        payload: dict[str, Any] = {
            "input_ids": req.input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": req.output_len, "ignore_eos": True},
        }
    elif endpoint.endswith("chat/completions"):
        payload: dict[str, Any] = {"model": model, "messages": req.messages, "temperature": 0, "max_tokens": req.output_len}
    else:
        payload = {"model": model, "temperature": 0, "max_tokens": req.output_len}
        payload["prompt"] = req.prompt or ""
    body = json.dumps(payload).encode()
    start = time.perf_counter()
    try:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        http_req = urlrequest.Request(url, data=body, headers=headers, method="POST")
        with urlrequest.urlopen(http_req, timeout=3600) as resp:
            data = json.loads(resp.read().decode() or "{}")
        latency = time.perf_counter() - start
        out_len = _completion_tokens(data, req.output_len)
        return RequestResult(uid=req.uid, success=True, latency=latency, output_len=out_len)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return RequestResult(uid=req.uid, success=False, latency=time.perf_counter() - start, output_len=0, error=_format_request_error(exc))


def _request_endpoint(req: BenchmarkRequest, use_chat_api: bool) -> str:
    if req.input_ids is not None:
        return "/generate"
    if use_chat_api and req.messages:
        return "/v1/chat/completions"
    return "/v1/completions"


def _completion_tokens(data: Any, fallback: int) -> int:
    if not isinstance(data, dict):
        return int(fallback)
    usage = data.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens") or usage.get("output_tokens")
        if value is not None:
            return int(value)
    meta = data.get("meta_info")
    if isinstance(meta, dict):
        value = meta.get("completion_tokens") or meta.get("output_tokens") or meta.get("completion_token_logprobs")
        if isinstance(value, list):
            return len(value)
        if value is not None:
            return int(value)
    output_ids = data.get("output_ids")
    if isinstance(output_ids, list):
        return len(output_ids)
    return int(fallback)


def _format_request_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if body:
            return f"HTTP Error {exc.code}: {exc.reason}; body={body[:1000]}"
    return str(exc)


def write_request_results(path: str, results: list[RequestResult]) -> None:
    from pathlib import Path
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result)) + "\n")
