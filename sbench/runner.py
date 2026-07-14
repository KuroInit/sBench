"""OpenAI-compatible async load runner."""

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
    endpoint = "/v1/chat/completions" if use_chat_api and req.messages else "/v1/completions"
    url = api_base.rstrip("/") + endpoint
    if endpoint.endswith("chat/completions"):
        payload: dict[str, Any] = {"model": model, "messages": req.messages, "temperature": 0, "max_tokens": req.output_len}
    else:
        payload = {"model": model, "temperature": 0, "max_tokens": req.output_len}
        if req.input_ids is not None:
            payload["input_ids"] = req.input_ids
        else:
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
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        out_len = int(usage.get("completion_tokens") or req.output_len)
        return RequestResult(uid=req.uid, success=True, latency=latency, output_len=out_len)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return RequestResult(uid=req.uid, success=False, latency=time.perf_counter() - start, output_len=0, error=str(exc))


def write_request_results(path: str, results: list[RequestResult]) -> None:
    from pathlib import Path
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result)) + "\n")
