"""Async load runner for SGLang and OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import re
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
    completion: str | None = None
    parsed_answer: str | None = None
    gold_answer: str | None = None
    correct: bool | None = None
    metadata: dict[str, Any] | None = None


async def run_requests(
    api_base: str,
    model: str,
    requests: list[BenchmarkRequest],
    concurrency: int,
    use_chat_api: bool = False,
    *,
    save_responses: bool = False,
    parse_answers: bool = False,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(max(concurrency, 1))

    async def one(req: BenchmarkRequest) -> RequestResult:
        async with sem:
            return await asyncio.to_thread(
                _send_request,
                api_base,
                model,
                req,
                use_chat_api,
                save_responses=save_responses,
                parse_answers=parse_answers,
            )

    return await asyncio.gather(*(one(req) for req in requests))


def _send_request(
    api_base: str,
    model: str,
    req: BenchmarkRequest,
    use_chat_api: bool,
    *,
    save_responses: bool = False,
    parse_answers: bool = False,
) -> RequestResult:
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
        completion = _completion_text(data) if save_responses or parse_answers else None
        parsed_answer = parse_answer_letter(completion) if parse_answers else None
        gold_answer = _gold_answer(req) if parse_answers else None
        correct = (parsed_answer == gold_answer) if parsed_answer and gold_answer else None
        metadata = req.metadata if save_responses and req.metadata else None
        return RequestResult(
            uid=req.uid,
            success=True,
            latency=latency,
            output_len=out_len,
            completion=completion if save_responses else None,
            parsed_answer=parsed_answer,
            gold_answer=gold_answer,
            correct=correct,
            metadata=metadata,
        )
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


def _completion_text(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message.get("content") or "")
            if first.get("text") is not None:
                return str(first.get("text") or "")
    for key in ("text", "output", "output_text", "generated_text"):
        value = data.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return str(value[0] or "")
    outputs = data.get("outputs")
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("text", "output", "content"):
                if first.get(key) is not None:
                    return str(first.get(key) or "")
    return ""


def parse_answer_letter(text: str | None) -> str | None:
    if not text:
        return None
    patterns = [
        r"(?i)\banswer\s*[:：]\s*\(?\s*([A-Z])\s*\)?",
        r"(?i)\bfinal\s+answer\s*[:：]\s*\(?\s*([A-Z])\s*\)?",
        r"(?i)\bthe\s+answer\s+is\s+\(?\s*([A-Z])\s*\)?",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return str(matches[-1]).upper()
    stripped = text.strip()
    if len(stripped) <= 4:
        match = re.search(r"\b([A-Z])\b", stripped.upper())
        if match:
            return match.group(1)
    return None


def _gold_answer(req: BenchmarkRequest) -> str | None:
    value = req.metadata.get("gold_answer") if req.metadata else None
    return str(value).strip().upper() if value else None


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
            row = {key: value for key, value in asdict(result).items() if value is not None}
            handle.write(json.dumps(row) + "\n")
