"""Dataset loaders for prefill, chat, reasoning, and agentic benchmark lanes."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SHAREGPT_DATASET = "json"
SHAREGPT_DATA_FILES = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"


@dataclass(frozen=True)
class BenchmarkRequest:
    prompt: str | None = None
    messages: list[dict[str, str]] | None = None
    input_ids: list[int] | None = None
    output_len: int = 1
    uid: str | None = None


def load_dataset(dataset: str, config: dict[str, Any], limit: int | None = None) -> list[BenchmarkRequest]:
    if dataset == "batched_prefill":
        return load_batched_prefill(config, limit)
    if dataset == "sharegpt":
        return load_sharegpt(config, limit)
    if dataset == "azure_chat":
        return load_azure_chat(config, limit)
    if dataset == "swe_bench":
        return load_swe_bench(config, limit)
    if dataset == "mmlu_pro":
        return load_mmlu_pro(config, limit)
    raise ValueError(f"unsupported dataset: {dataset}")


def load_batched_prefill(config: dict[str, Any], limit: int | None = None) -> list[BenchmarkRequest]:
    count = limit or int(config.get("num_samples", 16))
    prompt_tokens = int(config.get("target_input_tokens", 2048))
    output_len = int(config.get("target_output_tokens", 1))
    token_id = int(config.get("synthetic_token_id", 100))
    input_ids = [token_id] * max(prompt_tokens, 1)
    return [BenchmarkRequest(input_ids=input_ids, output_len=output_len, uid=f"prefill-{i}") for i in range(count)]


def load_sharegpt(config: dict[str, Any], limit: int | None = None) -> list[BenchmarkRequest]:
    rows = _load_rows_for_dataset(
        path=os.environ.get("S_MFU_SHAREGPT_PATH") or config.get("path"),
        hf_dataset=os.environ.get("S_MFU_SHAREGPT_HF_DATASET") or config.get("hf_dataset") or SHAREGPT_DATASET,
        hf_config=os.environ.get("S_MFU_SHAREGPT_HF_CONFIG") or config.get("hf_config"),
        hf_data_files=os.environ.get("S_MFU_SHAREGPT_HF_DATA_FILES") or config.get("hf_data_files") or SHAREGPT_DATA_FILES,
        split=config.get("dataset_split", "train"),
        dataset_label="sharegpt",
    )
    requests = _chat_rows_to_requests(rows, limit, default_output_len=int(config.get("target_output_tokens", 128)))
    return _apply_chat_token_limit(requests, config)


def load_azure_chat(config: dict[str, Any], limit: int | None = None) -> list[BenchmarkRequest]:
    rows = _load_rows_for_dataset(
        path=os.environ.get("S_MFU_AZURE_CHAT_PATH") or config.get("path"),
        hf_dataset=os.environ.get("S_MFU_AZURE_CHAT_HF_DATASET") or config.get("hf_dataset"),
        hf_config=os.environ.get("S_MFU_AZURE_CHAT_HF_CONFIG") or config.get("hf_config"),
        hf_data_files=os.environ.get("S_MFU_AZURE_CHAT_HF_DATA_FILES") or config.get("hf_data_files"),
        split=config.get("dataset_split", "train"),
        dataset_label="azure_chat",
    )
    requests = []
    for idx, row in enumerate(rows):
        if limit is not None and len(requests) >= limit:
            break
        if "ContextTokens" in row or "GeneratedTokens" in row:
            ctx = int(row.get("ContextTokens", 1))
            out = int(row.get("GeneratedTokens", config.get("target_output_tokens", 128)))
            requests.append(BenchmarkRequest(input_ids=[1] * ctx, output_len=out, uid=f"azure-{idx}"))
        elif row.get("new_input_ids"):
            ids = _token_ids_from_value(row["new_input_ids"])
            out = int(row.get("max_tokens") or row.get("output_len") or config.get("target_output_tokens", 128))
            requests.append(BenchmarkRequest(input_ids=ids, output_len=out, uid=f"azure-{idx}"))
        else:
            requests.extend(_chat_rows_to_requests([row], 1, int(config.get("target_output_tokens", 128))))
    return requests


def load_swe_bench(config: dict[str, Any], limit: int | None = None) -> list[BenchmarkRequest]:
    rows = _load_rows_for_dataset(
        path=os.environ.get("S_MFU_SWEBENCH_PATH") or config.get("path"),
        hf_dataset=os.environ.get("S_MFU_SWEBENCH_HF_DATASET") or config.get("hf_dataset") or SWEBENCH_DATASET,
        hf_config=os.environ.get("S_MFU_SWEBENCH_HF_CONFIG") or config.get("hf_config"),
        hf_data_files=os.environ.get("S_MFU_SWEBENCH_HF_DATA_FILES") or config.get("hf_data_files"),
        split=config.get("dataset_split", "test"),
        dataset_label="swe_bench",
    )
    requests = []
    for idx, row in enumerate(rows):
        if limit is not None and len(requests) >= limit:
            break
        prompt = _swe_prompt(row)
        if prompt:
            requests.append(BenchmarkRequest(prompt=prompt, output_len=int(config.get("target_output_tokens", 512)), uid=str(row.get("instance_id", idx))))
    return requests


def load_mmlu_pro(config: dict[str, Any], limit: int | None = None) -> list[BenchmarkRequest]:
    rows = _load_rows_for_dataset(
        path=os.environ.get("S_MFU_MMLU_PRO_PATH") or config.get("path"),
        hf_dataset=os.environ.get("S_MFU_MMLU_PRO_HF_DATASET") or config.get("hf_dataset") or MMLU_PRO_DATASET,
        hf_config=os.environ.get("S_MFU_MMLU_PRO_HF_CONFIG") or config.get("hf_config"),
        hf_data_files=os.environ.get("S_MFU_MMLU_PRO_HF_DATA_FILES") or config.get("hf_data_files"),
        split=config.get("dataset_split", "test"),
        dataset_label="mmlu_pro",
    )
    requests = []
    for idx, row in enumerate(rows):
        if limit is not None and len(requests) >= limit:
            break
        question = row.get("question") or row.get("input") or row.get("prompt")
        if not question:
            continue
        choices = row.get("options") or row.get("choices") or []
        if isinstance(choices, str):
            try:
                parsed = json.loads(choices)
                choices = parsed if isinstance(parsed, list) else choices
            except json.JSONDecodeError:
                choices = [item.strip() for item in choices.split("\n") if item.strip()]
        prompt = str(question)
        if isinstance(choices, list):
            prompt += "\n" + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
        if str(config.get("answer_mode", "direct")).lower() == "reasoning":
            prompt += (
                "\n\nSolve the problem carefully. Explain your reasoning briefly, then end with "
                "'Answer: <letter>' where <letter> is the selected option."
            )
        requests.append(BenchmarkRequest(prompt=prompt, output_len=int(config.get("target_output_tokens", 16)), uid=f"mmlu-{idx}"))
    return _apply_prompt_token_limit(requests, config)


def _apply_chat_token_limit(requests: list[BenchmarkRequest], config: dict[str, Any]) -> list[BenchmarkRequest]:
    """Apply an exact chat-context cap using the selected model's tokenizer.

    SGLang's chat endpoint accepts messages, but it cannot truncate them at a
    token boundary. Converting capped conversations to input_ids makes the
    configured request length deterministic and prevents one unusually long
    ShareGPT conversation from invalidating an entire sweep.
    """

    limit = config.get("max_input_tokens")
    if limit is None:
        return requests
    max_tokens = int(limit)
    if max_tokens <= 0:
        raise ValueError("max_input_tokens must be positive when configured")
    model_id = config.get("model_id")
    if not model_id:
        raise ValueError("max_input_tokens requires model_id for tokenizer-based chat truncation")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_id), trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load tokenizer for ShareGPT context limit: {exc}") from exc

    capped: list[BenchmarkRequest] = []
    for request in requests:
        if not request.messages:
            capped.append(request)
            continue
        messages = _trim_messages_to_token_limit(tokenizer, request.messages, max_tokens)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        capped.append(
            BenchmarkRequest(
                input_ids=[int(token) for token in token_ids[-max_tokens:]],
                output_len=request.output_len,
                uid=request.uid,
            )
        )
    return capped


def _trim_messages_to_token_limit(tokenizer: Any, messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """Discard oldest turns first, retaining the newest user request."""

    trimmed = list(messages)
    while len(trimmed) > 1:
        token_ids = tokenizer.apply_chat_template(trimmed, tokenize=True, add_generation_prompt=True)
        if len(token_ids) <= max_tokens:
            return trimmed
        trimmed = trimmed[1:]
    return trimmed


def _apply_prompt_token_limit(requests: list[BenchmarkRequest], config: dict[str, Any]) -> list[BenchmarkRequest]:
    """Apply an exact input cap to completion-style prompts when requested."""

    limit = config.get("max_input_tokens")
    if limit is None:
        return requests
    max_tokens = int(limit)
    if max_tokens <= 0:
        raise ValueError("max_input_tokens must be positive when configured")
    model_id = config.get("model_id")
    if not model_id:
        raise ValueError("max_input_tokens requires model_id for tokenizer-based prompt truncation")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_id), trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load tokenizer for prompt context limit: {exc}") from exc

    capped: list[BenchmarkRequest] = []
    for request in requests:
        if request.prompt is None:
            capped.append(request)
            continue
        token_ids = tokenizer(request.prompt, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_tokens:
            raise ValueError(
                f"prompt for {request.uid or 'request'} has {len(token_ids)} tokens, "
                f"exceeding max_input_tokens={max_tokens}; refusing to truncate the question"
            )
        capped.append(
            BenchmarkRequest(
                input_ids=[int(token) for token in token_ids],
                output_len=request.output_len,
                uid=request.uid,
            )
        )
    return capped


def _load_rows_for_dataset(
    *,
    path: str | None,
    hf_dataset: str | None,
    hf_config: str | None,
    hf_data_files: str | list[str] | dict[str, str] | None,
    split: str,
    dataset_label: str,
) -> Iterable[dict[str, Any]]:
    if path:
        return _read_rows(path)
    if hf_dataset:
        return _load_hf_rows(hf_dataset, split=split, dataset_config=hf_config, data_files=hf_data_files)
    raise ValueError(f"{dataset_label} requires a local path or Hugging Face dataset configuration")


def _load_hf_rows(
    dataset_name: str,
    *,
    split: str,
    dataset_config: str | None = None,
    data_files: str | list[str] | dict[str, str] | None = None,
) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Hugging Face dataset loading; install datasets or provide a local path") from exc
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if data_files is not None:
        kwargs["data_files"] = data_files
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def _read_rows(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    data = json.loads(source.read_text())
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return [row for row in data["data"] if isinstance(row, dict)]
        return [data]
    return []


def _token_ids_from_value(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(token) for token in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(token) for token in parsed]
        return [int(token) for token in text.replace(",", " ").split()]
    return []


def _chat_rows_to_requests(rows: Iterable[dict[str, Any]], limit: int | None, default_output_len: int) -> list[BenchmarkRequest]:
    requests = []
    for idx, row in enumerate(rows):
        if limit is not None and len(requests) >= limit:
            break
        messages = _extract_messages(row)
        if messages:
            requests.append(BenchmarkRequest(messages=messages, output_len=int(row.get("output_len") or row.get("max_tokens") or default_output_len), uid=str(row.get("id", idx))))
            continue
        prompt = row.get("prompt") or row.get("instruction") or row.get("input")
        if prompt:
            requests.append(BenchmarkRequest(prompt=str(prompt), output_len=int(row.get("output_len") or row.get("max_tokens") or default_output_len), uid=str(row.get("id", idx))))
    return requests


def _extract_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("messages") or row.get("conversations")
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        if not content:
            continue
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant"}:
            role = "assistant"
        else:
            role = "user"
        out.append({"role": role, "content": str(content)})
    while out and out[-1]["role"] == "assistant":
        out.pop()
    return out


def _swe_prompt(row: dict[str, Any]) -> str:
    problem = row.get("problem_statement") or row.get("problem") or row.get("issue")
    repo = row.get("repo", "")
    base = row.get("base_commit", "")
    if not problem:
        return ""
    return f"Repository: {repo}\nBase commit: {base}\nProblem:\n{problem}\n\nGenerate a patch that fixes the issue."
