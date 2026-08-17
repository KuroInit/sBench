import json

import pytest

from sbench.datasets import load_azure_chat, load_batched_prefill, load_mmlu_pro, load_sharegpt


def test_batched_prefill_uses_exact_synthetic_token_count():
    requests = load_batched_prefill({"target_input_tokens": 17, "synthetic_token_id": 42, "num_samples": 2})
    assert len(requests) == 2
    assert requests[0].input_ids == [42] * 17
    assert requests[0].prompt is None


def test_azure_loader_accepts_string_token_ids(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"new_input_ids": "[1, 2, 3]", "max_tokens": 4}) + "\n")
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))
    requests = load_azure_chat({}, limit=1)
    assert requests[0].input_ids == [1, 2, 3]
    assert requests[0].output_len == 4


def test_mmlu_reasoning_mode_requests_explanation(tmp_path, monkeypatch):
    path = tmp_path / "mmlu.jsonl"
    path.write_text(json.dumps({"question": "2 + 2?", "options": ["3", "4"]}) + "\n")
    monkeypatch.setenv("S_MFU_MMLU_PRO_PATH", str(path))
    request = load_mmlu_pro({"answer_mode": "reasoning", "target_output_tokens": 256}, limit=1)[0]
    assert "Explain your reasoning" in request.prompt
    assert request.output_len == 256


def test_mmlu_context_cap_rejects_semantically_destructive_truncation(tmp_path, monkeypatch):
    class Tokenizer:
        def __call__(self, _prompt, **_kwargs):
            return {"input_ids": [10, 11, 12, 13]}

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    path = tmp_path / "mmlu.jsonl"
    path.write_text(json.dumps({"question": "2 + 2?", "options": ["3", "4"]}) + "\n")
    monkeypatch.setenv("S_MFU_MMLU_PRO_PATH", str(path))
    with pytest.raises(ValueError, match="refusing to truncate the question"):
        load_mmlu_pro({"model_id": "Qwen/Test", "max_input_tokens": 2}, limit=1)


def test_sharegpt_context_cap_uses_model_chat_template(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            return list(range(sum(len(item["content"].split()) for item in messages) + 1))

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([{"conversations": [{"from": "human", "value": "old turn words"}, {"from": "gpt", "value": "answer"}, {"from": "human", "value": "latest question words"}]}]))
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))
    request = load_sharegpt({"model_id": "Qwen/Test", "max_input_tokens": 3}, limit=1)[0]
    assert request.messages is None
    assert request.input_ids == [1, 2, 3]
