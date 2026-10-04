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
    path.write_text(json.dumps({"question": "2 + 2?", "options": ["3", "4"], "answer": "B"}) + "\n")
    monkeypatch.setenv("S_MFU_MMLU_PRO_PATH", str(path))
    request = load_mmlu_pro({"answer_mode": "reasoning", "target_output_tokens": 256}, limit=1)[0]
    assert "Explain your reasoning" in request.prompt
    assert request.output_len == 256
    assert request.metadata["gold_answer"] == "B"
    assert request.metadata["choices"] == ["3", "4"]


def test_mmlu_direct_mode_requests_letter_only_answer(tmp_path, monkeypatch):
    path = tmp_path / "mmlu.jsonl"
    path.write_text(json.dumps({"question": "2 + 2?", "options": ["3", "4"], "answer": "B"}) + "\n")
    monkeypatch.setenv("S_MFU_MMLU_PRO_PATH", str(path))
    request = load_mmlu_pro({"answer_mode": "direct", "target_output_tokens": 16}, limit=1)[0]
    assert "Answer with only the selected option letter." in request.prompt
    assert "Explain your reasoning" not in request.prompt
    assert request.output_len == 16


def test_mmlu_gold_answer_can_be_choice_text(tmp_path, monkeypatch):
    path = tmp_path / "mmlu.jsonl"
    path.write_text(json.dumps({"question": "2 + 2?", "options": ["3", "4"], "answer": "4"}) + "\n")
    monkeypatch.setenv("S_MFU_MMLU_PRO_PATH", str(path))
    request = load_mmlu_pro({"answer_mode": "direct", "target_output_tokens": 16}, limit=1)[0]
    assert request.metadata["gold_answer"] == "B"


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


def test_sharegpt_loader_parses_json_encoded_conversation_turns(tmp_path, monkeypatch):
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([{"id": "sample-1", "conversations": [
        json.dumps({"from": "human", "value": "What is 2 + 2?"}),
        json.dumps({"from": "gpt", "value": "4"}),
        json.dumps({"from": "human", "value": "Show the calculation."}),
    ]}]))
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))

    requests = load_sharegpt({"target_output_tokens": 16}, limit=1)

    assert len(requests) == 1
    assert requests[0].messages == [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "Show the calculation."},
    ]


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


def test_sharegpt_context_cap_accepts_tokenizer_encoding_results(tmp_path, monkeypatch):
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            ids = list(range(sum(len(item["content"].split()) for item in messages) + 1))
            return [Encoding(ids)]

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([{"conversations": [{"from": "human", "value": "old turn words"}, {"from": "gpt", "value": "answer"}, {"from": "human", "value": "latest question words"}]}]))
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))
    request = load_sharegpt({"model_id": "Qwen/Test", "max_input_tokens": 4}, limit=1)[0]
    assert request.input_ids == [0, 1, 2, 3]


def test_sharegpt_context_trim_preserves_user_first_role_order(tmp_path, monkeypatch):
    seen_roles = []

    class Tokenizer:
        def apply_chat_template(self, messages, **_kwargs):
            roles = [item["role"] for item in messages]
            assert roles[0] == "user"
            assert roles[-1] == "user"
            for left, right in zip(roles, roles[1:]):
                assert left != right
            seen_roles.append(roles)
            return list(range(sum(len(item["content"].split()) for item in messages) + 1))

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps([{"conversations": [
        {"from": "human", "value": "old question words"},
        {"from": "gpt", "value": "old answer words"},
        {"from": "human", "value": "latest question words"},
    ]}]))
    monkeypatch.setenv("S_MFU_SHAREGPT_PATH", str(path))
    request = load_sharegpt({"model_id": "Qwen/Test", "max_input_tokens": 4}, limit=1)[0]
    assert request.input_ids == [0, 1, 2, 3]
    assert seen_roles[-1] == ["user"]
