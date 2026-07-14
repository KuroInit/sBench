import json

from sbench.datasets import load_azure_chat


def test_azure_loader_accepts_string_token_ids(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"new_input_ids": "[1, 2, 3]", "max_tokens": 4}) + "\n")
    monkeypatch.setenv("S_MFU_AZURE_CHAT_PATH", str(path))
    requests = load_azure_chat({}, limit=1)
    assert requests[0].input_ids == [1, 2, 3]
    assert requests[0].output_len == 4
