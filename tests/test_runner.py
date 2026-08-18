import json
from io import BytesIO
from urllib.error import HTTPError

from sbench.datasets import BenchmarkRequest
from sbench import runner
from sbench.runner import _format_request_error, _request_endpoint


def test_input_ids_use_sglang_generate_endpoint():
    req = BenchmarkRequest(input_ids=[1, 2, 3], output_len=4)
    assert _request_endpoint(req, use_chat_api=True) == "/generate"
    assert _request_endpoint(req, use_chat_api=False) == "/generate"


def test_chat_messages_still_use_openai_chat_endpoint():
    req = BenchmarkRequest(messages=[{"role": "user", "content": "hello"}], output_len=4)
    assert _request_endpoint(req, use_chat_api=True) == "/v1/chat/completions"


def test_http_error_includes_response_body():
    err = HTTPError(
        url="http://127.0.0.1:30000/v1/completions",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=BytesIO(b'{"detail":"input_ids is not supported"}'),
    )
    message = _format_request_error(err)
    assert "HTTP Error 400" in message
    assert "input_ids is not supported" in message


def test_input_id_request_payload_uses_native_sglang_generate(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"meta_info": {"completion_tokens": 4}}).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode())
        return Response()

    monkeypatch.setattr(runner.urlrequest, "urlopen", fake_urlopen)

    result = runner._send_request("http://127.0.0.1:30000", "Qwen/Test", BenchmarkRequest(input_ids=[1, 2, 3], output_len=4), False)

    assert result.success
    assert captured["url"] == "http://127.0.0.1:30000/generate"
    assert captured["payload"]["input_ids"] == [1, 2, 3]
    assert captured["payload"]["sampling_params"] == {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True}
