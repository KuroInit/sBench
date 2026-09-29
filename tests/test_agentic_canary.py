import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from scripts_server.agentic_canary import build_canary_config, check_server


class _CanaryHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            payload = b"ok"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/v1/models":
            payload = json.dumps({"data": [{"id": "qwen3_5_9b"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def log_message(self, *_):
        pass


class CheckServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _CanaryHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.api_base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_requires_health_and_requested_model_to_be_available(self):
        models = check_server(self.api_base, "qwen3_5_9b", timeout=2)
        self.assertEqual(models, ["qwen3_5_9b"])

    def test_rejects_unserved_model_name(self):
        with self.assertRaisesRegex(RuntimeError, "not served"):
            check_server(self.api_base, "wrong-model", timeout=2)


class BuildCanaryConfigTests(unittest.TestCase):
    def test_selects_exactly_one_issue_and_enforces_singularity_tmpdir(self):
        source = {
            "environment_class": "singularity",
            "issue_count": 8,
            "workers": 4,
            "instance_ids": ["old__issue"],
            "mini_model_name": "openai/old-model",
        }

        config = build_canary_config(source, "sqlfluff__sqlfluff-2419", "qwen3_5_9b")

        self.assertEqual(config["instance_ids"], ["sqlfluff__sqlfluff-2419"])
        self.assertEqual(config["issue_count"], 1)
        self.assertEqual(config["workers"], 1)
        self.assertEqual(config["sandbox_tmpdir"], "/tmp")
        self.assertEqual(config["mini_model_name"], "openai/qwen3_5_9b")
        self.assertEqual(source["instance_ids"], ["old__issue"])

    def test_applies_served_model_override_before_constraining_canary(self):
        source = {
            "environment_class": "singularity",
            "mini_swe_configs": ["swebench.yaml", "swebench_xml"],
            "model_overrides": {
                "qwen3_5_9b": {
                    "mini_swe_configs": [
                        "swebench.yaml",
                        "swebench_xml",
                        "configs/mini_swe_agent_qwen_nonthinking.yaml",
                    ]
                }
            },
        }

        config = build_canary_config(source, "sqlfluff__sqlfluff-2419", "qwen3_5_9b")

        self.assertEqual(
            config["mini_swe_configs"],
            ["swebench.yaml", "swebench_xml", "configs/mini_swe_agent_qwen_nonthinking.yaml"],
        )
        self.assertNotIn("model_overrides", config)

    def test_preserves_explicit_sandbox_directory_and_docker_class(self):
        source = {"environment_class": "docker", "sandbox_tmpdir": "/scratch/sandbox"}

        config = build_canary_config(source, "repo__issue-1", "served-model")

        self.assertEqual(config["environment_class"], "docker")
        self.assertEqual(config["sandbox_tmpdir"], "/scratch/sandbox")
        self.assertEqual(config["instance_ids"], ["repo__issue-1"])
        self.assertEqual(config["workers"], 1)


if __name__ == "__main__":
    unittest.main()
