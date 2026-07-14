"""SGLang server entrypoint with sBench probe installed."""

from __future__ import annotations

import os

from .sglang_probe import install_probe


def main() -> None:
    install_probe(
        output_path=os.environ.get("SBENCH_PROBE_RECORD_PATH"),
        profiling_only=os.environ.get("SBENCH_PROFILING_ONLY", "0").lower() in {"1", "true", "yes"},
    )
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    import os as _os
    import sys

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(_os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
