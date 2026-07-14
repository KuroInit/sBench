"""Optional process-wide sBench probe bootstrap.

SGLang 0.5.x starts model workers in child Python processes. Installing the
probe only in the HTTP entrypoint process is not enough, because forward passes
happen in those workers. Python imports ``sitecustomize`` during interpreter
startup when it is importable on ``sys.path``; this guarded hook lets the
orchestrator opt child processes into the same lightweight probe.
"""

from __future__ import annotations

import os


if os.environ.get("SBENCH_AUTO_INSTALL_PROBE", "0").lower() in {"1", "true", "yes"}:
    try:
        from sbench_probe.sglang_probe import install_probe

        install_probe(
            output_path=os.environ.get("SBENCH_PROBE_RECORD_PATH"),
            profiling_only=os.environ.get("SBENCH_PROFILING_ONLY", "0").lower() in {"1", "true", "yes"},
        )
    except Exception:
        pass
