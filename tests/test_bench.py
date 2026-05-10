import importlib
import subprocess
import sys
from pathlib import Path
from typing import Never

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
bench_main = importlib.import_module("bench.__main__")
PayloadCase = importlib.import_module("bench.payloads").PayloadCase


def test_benchmark_repeats_must_be_positive(capsys):
    with pytest.raises(SystemExit):
        bench_main.main(["--repeats", "0"])

    assert "must be >= 1" in capsys.readouterr().err


def test_mathjs_benchmark_timeout_has_clear_error(monkeypatch):
    case = PayloadCase(
        name="example",
        payload={"expressions": {}, "inputs": [], "target": "res"},
        scope={},
        python_expression="1",
        mathjs_expression="1",
    )

    monkeypatch.setattr(bench_main.shutil, "which", lambda _: "/usr/bin/node")

    def run_timeout(command: list[str], **kwargs: object) -> Never:
        assert kwargs["timeout"] == bench_main.MATHJS_BENCH_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(bench_main.subprocess, "run", run_timeout)

    with pytest.raises(RuntimeError) as excinfo:
        bench_main._benchmark_mathjs([case], repeats=1)  # noqa: SLF001

    assert "math.js benchmark command timed out after 120s" in str(excinfo.value)
    assert "/usr/bin/node" in str(excinfo.value)
    assert "mathjs_bench.mjs" in str(excinfo.value)
