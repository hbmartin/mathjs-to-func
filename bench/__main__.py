"""Run mathjs-to-func benchmarks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING

import numpy as np
from simpleeval import SimpleEval

from mathjs_to_func import build_evaluator

from .payloads import PayloadCase, payload_cases

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
MATHJS_BENCH_TIMEOUT_SECONDS = 120


def _sum(*values: object) -> object:
    result: object = 0
    for value in values:
        result = result + value  # type: ignore[operator]
    return result


SAFE_FUNCTIONS: dict[str, Callable[..., object]] = {
    "max": max,
    "min": min,
    "sqrt": np.sqrt,
    "sum": _sum,
}
SAFE_GLOBALS: dict[str, object] = {"__builtins__": {}, **SAFE_FUNCTIONS}


@dataclass(frozen=True)
class BenchmarkResult:
    """Single benchmark measurement."""

    case: str
    runner: str
    phase: str
    seconds_per_op: float


@dataclass(frozen=True)
class PerfCheck:
    """Relative performance check for the CI gate."""

    case: str
    baseline_runner: str
    current_phase: str
    baseline_phase: str
    ratio: float
    max_ratio: float
    passed: bool


def _measure(iterations: int, repeats: int, callback: Callable[[], object]) -> float:
    timings: list[float] = []
    result: object = None
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            result = callback()
        timings.append((time.perf_counter() - start) / iterations)
    if isinstance(result, np.ndarray) and result.size == -1:
        raise AssertionError("unreachable")
    return median(timings)


def _benchmark_mathjs_to_func(case: PayloadCase, repeats: int) -> list[BenchmarkResult]:
    build_seconds = _measure(
        case.build_iterations,
        repeats,
        lambda: build_evaluator(payload=case.payload),
    )
    evaluator = build_evaluator(payload=case.payload)
    call_seconds = _measure(
        case.iterations,
        repeats,
        lambda: evaluator(case.scope),
    )
    return [
        BenchmarkResult(
            case=case.name,
            runner="mathjs_to_func",
            phase="build",
            seconds_per_op=build_seconds,
        ),
        BenchmarkResult(
            case=case.name,
            runner="mathjs_to_func",
            phase="reusable_call",
            seconds_per_op=call_seconds,
        ),
    ]


def _benchmark_python_eval(case: PayloadCase, repeats: int) -> list[BenchmarkResult]:
    build_seconds = _measure(
        case.build_iterations,
        repeats,
        lambda: compile(case.python_expression, f"<bench:{case.name}>", "eval"),
    )
    code = compile(case.python_expression, f"<bench:{case.name}>", "eval")
    call_seconds = _measure(
        case.iterations,
        repeats,
        lambda: eval(code, SAFE_GLOBALS, case.scope),  # noqa: S307
    )
    return [
        BenchmarkResult(
            case=case.name,
            runner="python_eval",
            phase="build",
            seconds_per_op=build_seconds,
        ),
        BenchmarkResult(
            case=case.name,
            runner="python_eval",
            phase="reusable_call",
            seconds_per_op=call_seconds,
        ),
    ]


def _benchmark_simpleeval(case: PayloadCase, repeats: int) -> list[BenchmarkResult]:
    if not case.supports_simpleeval:
        return []
    build_seconds = _measure(
        case.build_iterations,
        repeats,
        lambda: SimpleEval(functions=SAFE_FUNCTIONS, names=case.scope),
    )
    evaluator = SimpleEval(functions=SAFE_FUNCTIONS, names=case.scope)
    call_seconds = _measure(
        case.iterations,
        repeats,
        lambda: evaluator.eval(case.python_expression),
    )
    return [
        BenchmarkResult(
            case=case.name,
            runner="simpleeval",
            phase="build",
            seconds_per_op=build_seconds,
        ),
        BenchmarkResult(
            case=case.name,
            runner="simpleeval",
            phase="reusable_call",
            seconds_per_op=call_seconds,
        ),
    ]


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _benchmark_mathjs(
    cases: Sequence[PayloadCase],
    repeats: int,
) -> list[BenchmarkResult]:
    script = ROOT / "bench" / "js" / "mathjs_bench.mjs"
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required for the math.js benchmark")
    input_payload = {
        "cases": [
            {
                "buildIterations": case.build_iterations,
                "iterations": case.iterations,
                "mathjsExpression": case.mathjs_expression,
                "name": case.name,
                "scope": _jsonable(case.scope),
            }
            for case in cases
        ],
        "repeats": repeats,
    }
    command = [node, str(script)]
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            input=json.dumps(input_payload),
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=MATHJS_BENCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "math.js benchmark command timed out after "
            f"{MATHJS_BENCH_TIMEOUT_SECONDS}s: {' '.join(command)}",
        ) from exc
    return [
        BenchmarkResult(
            case=item["case"],
            runner=item["runner"],
            phase=item["phase"],
            seconds_per_op=item["seconds_per_op"],
        )
        for item in json.loads(completed.stdout)
    ]


def _run_benchmarks(repeats: int) -> list[BenchmarkResult]:
    cases = payload_cases()
    results: list[BenchmarkResult] = []
    for case in cases:
        results.extend(_benchmark_mathjs_to_func(case, repeats))
        results.extend(_benchmark_python_eval(case, repeats))
        results.extend(_benchmark_simpleeval(case, repeats))
    results.extend(_benchmark_mathjs(cases, repeats))
    return results


def _format_results(results: Sequence[BenchmarkResult]) -> str:
    lines = [
        f"{'case':<22} {'runner':<16} {'phase':<18} {'us/op':>12}",
        "-" * 72,
    ]
    sorted_results = sorted(
        results,
        key=lambda item: (item.case, item.phase, item.runner),
    )
    lines.extend(
        (
            f"{result.case:<22} {result.runner:<16} "
            f"{result.phase:<18} {result.seconds_per_op * 1_000_000:>12.3f}"
        )
        for result in sorted_results
    )
    return "\n".join(lines)


def _perf_checks(results: Sequence[BenchmarkResult]) -> list[PerfCheck]:
    indexed = {
        (result.case, result.runner, result.phase): result.seconds_per_op
        for result in results
    }
    checks: list[PerfCheck] = []
    gates = (
        ("python_eval", "reusable_call", 12.0),
        ("simpleeval", "reusable_call", 1.0),
        ("mathjs", "roundtrip_compile", 1.0),
    )
    for case in {result.case for result in results}:
        current = indexed.get((case, "mathjs_to_func", "reusable_call"))
        if current is None:
            continue
        for runner, baseline_phase, max_ratio in gates:
            baseline = indexed.get((case, runner, baseline_phase))
            if baseline is None:
                continue
            ratio = current / baseline
            checks.append(
                PerfCheck(
                    case=case,
                    baseline_runner=runner,
                    current_phase="reusable_call",
                    baseline_phase=baseline_phase,
                    ratio=ratio,
                    max_ratio=max_ratio,
                    passed=ratio <= max_ratio,
                ),
            )
    return checks


def _write_json(
    results: Sequence[BenchmarkResult],
    checks: Sequence[PerfCheck],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checks": [asdict(check) for check in checks],
                "results": [asdict(result) for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bench")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the relative performance gate is not met",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="write machine-readable benchmark results to this path",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=5,
        help="number of timing repeats per benchmark",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run benchmarks and optional performance checks."""
    args = _build_parser().parse_args(argv)
    results = _run_benchmarks(repeats=args.repeats)
    checks = _perf_checks(results)
    sys.stdout.write(f"{_format_results(results)}\n")
    if args.json is not None:
        _write_json(results, checks, args.json)
    failures = [check for check in checks if not check.passed]
    if args.check and failures:
        for failure in failures:
            sys.stderr.write(
                "perf check failed: "
                f"{failure.case} mathjs_to_func/{failure.current_phase} was "
                f"{failure.ratio:.2f}x "
                f"{failure.baseline_runner}/{failure.baseline_phase}; "
                f"max {failure.max_ratio:.2f}x\n",
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
