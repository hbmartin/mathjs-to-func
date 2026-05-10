"""Command-line helpers for mathjs-to-func."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from . import ExpressionError, build_evaluator


def _load_json(path: str) -> dict[str, Any]:
    """Load a JSON object from a file path or stdin."""
    source = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    data = json.loads(source)
    if not isinstance(data, dict):
        raise TypeError("Payload JSON must be an object")
    return data


def _write_json(data: dict[str, Any], output: Path | None) -> None:
    """Write formatted JSON to stdout or a file."""
    content = json.dumps(data, indent=2, sort_keys=True)
    if output is None:
        sys.stdout.write(f"{content}\n")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{content}\n", encoding="utf-8")


def _compile(args: argparse.Namespace) -> int:
    """Compile a payload and emit source or metadata."""
    payload = _load_json(args.payload)
    target = args.target if args.target is not None else payload.get("target")
    evaluator = build_evaluator(
        expressions=payload.get("expressions"),
        inputs=payload.get("inputs"),
        target=target,
        include_source=True,
    )
    if args.emit_source:
        sys.stdout.write(f"{evaluator.__mathjs_source__}\n")
        return 0
    _write_json(
        {
            "evaluation_order": list(evaluator.__mathjs_evaluation_order__),
            "required_inputs": list(evaluator.__mathjs_required_inputs__),
            "target": target,
        },
        output=None,
    )
    return 0


def _schema(args: argparse.Namespace) -> int:
    """Export a JSON Schema for expression or payload validation."""
    try:
        parse_module = importlib.import_module("mathjs_to_func.parse")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Schema export requires the optional parse extra: "
            "install mathjs-to-func[parse]",
        ) from exc

    schema_name = (
        "expression_json_schema" if args.kind == "expression" else "payload_json_schema"
    )
    schema_func = getattr(parse_module, schema_name)
    schema = schema_func()
    schema.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    _write_json(schema, args.output)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(prog="python -m mathjs_to_func")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile",
        help="compile a math.js evaluator payload",
    )
    compile_parser.add_argument("payload", help="payload JSON path, or '-' for stdin")
    compile_parser.add_argument(
        "--target",
        help="override the payload target expression",
    )
    compile_parser.add_argument(
        "--emit-source",
        action="store_true",
        help="print generated Python source instead of metadata JSON",
    )
    compile_parser.set_defaults(func=_compile)

    schema_parser = subparsers.add_parser(
        "schema",
        help="export JSON Schema for supported math.js payloads",
    )
    schema_parser.add_argument(
        "--kind",
        choices=("payload", "expression"),
        default="payload",
        help="schema shape to export",
    )
    schema_parser.add_argument(
        "--output",
        type=Path,
        help="write schema JSON to this path instead of stdout",
    )
    schema_parser.set_defaults(func=_schema)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ExpressionError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
