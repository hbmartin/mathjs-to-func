"""Representative benchmark payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PayloadCase:
    """One benchmark payload and equivalent expressions for comparison."""

    name: str
    payload: dict[str, Any]
    scope: dict[str, Any]
    python_expression: str
    mathjs_expression: str | None
    supports_simpleeval: bool = True
    iterations: int = 20_000
    build_iterations: int = 500


def const(value: object) -> dict[str, Any]:
    """Build a math.js ConstantNode."""
    value_type = "boolean" if isinstance(value, bool) else "number"
    return {
        "type": "ConstantNode",
        "value": str(value).lower() if isinstance(value, bool) else str(value),
        "valueType": value_type,
    }


def symbol(name: str) -> dict[str, Any]:
    """Build a math.js SymbolNode."""
    return {"type": "SymbolNode", "name": name}


def op(fn: str, *args: dict[str, Any]) -> dict[str, Any]:
    """Build a math.js OperatorNode."""
    return {"type": "OperatorNode", "fn": fn, "op": fn, "args": list(args)}


def func(name: str, *args: dict[str, Any]) -> dict[str, Any]:
    """Build a math.js FunctionNode."""
    return {"type": "FunctionNode", "fn": name, "args": list(args)}


def conditional(
    condition: dict[str, Any],
    true_expr: dict[str, Any],
    false_expr: dict[str, Any],
) -> dict[str, Any]:
    """Build a math.js ConditionalNode."""
    return {
        "type": "ConditionalNode",
        "condition": condition,
        "trueExpr": true_expr,
        "falseExpr": false_expr,
    }


def payload_cases() -> list[PayloadCase]:
    """Return benchmark cases covering scalar, helper, and vector workloads."""
    scalar_payload = {
        "expressions": {
            "total": op("add", symbol("x"), symbol("y")),
            "adjusted": op(
                "subtract",
                op("multiply", symbol("total"), const(2)),
                symbol("offset"),
            ),
            "z": op("divide", symbol("adjusted"), symbol("scale")),
        },
        "inputs": ["x", "y", "offset", "scale"],
        "target": "z",
    }

    distance = func(
        "sqrt",
        op(
            "add",
            op("pow", symbol("x"), const(2)),
            op("pow", symbol("y"), const(2)),
        ),
    )
    conditional_payload = {
        "expressions": {
            "distance": distance,
            "z": conditional(
                op("larger", symbol("distance"), symbol("limit")),
                symbol("distance"),
                const(0),
            ),
        },
        "inputs": ["x", "y", "limit"],
        "target": "z",
    }

    aggregate_payload = {
        "expressions": {
            "z": op(
                "subtract",
                op(
                    "add",
                    func("sum", symbol("a"), symbol("b"), symbol("c")),
                    func("max", symbol("a"), symbol("b"), symbol("c")),
                ),
                func("min", symbol("a"), symbol("b"), symbol("c")),
            ),
        },
        "inputs": ["a", "b", "c"],
        "target": "z",
    }

    vector_payload = {
        "expressions": {
            "squares": op("pow", symbol("vec"), const(2)),
            "shifted": op("add", symbol("squares"), const(1)),
            "z": func("sqrt", symbol("shifted")),
        },
        "inputs": ["vec"],
        "target": "z",
    }

    return [
        PayloadCase(
            name="scalar_arithmetic",
            payload=scalar_payload,
            scope={"x": 10.5, "y": 6.25, "offset": 4.0, "scale": 2.5},
            python_expression="((x + y) * 2 - offset) / scale",
            mathjs_expression="((x + y) * 2 - offset) / scale",
        ),
        PayloadCase(
            name="conditional_distance",
            payload=conditional_payload,
            scope={"x": 12.0, "y": 9.0, "limit": 10.0},
            python_expression=(
                "sqrt(x ** 2 + y ** 2) if sqrt(x ** 2 + y ** 2) > limit else 0"
            ),
            mathjs_expression=("sqrt(x ^ 2 + y ^ 2) > limit ? sqrt(x ^ 2 + y ^ 2) : 0"),
        ),
        PayloadCase(
            name="aggregate_helpers",
            payload=aggregate_payload,
            scope={"a": 7.25, "b": -3.5, "c": 11.0},
            python_expression="sum(a, b, c) + max(a, b, c) - min(a, b, c)",
            mathjs_expression="sum(a, b, c) + max(a, b, c) - min(a, b, c)",
        ),
        PayloadCase(
            name="numpy_vector",
            payload=vector_payload,
            scope={"vec": np.linspace(0.1, 10.0, 128)},
            python_expression="sqrt(vec ** 2 + 1)",
            mathjs_expression=None,
            supports_simpleeval=False,
            iterations=5_000,
            build_iterations=250,
        ),
    ]
