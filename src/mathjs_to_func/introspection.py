"""Formatting and dependency graph helpers for math.js payloads."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .ast_builder import MATHJS_BUILTIN_SYMBOLS, SymbolDependencyCollector
from .errors import ExpressionError

_OPERATOR_TEXT = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "pow": "^",
    "mod": "mod",
    "and": "and",
    "or": "or",
    "xor": "xor",
    "larger": ">",
    "largerEq": ">=",
    "smaller": "<",
    "smallerEq": "<=",
    "equal": "==",
    "unequal": "!=",
    "nullish": "??",
}
_RELATIONAL_TEXT = {
    "larger": ">",
    "largerEq": ">=",
    "smaller": "<",
    "smallerEq": "<=",
    "equal": "==",
    "unequal": "!=",
}


def _node_type(node: Mapping[str, Any]) -> str:
    value = node.get("type") or node.get("mathjs")
    if not isinstance(value, str):
        raise ExpressionError("math.js node is missing a type")
    return value


def _is_node(value: Mapping[str, Any]) -> bool:
    return isinstance(value.get("type") or value.get("mathjs"), str)


def _function_name(node: Mapping[str, Any]) -> str:
    raw = node.get("fn")
    name = raw.get("name") if isinstance(raw, Mapping) else raw
    if not isinstance(name, str):
        raise ExpressionError("FunctionNode missing function name")
    return name


def _constant_text(node: Mapping[str, Any]) -> str:
    value_type = node.get("valueType")
    value = node.get("value")
    if value_type == "boolean":
        return "true" if (value == "true" or value is True) else "false"
    if value_type == "null" or value is None:
        return "null"
    if value_type == "string" or (
        value_type is None and isinstance(value, str) and not _looks_numeric(value)
    ):
        return json.dumps(value)
    return str(value)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _render(  # noqa: C901, PLR0912, PLR0915
    node: Mapping[str, Any],
    *,
    tex: bool = False,
) -> str:
    node_type = _node_type(node)
    if node_type == "ConstantNode":
        return _constant_text(node)
    if node_type == "SymbolNode":
        name = node.get("name")
        if not isinstance(name, str):
            raise ExpressionError("SymbolNode missing name")
        return name
    if node_type == "ParenthesisNode":
        child = node.get("content") or node.get("expr")
        if not isinstance(child, Mapping):
            raise ExpressionError("ParenthesisNode missing content")
        return f"({_render(child, tex=tex)})"
    if node_type == "ArrayNode":
        items = node.get("items") or []
        return "[" + ", ".join(_render(item, tex=tex) for item in items) + "]"
    if node_type == "ObjectNode":
        properties = node.get("properties") or {}
        return (
            "{"
            + ", ".join(
                f"{key}: {_render(value, tex=tex)}" for key, value in properties.items()
            )
            + "}"
        )
    if node_type == "RangeNode":
        start = _render(_require_mapping(node.get("start")), tex=tex)
        end = _render(_require_mapping(node.get("end")), tex=tex)
        step = node.get("step")
        if step is None:
            return f"{start}:{end}"
        return f"{start}:{_render(_require_mapping(step), tex=tex)}:{end}"
    if node_type == "AccessorNode":
        obj = _render(_require_mapping(node.get("object")), tex=tex)
        index = _require_mapping(node.get("index"))
        dimensions = index.get("dimensions") or []
        return f"{obj}[{', '.join(_render(item, tex=tex) for item in dimensions)}]"
    if node_type == "ConditionalNode":
        condition = _render(_require_mapping(node.get("condition")), tex=tex)
        true_expr = _render(_require_mapping(node.get("trueExpr")), tex=tex)
        false_expr = _render(_require_mapping(node.get("falseExpr")), tex=tex)
        return f"{condition} ? {true_expr} : {false_expr}"
    if node_type == "RelationalNode":
        conditionals = node.get("conditionals") or []
        params = node.get("params") or []
        rendered = [_render(item, tex=tex) for item in params]
        pieces: list[str] = []
        for index, value in enumerate(rendered):
            pieces.append(value)
            if index < len(conditionals):
                pieces.append(
                    _RELATIONAL_TEXT.get(conditionals[index], conditionals[index]),
                )
        return " ".join(pieces)
    if node_type == "FunctionNode":
        name = _function_name(node)
        args = ", ".join(_render(arg, tex=tex) for arg in node.get("args") or [])
        if tex and name in {"sin", "cos", "tan", "log", "sqrt"}:
            if name == "sqrt":
                return f"\\sqrt{{{args}}}"
            return f"\\{name}\\left({args}\\right)"
        return f"{name}({args})"
    if node_type == "OperatorNode":
        args = node.get("args") or []
        fn = node.get("fn")
        if node.get("isPercentage") is True and len(args) == 2:
            return f"{_render(_require_mapping(args[0]), tex=tex)}%"
        if len(args) == 1:
            child = _render(_require_mapping(args[0]), tex=tex)
            if fn == "unaryMinus":
                return f"-{child}"
            if fn == "unaryPlus":
                return f"+{child}"
            if fn == "not":
                return f"not {child}"
            if fn == "percentage" or node.get("op") == "%":
                return f"{child}%"
        if len(args) == 2:
            left = _render(_require_mapping(args[0]), tex=tex)
            right = _render(_require_mapping(args[1]), tex=tex)
            if tex and fn == "divide":
                return f"\\frac{{{left}}}{{{right}}}"
            if tex and fn == "multiply":
                return f"{left} \\cdot {right}"
            if tex and fn == "pow":
                return f"{left}^{{{right}}}"
            return f"{left} {_OPERATOR_TEXT.get(fn, str(fn))} {right}"
    raise ExpressionError(f"Unsupported node type for rendering: {node_type}")


def _require_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExpressionError("Expected math.js child node")
    return value


def _payload_parts(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    expressions = payload.get("expressions")
    target = payload.get("target")
    if not isinstance(expressions, Mapping):
        raise ExpressionError("Payload missing expressions mapping")
    if isinstance(target, str):
        return expressions, (target,)
    if isinstance(target, Sequence) and not isinstance(target, (bytes, bytearray)):
        return expressions, tuple(str(item) for item in target)
    raise ExpressionError("Payload missing target")


def _render_payload(payload: Mapping[str, Any], *, tex: bool) -> str | dict[str, str]:
    if _is_node(payload):
        return _render(payload, tex=tex)
    expressions, targets = _payload_parts(payload)
    rendered = {
        target: _render(_require_mapping(expressions[target]), tex=tex)
        for target in targets
    }
    if len(targets) == 1:
        return rendered[targets[0]]
    return rendered


def to_string(payload: Mapping[str, Any]) -> str | dict[str, str]:
    """Render a math.js node or evaluator payload as infix text."""
    return _render_payload(payload, tex=False)


def to_tex(payload: Mapping[str, Any]) -> str | dict[str, str]:
    """Render a math.js node or evaluator payload as simple LaTeX."""
    return _render_payload(payload, tex=True)


def _dependency_map(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, set[str]]]:
    expressions, _targets = _payload_parts(payload)
    expression_names = set(expressions)
    result: dict[str, set[str]] = {}
    for name, node in expressions.items():
        collector = SymbolDependencyCollector(expression_name=name)
        deps = collector.collect(_require_mapping(node))
        result[name] = {
            dep
            for dep in deps
            if dep in expression_names or dep not in MATHJS_BUILTIN_SYMBOLS
        }
    return expressions, result


def _input_dependencies(
    target: str,
    dependency_map: Mapping[str, set[str]],
    expression_names: set[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    inputs: set[str] = set()
    stack = [target]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for dep in dependency_map.get(current, set()):
            if dep in expression_names:
                stack.append(dep)
            else:
                inputs.add(dep)
    return tuple(sorted(inputs))


def inputs_referenced_per_target(
    payload: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    """Return input identifiers needed by each target expression."""
    expressions, targets = _payload_parts(payload)
    _exprs, dependency_map = _dependency_map(payload)
    expression_names = set(expressions)
    return {
        target: _input_dependencies(target, dependency_map, expression_names)
        for target in targets
    }


def to_dot(payload: Mapping[str, Any]) -> str:
    """Render the evaluator dependency graph in Graphviz DOT format."""
    expressions, dependency_map = _dependency_map(payload)
    lines = ["digraph mathjs {"]
    lines.extend(f'  "{name}";' for name in sorted(expressions))
    inputs = sorted(
        {
            dep
            for deps in dependency_map.values()
            for dep in deps
            if dep not in expressions
        },
    )
    lines.extend(f'  "{name}" [shape=box];' for name in inputs)
    for expr, deps in sorted(dependency_map.items()):
        lines.extend(f'  "{dep}" -> "{expr}";' for dep in sorted(deps))
    lines.append("}")
    return "\n".join(lines)


def to_mermaid(payload: Mapping[str, Any]) -> str:
    """Render the evaluator dependency graph in Mermaid flowchart syntax."""
    expressions, dependency_map = _dependency_map(payload)
    lines = ["graph TD"]
    lines.extend(f"  {name}[{name}]" for name in sorted(expressions))
    for expr, deps in sorted(dependency_map.items()):
        lines.extend(f"  {dep}[{dep}] --> {expr}[{expr}]" for dep in sorted(deps))
    return "\n".join(lines)


__all__ = [
    "inputs_referenced_per_target",
    "to_dot",
    "to_mermaid",
    "to_string",
    "to_tex",
]
