"""Convert serialized math.js AST nodes into Python AST nodes."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Iterable, Mapping

from .errors import InvalidNodeError

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ensure_identifier(name: str, *, expression: str | None) -> str:
    if not IDENTIFIER_PATTERN.match(name):
        raise InvalidNodeError(
            f"Unsupported identifier name: {name!r}",
            expression=expression,
            node=None,
        )
    return name


def _extract_type(node: Mapping[str, Any]) -> str:
    node_type = node.get("type") or node.get("mathjs")
    if not isinstance(node_type, str):
        raise InvalidNodeError(
            "Node is missing 'type' field",
            expression=None,
            node=node,
        )
    return node_type


def _to_number(value: Any, *, expression: str | None) -> float | int:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidNodeError(
                "Non-finite literal encountered",
                expression=expression,
                node=None,
            )
        return value
    if isinstance(value, str):
        try:
            if "." in value or "e" in value.lower():
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError
                return parsed
            return int(value, 10)
        except ValueError as exc:
            raise InvalidNodeError(
                f"Invalid numeric literal: {value!r}",
                expression=expression,
                node=None,
            ) from exc
    raise InvalidNodeError(
        f"Unsupported literal type: {type(value).__name__}",
        expression=expression,
        node=None,
    )


class MathJsAstBuilder:
    """Translate math.js AST nodes into Python AST expressions."""

    def __init__(
        self,
        *,
        expression_name: str,
        helper_names: Mapping[str, str],
    ) -> None:
        self.expression_name = expression_name
        self.helper_names = helper_names

    def build(self, node: Mapping[str, Any]) -> ast.expr:
        node_type = _extract_type(node)
        method = getattr(self, f"_handle_{node_type}", None)
        if method is None:
            raise InvalidNodeError(
                f"Unsupported node type {node_type!r}",
                expression=self.expression_name,
                node=node,
            )
        return method(node)

    def _handle_ConstantNode(self, node: Mapping[str, Any]) -> ast.expr:
        value_type = node.get("valueType")
        value = node.get("value")
        if value_type in {None, "number"}:
            number = _to_number(value, expression=self.expression_name)
            return ast.Constant(value=number)
        if value_type == "boolean":
            if isinstance(value, str):
                parsed = value.lower() == "true"
            else:
                parsed = bool(value)
            return ast.Constant(value=parsed)
        raise InvalidNodeError(
            f"Unsupported constant value type: {value_type!r}",
            expression=self.expression_name,
            node=node,
        )

    def _handle_SymbolNode(self, node: Mapping[str, Any]) -> ast.expr:
        name = node.get("name")
        if not isinstance(name, str):
            raise InvalidNodeError(
                "SymbolNode missing name",
                expression=self.expression_name,
                node=node,
            )
        safe_name = ensure_identifier(name, expression=self.expression_name)
        return ast.Name(id=safe_name, ctx=ast.Load())

    def _handle_ParenthesisNode(self, node: Mapping[str, Any]) -> ast.expr:
        content = node.get("content") or node.get("expr")
        if not isinstance(content, Mapping):
            raise InvalidNodeError(
                "ParenthesisNode missing child content",
                expression=self.expression_name,
                node=node,
            )
        return self.build(content)

    def _handle_OperatorNode(self, node: Mapping[str, Any]) -> ast.expr:
        args = node.get("args")
        if not isinstance(args, Iterable):
            raise InvalidNodeError(
                "OperatorNode missing args",
                expression=self.expression_name,
                node=node,
            )
        args_list = list(args)
        fn = node.get("fn")
        if len(args_list) == 1:
            child = args_list[0]
            if not isinstance(child, Mapping):
                raise InvalidNodeError(
                    "OperatorNode child must be object",
                    expression=self.expression_name,
                    node=node,
                )
            return self._handle_unary_operator(fn, child)
        if len(args_list) == 2:
            left_node, right_node = args_list
            if not isinstance(left_node, Mapping) or not isinstance(
                right_node, Mapping
            ):
                raise InvalidNodeError(
                    "OperatorNode children must be objects",
                    expression=self.expression_name,
                    node=node,
                )
            return self._handle_binary_operator(fn, left_node, right_node)
        raise InvalidNodeError(
            "OperatorNode args must be unary or binary",
            expression=self.expression_name,
            node=node,
        )

    def _handle_unary_operator(
        self,
        fn: Any,
        child: Mapping[str, Any],
    ) -> ast.expr:
        if fn not in {"unaryMinus", "unaryPlus"}:
            raise InvalidNodeError(
                f"Unsupported unary operator: {fn!r}",
                expression=self.expression_name,
                node=None,
            )
        operand = self.build(child)
        op = ast.USub() if fn == "unaryMinus" else ast.UAdd()
        return ast.UnaryOp(op=op, operand=operand)

    def _handle_binary_operator(
        self,
        fn: Any,
        left_node: Mapping[str, Any],
        right_node: Mapping[str, Any],
    ) -> ast.expr:
        left = self.build(left_node)
        right = self.build(right_node)
        match fn:
            case "add":
                op = ast.Add()
            case "subtract":
                op = ast.Sub()
            case "multiply":
                op = ast.Mult()
            case "divide":
                op = ast.Div()
            case "pow":
                op = ast.Pow()
            case "mod":
                op = ast.Mod()
            case _:
                raise InvalidNodeError(
                    f"Unsupported binary operator: {fn!r}",
                    expression=self.expression_name,
                    node=None,
                )
        return ast.BinOp(left=left, op=op, right=right)

    def _handle_FunctionNode(self, node: Mapping[str, Any]) -> ast.expr:
        raw_fn = node.get("fn")
        if isinstance(raw_fn, Mapping):
            fn_name = raw_fn.get("name")
        else:
            fn_name = raw_fn
        if not isinstance(fn_name, str):
            raise InvalidNodeError(
                "FunctionNode missing function name",
                expression=self.expression_name,
                node=node,
            )
        normalized = fn_name.strip()
        helper_name = self.helper_names.get(normalized)
        if helper_name is None:
            raise InvalidNodeError(
                f"Unsupported function {normalized!r}",
                expression=self.expression_name,
                node=node,
            )
        args = node.get("args") or []
        if not isinstance(args, Iterable):
            raise InvalidNodeError(
                "FunctionNode args must be iterable",
                expression=self.expression_name,
                node=node,
            )
        call_args = []
        for arg in args:
            if not isinstance(arg, Mapping):
                raise InvalidNodeError(
                    "FunctionNode argument must be object",
                    expression=self.expression_name,
                    node=node,
                )
            call_args.append(self.build(arg))
        return ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=call_args,
            keywords=[],
        )

    def _handle_ArrayNode(self, node: Mapping[str, Any]) -> ast.expr:
        items = node.get("items")
        if not isinstance(items, Iterable):
            raise InvalidNodeError(
                "ArrayNode items must be iterable",
                expression=self.expression_name,
                node=node,
            )
        elts: list[ast.expr] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise InvalidNodeError(
                    "ArrayNode element must be object",
                    expression=self.expression_name,
                    node=node,
                )
            elts.append(self.build(item))
        return ast.List(elts=elts, ctx=ast.Load())


__all__ = ["MathJsAstBuilder", "ensure_identifier", "IDENTIFIER_PATTERN"]
