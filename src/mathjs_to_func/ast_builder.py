"""Convert serialized math.js AST nodes into Python AST nodes."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable as AbcIterable
from collections.abc import Mapping
from collections.abc import Mapping as AbcMapping
from typing import Any

from .errors import InvalidNodeError

IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ensure_identifier(name: str, *, expression: str | None) -> str:
    """Validate and return a safe identifier for generated expressions."""
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


class MathJsAstVisitor[T]:
    """Generic visitor for math.js AST nodes."""

    def __init__(self, *, expression_name: str) -> None:
        self.expression_name = expression_name

    def visit(self, node: Mapping[str, Any]) -> T:
        node_type = _extract_type(node)
        method = getattr(self, f"visit_{node_type}", None)
        if method is None:
            raise InvalidNodeError(
                f"Unsupported node type {node_type!r}",
                expression=self.expression_name,
                node=node,
            )
        return method(node)

    def _ensure_mapping(
        self,
        value: Any,
        *,
        node: Mapping[str, Any],
        message: str,
    ) -> Mapping[str, Any]:
        if not isinstance(value, AbcMapping):
            raise InvalidNodeError(
                message,
                expression=self.expression_name,
                node=node,
            )
        return value

    def _ensure_iterable(
        self,
        value: Any,
        *,
        node: Mapping[str, Any],
        message: str,
    ) -> list[Any]:
        if not isinstance(value, AbcIterable):
            raise InvalidNodeError(
                message,
                expression=self.expression_name,
                node=node,
            )
        return list(value)


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
        lowered = value.lower()
        if "." in value or "e" in lowered:
            try:
                parsed = float(value)
            except ValueError as exc:
                raise InvalidNodeError(
                    f"Invalid numeric literal: {value!r}",
                    expression=expression,
                    node=None,
                ) from exc
            if not math.isfinite(parsed):
                raise InvalidNodeError(
                    "Non-finite literal encountered",
                    expression=expression,
                    node=None,
                )
            return parsed
        try:
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


class MathJsAstBuilder(MathJsAstVisitor[ast.expr]):
    """Translate math.js AST nodes into Python AST expressions."""

    def __init__(
        self,
        *,
        expression_name: str,
        helper_names: Mapping[str, str],
    ) -> None:
        super().__init__(expression_name=expression_name)
        self.helper_names = helper_names

    def build(self, node: Mapping[str, Any]) -> ast.expr:
        return self.visit(node)

    def visit_ConstantNode(self, node: Mapping[str, Any]) -> ast.expr:
        value_type = node.get("valueType")
        value = node.get("value")
        if value_type in {None, "number"}:
            number = _to_number(value, expression=self.expression_name)
            return ast.Constant(value=number)
        if value_type == "boolean":
            parsed = value.lower() == "true" if isinstance(value, str) else bool(value)
            return ast.Constant(value=parsed)
        if value_type == "null":
            return ast.Constant(value=None)
        raise InvalidNodeError(
            f"Unsupported constant value type: {value_type!r}",
            expression=self.expression_name,
            node=node,
        )

    def visit_SymbolNode(self, node: Mapping[str, Any]) -> ast.expr:
        name = node.get("name")
        if not isinstance(name, str):
            raise InvalidNodeError(
                "SymbolNode missing name",
                expression=self.expression_name,
                node=node,
            )
        safe_name = ensure_identifier(name, expression=self.expression_name)
        return ast.Name(id=safe_name, ctx=ast.Load())

    def visit_ParenthesisNode(self, node: Mapping[str, Any]) -> ast.expr:
        content = node.get("content") or node.get("expr")
        child = self._ensure_mapping(
            content,
            node=node,
            message="ParenthesisNode missing child content",
        )
        return self.visit(child)

    def visit_OperatorNode(self, node: Mapping[str, Any]) -> ast.expr:
        args_list = self._ensure_iterable(
            node.get("args"),
            node=node,
            message="OperatorNode missing args",
        )
        fn = node.get("fn")
        if len(args_list) == 1:
            child = self._ensure_mapping(
                args_list[0],
                node=node,
                message="OperatorNode child must be object",
            )
            return self._visit_unary_operator(fn, child)
        if len(args_list) == 2:
            left_node = self._ensure_mapping(
                args_list[0],
                node=node,
                message="OperatorNode children must be objects",
            )
            right_node = self._ensure_mapping(
                args_list[1],
                node=node,
                message="OperatorNode children must be objects",
            )
            return self._visit_binary_operator(fn, left_node, right_node)
        raise InvalidNodeError(
            "OperatorNode args must be unary or binary",
            expression=self.expression_name,
            node=node,
        )

    def _visit_unary_operator(
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
        operand = self.visit(child)
        op = ast.USub() if fn == "unaryMinus" else ast.UAdd()
        return ast.UnaryOp(op=op, operand=operand)

    def _visit_binary_operator(
        self,
        fn: Any,
        left_node: Mapping[str, Any],
        right_node: Mapping[str, Any],
    ) -> ast.expr:
        left = self.visit(left_node)
        right = self.visit(right_node)
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

    def visit_FunctionNode(self, node: Mapping[str, Any]) -> ast.expr:
        raw_fn = node.get("fn")
        fn_name = raw_fn.get("name") if isinstance(raw_fn, AbcMapping) else raw_fn
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
        args_list = self._ensure_iterable(
            args,
            node=node,
            message="FunctionNode args must be iterable",
        )
        call_args = []
        for arg in args_list:
            child = self._ensure_mapping(
                arg,
                node=node,
                message="FunctionNode argument must be object",
            )
            call_args.append(self.visit(child))

        if normalized == "ifnull" and len(call_args) != 2:
            raise InvalidNodeError(
                "ifnull expects exactly two arguments",
                expression=self.expression_name,
                node=node,
            )

        if normalized in {"min", "max", "sum"} and not call_args:
            raise InvalidNodeError(
                f"{normalized} requires at least one argument",
                expression=self.expression_name,
                node=node,
            )
        return ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=call_args,
            keywords=[],
        )

    def visit_ArrayNode(self, node: Mapping[str, Any]) -> ast.expr:
        items = node.get("items")
        items_list = self._ensure_iterable(
            items,
            node=node,
            message="ArrayNode items must be iterable",
        )
        elts: list[ast.expr] = []
        for item in items_list:
            element = self._ensure_mapping(
                item,
                node=node,
                message="ArrayNode element must be object",
            )
            elts.append(self.visit(element))
        return ast.List(elts=elts, ctx=ast.Load())


class SymbolDependencyCollector(MathJsAstVisitor[set[str]]):
    """Collect symbol dependencies from math.js AST nodes."""

    def collect(self, node: Mapping[str, Any]) -> set[str]:
        return self.visit(node)

    def visit_ConstantNode(self, node: Mapping[str, Any]) -> set[str]:
        return set()

    def visit_SymbolNode(self, node: Mapping[str, Any]) -> set[str]:
        name = node.get("name")
        if not isinstance(name, str):
            raise InvalidNodeError(
                "SymbolNode missing name",
                expression=self.expression_name,
                node=node,
            )
        ensure_identifier(name, expression=self.expression_name)
        return {name}

    def visit_ParenthesisNode(self, node: Mapping[str, Any]) -> set[str]:
        content = node.get("content") or node.get("expr")
        child = self._ensure_mapping(
            content,
            node=node,
            message="ParenthesisNode missing child content",
        )
        return self.visit(child)

    def visit_OperatorNode(self, node: Mapping[str, Any]) -> set[str]:
        args_list = self._ensure_iterable(
            node.get("args"),
            node=node,
            message="OperatorNode missing args",
        )
        result: set[str] = set()
        for child in args_list:
            mapping = self._ensure_mapping(
                child,
                node=node,
                message="OperatorNode argument must be object",
            )
            result.update(self.visit(mapping))
        return result

    def visit_FunctionNode(self, node: Mapping[str, Any]) -> set[str]:
        args = node.get("args") or []
        args_list = self._ensure_iterable(
            args,
            node=node,
            message="FunctionNode args must be iterable",
        )
        result: set[str] = set()
        for arg in args_list:
            child = self._ensure_mapping(
                arg,
                node=node,
                message="FunctionNode argument must be object",
            )
            result.update(self.visit(child))
        return result

    def visit_ArrayNode(self, node: Mapping[str, Any]) -> set[str]:
        items = node.get("items")
        items_list = self._ensure_iterable(
            items,
            node=node,
            message="ArrayNode items must be iterable",
        )
        result: set[str] = set()
        for item in items_list:
            element = self._ensure_mapping(
                item,
                node=node,
                message="ArrayNode element must be object",
            )
            result.update(self.visit(element))
        return result


__all__ = [
    "IDENTIFIER_PATTERN",
    "MathJsAstBuilder",
    "MathJsAstVisitor",
    "SymbolDependencyCollector",
    "ensure_identifier",
]
