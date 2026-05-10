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
MATHJS_BUILTIN_SYMBOLS = {
    "e": math.e,
    "E": math.e,
    "i": 1j,
    "Infinity": math.inf,
    "LN2": math.log(2),
    "LN10": math.log(10),
    "LOG2E": math.log2(math.e),
    "LOG10E": math.log10(math.e),
    "NaN": math.nan,
    "null": None,
    "phi": (1 + math.sqrt(5)) / 2,
    "pi": math.pi,
    "PI": math.pi,
    "SQRT1_2": math.sqrt(1 / 2),
    "SQRT2": math.sqrt(2),
    "tau": math.tau,
    "undefined": None,
}

UNARY_OPERATOR_FUNCTIONS = {"not", "unaryMinus", "unaryPlus"}
BINARY_OPERATOR_FUNCTIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "pow",
    "mod",
    "nullish",
    "and",
    "or",
    "larger",
    "largerEq",
    "smaller",
    "smallerEq",
    "equal",
    "unequal",
    "xor",
}
MIN_ARITY_FUNCTIONS = {
    "gcd",
    "hypot",
    "lcm",
    "log",
    "log1p",
    "max",
    "mean",
    "median",
    "min",
    "mode",
    "sum",
    "std",
    "permutations",
    "round",
    "variance",
}
EXACT_ARITY_FUNCTIONS = {
    "abs": 1,
    "acos": 1,
    "acosh": 1,
    "asin": 1,
    "asinh": 1,
    "atan": 1,
    "atan2": 2,
    "atanh": 1,
    "cbrt": 1,
    "ceil": 1,
    "clamp": 3,
    "combinations": 2,
    "cos": 1,
    "cosh": 1,
    "exp": 1,
    "factorial": 1,
    "floor": 1,
    "ifnull": 2,
    "log2": 1,
    "log10": 1,
    "sign": 1,
    "sin": 1,
    "sinh": 1,
    "sqrt": 1,
    "tan": 1,
    "tanh": 1,
}
MAX_ARITY_FUNCTIONS = {
    "log": 2,
    "log1p": 2,
    "permutations": 2,
    "round": 2,
}


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
        return value
    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        non_finite_literals = {
            "inf": math.inf,
            "+inf": math.inf,
            "infinity": math.inf,
            "+infinity": math.inf,
            "-inf": -math.inf,
            "-infinity": -math.inf,
            "nan": math.nan,
            "+nan": math.nan,
            "-nan": math.nan,
        }
        if lowered in non_finite_literals:
            return non_finite_literals[lowered]
        if "." in stripped or "e" in lowered:
            try:
                return float(stripped)
            except ValueError as exc:
                raise InvalidNodeError(
                    f"Invalid numeric literal: {value!r}",
                    expression=expression,
                    node=None,
                ) from exc
        try:
            return int(stripped, 10)
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
        local_names: set[str] | None = None,
    ) -> None:
        super().__init__(expression_name=expression_name)
        self.helper_names = helper_names
        self.local_names = local_names or set()

    def build(self, node: Mapping[str, Any]) -> ast.expr:
        return self.visit(node)

    def _defer(self, expression: ast.expr) -> ast.Lambda:
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=expression,
        )

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
        if safe_name not in self.local_names and safe_name in MATHJS_BUILTIN_SYMBOLS:
            return ast.Constant(value=MATHJS_BUILTIN_SYMBOLS[safe_name])
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
        if fn == "not":
            return ast.Call(
                func=ast.Name(id="__mj_not", ctx=ast.Load()),
                args=[self.visit(child)],
                keywords=[],
            )
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
            case "nullish":
                return ast.Call(
                    func=ast.Name(id="__mj_lazy_ifnull", ctx=ast.Load()),
                    args=[left, self._defer(right)],
                    keywords=[],
                )
            case "and" | "or":
                helper_name = "__mj_lazy_and" if fn == "and" else "__mj_lazy_or"
                return ast.Call(
                    func=ast.Name(id=helper_name, ctx=ast.Load()),
                    args=[left, self._defer(right)],
                    keywords=[],
                )
            case (
                "larger"
                | "largerEq"
                | "smaller"
                | "smallerEq"
                | "equal"
                | "unequal"
                | "xor"
            ):
                helper_name = {
                    "larger": "__mj_larger",
                    "largerEq": "__mj_larger_eq",
                    "smaller": "__mj_smaller",
                    "smallerEq": "__mj_smaller_eq",
                    "equal": "__mj_equal",
                    "unequal": "__mj_unequal",
                    "xor": "__mj_xor",
                }[fn]
                return ast.Call(
                    func=ast.Name(id=helper_name, ctx=ast.Load()),
                    args=[left, right],
                    keywords=[],
                )
            case _:
                raise InvalidNodeError(
                    f"Unsupported binary operator: {fn!r}",
                    expression=self.expression_name,
                    node=None,
                )
        return ast.BinOp(left=left, op=op, right=right)

    def _function_name(self, node: Mapping[str, Any]) -> str:
        raw_fn = node.get("fn")
        fn_name = raw_fn.get("name") if isinstance(raw_fn, AbcMapping) else raw_fn
        if not isinstance(fn_name, str):
            raise InvalidNodeError(
                "FunctionNode missing function name",
                expression=self.expression_name,
                node=node,
            )
        return fn_name.strip()

    def _function_arg_nodes(self, node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        args = node.get("args") or []
        args_list = self._ensure_iterable(
            args,
            node=node,
            message="FunctionNode args must be iterable",
        )
        return [
            self._ensure_mapping(
                arg,
                node=node,
                message="FunctionNode argument must be object",
            )
            for arg in args_list
        ]

    def _visit_operator_function_alias(
        self,
        normalized: str,
        arg_nodes: list[Mapping[str, Any]],
        *,
        node: Mapping[str, Any],
    ) -> ast.expr | None:
        if normalized in UNARY_OPERATOR_FUNCTIONS:
            if len(arg_nodes) != 1:
                raise InvalidNodeError(
                    f"{normalized} expects exactly one argument",
                    expression=self.expression_name,
                    node=node,
                )
            return self._visit_unary_operator(normalized, arg_nodes[0])

        if normalized in BINARY_OPERATOR_FUNCTIONS:
            if len(arg_nodes) != 2:
                raise InvalidNodeError(
                    f"{normalized} expects exactly two arguments",
                    expression=self.expression_name,
                    node=node,
                )
            return self._visit_binary_operator(normalized, arg_nodes[0], arg_nodes[1])
        return None

    def _validate_function_arity(
        self,
        normalized: str,
        arg_count: int,
        *,
        node: Mapping[str, Any],
    ) -> None:
        if normalized in EXACT_ARITY_FUNCTIONS:
            expected = EXACT_ARITY_FUNCTIONS[normalized]
            if arg_count != expected:
                raise InvalidNodeError(
                    f"{normalized} expects exactly {expected} arguments",
                    expression=self.expression_name,
                    node=node,
                )

        if normalized in MIN_ARITY_FUNCTIONS and arg_count == 0:
            raise InvalidNodeError(
                f"{normalized} requires at least one argument",
                expression=self.expression_name,
                node=node,
            )

        max_arity = MAX_ARITY_FUNCTIONS.get(normalized)
        if max_arity is not None and arg_count > max_arity:
            raise InvalidNodeError(
                f"{normalized} expects at most {max_arity} arguments",
                expression=self.expression_name,
                node=node,
            )

    def visit_FunctionNode(self, node: Mapping[str, Any]) -> ast.expr:
        normalized = self._function_name(node)
        arg_nodes = self._function_arg_nodes(node)

        alias = self._visit_operator_function_alias(normalized, arg_nodes, node=node)
        if alias is not None:
            return alias

        helper_name = self.helper_names.get(normalized)
        if helper_name is None:
            raise InvalidNodeError(
                f"Unsupported function {normalized!r}",
                expression=self.expression_name,
                node=node,
            )

        self._validate_function_arity(normalized, len(arg_nodes), node=node)
        return ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=[self.visit(arg) for arg in arg_nodes],
            keywords=[],
        )

    def _build_range_call(
        self,
        node: Mapping[str, Any],
        *,
        helper_name: str,
    ) -> ast.Call:
        start = self._ensure_mapping(
            node.get("start"),
            node=node,
            message="RangeNode missing start",
        )
        end = self._ensure_mapping(
            node.get("end"),
            node=node,
            message="RangeNode missing end",
        )
        args = [self.visit(start), self.visit(end)]
        step = node.get("step")
        if step is not None:
            step_node = self._ensure_mapping(
                step,
                node=node,
                message="RangeNode step must be object",
            )
            args.append(self.visit(step_node))
        return ast.Call(
            func=ast.Name(id=helper_name, ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    def visit_RangeNode(self, node: Mapping[str, Any]) -> ast.expr:
        return self._build_range_call(node, helper_name="__mj_range")

    def _build_index_dimensions(self, node: Mapping[str, Any]) -> list[ast.expr]:
        if node.get("dotNotation") is True:
            raise InvalidNodeError(
                "AccessorNode dot notation is not supported",
                expression=self.expression_name,
                node=node,
            )
        dimensions = self._ensure_iterable(
            node.get("dimensions"),
            node=node,
            message="IndexNode dimensions must be iterable",
        )
        result: list[ast.expr] = []
        for dimension in dimensions:
            dimension_node = self._ensure_mapping(
                dimension,
                node=node,
                message="IndexNode dimension must be object",
            )
            if _extract_type(dimension_node) == "RangeNode":
                result.append(
                    self._build_range_call(
                        dimension_node,
                        helper_name="__mj_index_range",
                    ),
                )
            else:
                result.append(
                    ast.Call(
                        func=ast.Name(id="__mj_index", ctx=ast.Load()),
                        args=[self.visit(dimension_node)],
                        keywords=[],
                    ),
                )
        return result

    def visit_IndexNode(self, node: Mapping[str, Any]) -> ast.expr:
        raise InvalidNodeError(
            "IndexNode is only supported inside AccessorNode",
            expression=self.expression_name,
            node=node,
        )

    def visit_AccessorNode(self, node: Mapping[str, Any]) -> ast.expr:
        object_node = self._ensure_mapping(
            node.get("object"),
            node=node,
            message="AccessorNode missing object",
        )
        index_node = self._ensure_mapping(
            node.get("index"),
            node=node,
            message="AccessorNode missing index",
        )
        if _extract_type(index_node) != "IndexNode":
            raise InvalidNodeError(
                "AccessorNode index must be IndexNode",
                expression=self.expression_name,
                node=node,
            )
        return ast.Call(
            func=ast.Name(id="__mj_access", ctx=ast.Load()),
            args=[
                self.visit(object_node),
                *self._build_index_dimensions(index_node),
            ],
            keywords=[],
        )

    def visit_ObjectNode(self, node: Mapping[str, Any]) -> ast.expr:
        properties = node.get("properties")
        if not isinstance(properties, AbcMapping):
            raise InvalidNodeError(
                "ObjectNode properties must be mapping",
                expression=self.expression_name,
                node=node,
            )
        keys: list[ast.expr | None] = []
        values: list[ast.expr] = []
        for key, value in properties.items():
            if not isinstance(key, str):
                raise InvalidNodeError(
                    "ObjectNode property keys must be strings",
                    expression=self.expression_name,
                    node=node,
                )
            value_node = self._ensure_mapping(
                value,
                node=node,
                message="ObjectNode property value must be object",
            )
            keys.append(ast.Constant(value=key))
            values.append(self.visit(value_node))
        return ast.Dict(keys=keys, values=values)

    def visit_RelationalNode(self, node: Mapping[str, Any]) -> ast.expr:
        conditionals = self._ensure_iterable(
            node.get("conditionals"),
            node=node,
            message="RelationalNode conditionals must be iterable",
        )
        params = self._ensure_iterable(
            node.get("params"),
            node=node,
            message="RelationalNode params must be iterable",
        )
        if len(params) < 2 or len(conditionals) != len(params) - 1:
            raise InvalidNodeError(
                "RelationalNode requires one fewer conditional than params",
                expression=self.expression_name,
                node=node,
            )

        allowed = {
            "smaller",
            "larger",
            "smallerEq",
            "largerEq",
            "equal",
            "unequal",
        }
        for conditional in conditionals:
            if not isinstance(conditional, str) or conditional not in allowed:
                raise InvalidNodeError(
                    f"Unsupported relational conditional: {conditional!r}",
                    expression=self.expression_name,
                    node=node,
                )

        param_calls: list[ast.expr] = []
        for param in params:
            child = self._ensure_mapping(
                param,
                node=node,
                message="RelationalNode param must be object",
            )
            param_calls.append(self._defer(self.visit(child)))

        return ast.Call(
            func=ast.Name(id="__mj_relational", ctx=ast.Load()),
            args=[
                ast.Tuple(
                    elts=[ast.Constant(value=item) for item in conditionals],
                    ctx=ast.Load(),
                ),
                *param_calls,
            ],
            keywords=[],
        )

    def visit_ConditionalNode(self, node: Mapping[str, Any]) -> ast.expr:
        condition = self._ensure_mapping(
            node.get("condition"),
            node=node,
            message="ConditionalNode missing condition",
        )
        true_expr = self._ensure_mapping(
            node.get("trueExpr"),
            node=node,
            message="ConditionalNode missing true expression",
        )
        false_expr = self._ensure_mapping(
            node.get("falseExpr"),
            node=node,
            message="ConditionalNode missing false expression",
        )
        return ast.Call(
            func=ast.Name(id="__mj_lazy_where", ctx=ast.Load()),
            args=[
                self.visit(condition),
                self._defer(self.visit(true_expr)),
                self._defer(self.visit(false_expr)),
            ],
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
        result: set[str] = set()
        args = node.get("args") or []
        args_list = self._ensure_iterable(
            args,
            node=node,
            message="FunctionNode args must be iterable",
        )
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

    def visit_RangeNode(self, node: Mapping[str, Any]) -> set[str]:
        result: set[str] = set()
        for key, message in (
            ("start", "RangeNode missing start"),
            ("end", "RangeNode missing end"),
        ):
            child = self._ensure_mapping(node.get(key), node=node, message=message)
            result.update(self.visit(child))
        step = node.get("step")
        if step is not None:
            child = self._ensure_mapping(
                step,
                node=node,
                message="RangeNode step must be object",
            )
            result.update(self.visit(child))
        return result

    def visit_IndexNode(self, node: Mapping[str, Any]) -> set[str]:
        dimensions = self._ensure_iterable(
            node.get("dimensions"),
            node=node,
            message="IndexNode dimensions must be iterable",
        )
        result: set[str] = set()
        for dimension in dimensions:
            child = self._ensure_mapping(
                dimension,
                node=node,
                message="IndexNode dimension must be object",
            )
            result.update(self.visit(child))
        return result

    def visit_AccessorNode(self, node: Mapping[str, Any]) -> set[str]:
        result: set[str] = set()
        object_node = self._ensure_mapping(
            node.get("object"),
            node=node,
            message="AccessorNode missing object",
        )
        index_node = self._ensure_mapping(
            node.get("index"),
            node=node,
            message="AccessorNode missing index",
        )
        result.update(self.visit(object_node))
        result.update(self.visit(index_node))
        return result

    def visit_ObjectNode(self, node: Mapping[str, Any]) -> set[str]:
        properties = node.get("properties")
        if not isinstance(properties, AbcMapping):
            raise InvalidNodeError(
                "ObjectNode properties must be mapping",
                expression=self.expression_name,
                node=node,
            )
        result: set[str] = set()
        for value in properties.values():
            child = self._ensure_mapping(
                value,
                node=node,
                message="ObjectNode property value must be object",
            )
            result.update(self.visit(child))
        return result

    def visit_ConditionalNode(self, node: Mapping[str, Any]) -> set[str]:
        result: set[str] = set()
        for key, message in (
            ("condition", "ConditionalNode missing condition"),
            ("trueExpr", "ConditionalNode missing true expression"),
            ("falseExpr", "ConditionalNode missing false expression"),
        ):
            child = self._ensure_mapping(node.get(key), node=node, message=message)
            result.update(self.visit(child))
        return result

    def visit_RelationalNode(self, node: Mapping[str, Any]) -> set[str]:
        params = self._ensure_iterable(
            node.get("params"),
            node=node,
            message="RelationalNode params must be iterable",
        )
        result: set[str] = set()
        for param in params:
            child = self._ensure_mapping(
                param,
                node=node,
                message="RelationalNode param must be object",
            )
            result.update(self.visit(child))
        return result


__all__ = [
    "IDENTIFIER_PATTERN",
    "MATHJS_BUILTIN_SYMBOLS",
    "MathJsAstBuilder",
    "MathJsAstVisitor",
    "SymbolDependencyCollector",
    "ensure_identifier",
]
