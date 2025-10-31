"""Compile math.js JSON expressions into callable Python functions."""

from __future__ import annotations

import ast
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, Iterable, Mapping

from .ast_builder import (
    MathJsAstBuilder,
    SymbolDependencyCollector,
    ensure_identifier,
)
from .errors import (
    CircularDependencyError,
    ExpressionError,
    InputValidationError,
    InvalidNodeError,
    MissingTargetError,
    UnknownIdentifierError,
)
from .helpers import HELPER_FUNCTIONS, HELPER_NAME_MAP


@dataclass(frozen=True)
class CompilationResult:
    """Metadata for a compiled expression graph."""

    function: Any
    required_inputs: tuple[str, ...]
    evaluation_order: tuple[str, ...]
    module_ast: ast.Module


def _normalise_inputs(inputs: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in inputs:
        if not isinstance(raw, str):
            raise ExpressionError("Input identifiers must be strings")
        name = ensure_identifier(raw, expression=None)
        if name in seen:
            raise ExpressionError(f"Duplicate input identifier: {name}")
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _validate_expressions(
    expressions: Mapping[str, Mapping[str, Any]],
    input_names: set[str],
) -> dict[str, Mapping[str, Any]]:
    validated: dict[str, Mapping[str, Any]] = {}
    for raw_name, node in expressions.items():
        if not isinstance(raw_name, str):
            raise ExpressionError("Expression identifier must be string")
        name = ensure_identifier(raw_name, expression=None)
        if name in validated:
            raise ExpressionError(f"Duplicate expression identifier: {name}")
        if name in input_names:
            raise ExpressionError(
                f"Expression identifier {name!r} conflicts with input identifier",
            )
        if not isinstance(node, AbcMapping):
            raise InvalidNodeError(
                "Expression definition must be mapping",
                expression=name,
                node=None,
            )
        validated[name] = node
    return validated


def _dependency_closure(
    target: str,
    dependency_map: Mapping[str, set[str]],
    expression_names: set[str],
) -> set[str]:
    needed: set[str] = set()
    stack = [target]
    while stack:
        current = stack.pop()
        if current in needed:
            continue
        needed.add(current)
        for dep in dependency_map[current]:
            if dep in expression_names:
                stack.append(dep)
    return needed


def _build_function_ast(
    *,
    evaluation_order: tuple[str, ...],
    expressions: Mapping[str, Mapping[str, Any]],
    required_inputs: tuple[str, ...],
    allowed_inputs: tuple[str, ...],
    target: str,
) -> ast.Module:
    builder_cache: dict[str, MathJsAstBuilder] = {}

    def _builder(expr_name: str) -> MathJsAstBuilder:
        if expr_name not in builder_cache:
            builder_cache[expr_name] = MathJsAstBuilder(
                expression_name=expr_name,
                helper_names=HELPER_NAME_MAP,
            )
        return builder_cache[expr_name]

    scope_arg = ast.arg(arg="scope")
    body: list[ast.stmt] = []

    isinstance_call = ast.Call(
        func=ast.Name(id="isinstance", ctx=ast.Load()),
        args=[
            ast.Name(id="scope", ctx=ast.Load()),
            ast.Name(id="Mapping", ctx=ast.Load()),
        ],
        keywords=[],
    )
    body.append(
        ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=isinstance_call),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="InputValidationError", ctx=ast.Load()),
                        args=[ast.Constant(value="Inputs payload must be a mapping")],
                        keywords=[],
                    ),
                    cause=None,
                ),
            ],
            orelse=[],
        ),
    )

    required_set = ast.Set(elts=[ast.Constant(value=name) for name in required_inputs])
    scope_key_call = ast.Call(
        func=ast.Name(id="set", ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="scope", ctx=ast.Load()),
                    attr="keys",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
        ],
        keywords=[],
    )
    body.append(
        ast.Assign(
            targets=[ast.Name(id="_scope_keys", ctx=ast.Store())],
            value=scope_key_call,
        ),
    )
    body.append(
        ast.Assign(
            targets=[ast.Name(id="_missing", ctx=ast.Store())],
            value=ast.BinOp(
                left=required_set,
                op=ast.Sub(),
                right=ast.Name(id="_scope_keys", ctx=ast.Load()),
            ),
        ),
    )

    missing_message = ast.JoinedStr(
        values=[
            ast.Constant(value="Missing required inputs: "),
            ast.FormattedValue(
                value=ast.Call(
                    func=ast.Name(id="sorted", ctx=ast.Load()),
                    args=[ast.Name(id="_missing", ctx=ast.Load())],
                    keywords=[],
                ),
                conversion=-1,
                format_spec=None,
            ),
        ],
    )
    body.append(
        ast.If(
            test=ast.Name(id="_missing", ctx=ast.Load()),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="InputValidationError", ctx=ast.Load()),
                        args=[missing_message],
                        keywords=[],
                    ),
                    cause=None,
                ),
            ],
            orelse=[],
        ),
    )

    allowed_set = ast.Set(elts=[ast.Constant(value=name) for name in allowed_inputs])
    body.append(
        ast.Assign(
            targets=[ast.Name(id="_extra", ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id="_scope_keys", ctx=ast.Load()),
                op=ast.Sub(),
                right=allowed_set,
            ),
        ),
    )

    extra_message = ast.JoinedStr(
        values=[
            ast.Constant(value="Unexpected inputs provided: "),
            ast.FormattedValue(
                value=ast.Call(
                    func=ast.Name(id="sorted", ctx=ast.Load()),
                    args=[ast.Name(id="_extra", ctx=ast.Load())],
                    keywords=[],
                ),
                conversion=-1,
                format_spec=None,
            ),
        ],
    )
    body.append(
        ast.If(
            test=ast.Name(id="_extra", ctx=ast.Load()),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="InputValidationError", ctx=ast.Load()),
                        args=[extra_message],
                        keywords=[],
                    ),
                    cause=None,
                ),
            ],
            orelse=[],
        ),
    )

    for name in required_inputs:
        body.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Subscript(
                    value=ast.Name(id="scope", ctx=ast.Load()),
                    slice=ast.Constant(value=name),
                    ctx=ast.Load(),
                ),
            ),
        )

    for expr_name in evaluation_order:
        expr_ast = _builder(expr_name).build(expressions[expr_name])
        body.append(
            ast.Assign(
                targets=[ast.Name(id=expr_name, ctx=ast.Store())],
                value=expr_ast,
            ),
        )

    body.append(ast.Return(value=ast.Name(id=target, ctx=ast.Load())))

    func_def = ast.FunctionDef(
        name="_compiled",
        args=ast.arguments(
            posonlyargs=[],
            args=[scope_arg],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )

    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)
    return module


def compile_to_callable(
    *,
    expressions: Mapping[str, Mapping[str, Any]],
    inputs: Iterable[str],
    target: str,
) -> CompilationResult:
    if not isinstance(target, str):
        raise ExpressionError("Target identifier must be a string")
    target = ensure_identifier(target, expression=None)

    normalised_inputs = _normalise_inputs(inputs)
    input_set = set(normalised_inputs)

    validated_exprs = _validate_expressions(expressions, input_set)

    if target not in validated_exprs:
        raise MissingTargetError(f"Expression {target!r} missing from expressions")

    dependency_map: dict[str, set[str]] = {}
    for name, node in validated_exprs.items():
        collector = SymbolDependencyCollector(expression_name=name)
        deps = collector.collect(node)
        unknown = deps - input_set - set(validated_exprs)
        if unknown:
            identifier = sorted(unknown)[0]
            raise UnknownIdentifierError(
                f"Expression {name!r} references unknown identifier {identifier!r}",
                expression=name,
                identifier=identifier,
            )
        dependency_map[name] = deps

    closure = _dependency_closure(target, dependency_map, set(validated_exprs))
    required_inputs = sorted(
        {dep for expr in closure for dep in dependency_map[expr] if dep in input_set},
    )

    sorter = TopologicalSorter()
    for expr in closure:
        deps = [dep for dep in dependency_map[expr] if dep in closure]
        sorter.add(expr, *deps)

    try:
        order = tuple(sorter.static_order())
    except CycleError as exc:
        cycle = tuple(exc.args[1]) if len(exc.args) > 1 else tuple()
        raise CircularDependencyError("Dependency cycle detected", cycle=cycle)

    module_ast = _build_function_ast(
        evaluation_order=order,
        expressions=validated_exprs,
        required_inputs=tuple(required_inputs),
        allowed_inputs=normalised_inputs,
        target=target,
    )

    compiled = compile(module_ast, filename="<mathjs>", mode="exec")
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "InputValidationError": InputValidationError,
        "Mapping": AbcMapping,
        "isinstance": isinstance,
        "set": set,
        "sorted": sorted,
    }
    safe_globals.update(HELPER_FUNCTIONS)

    namespace: dict[str, Any] = {}
    exec(compiled, safe_globals, namespace)
    function = namespace["_compiled"]

    return CompilationResult(
        function=function,
        required_inputs=tuple(required_inputs),
        evaluation_order=order,
        module_ast=module_ast,
    )


__all__ = ["CompilationResult", "compile_to_callable"]
