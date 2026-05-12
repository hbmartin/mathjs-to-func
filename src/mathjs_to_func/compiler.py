"""Compile math.js JSON expressions into callable Python functions."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any

from .ast_builder import (
    MATHJS_BUILTIN_SYMBOLS,
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
    RuntimeEvaluationError,
    UnknownIdentifierError,
)
from .helpers import (
    HELPER_NAME_MAP,
    EvalConfig,
    coerce_eval_config,
    create_helper_functions,
)

RESERVED_INTERNAL_PREFIX = "__mj_"


@dataclass(frozen=True)
class CompilationResult:
    """Metadata for a compiled expression graph."""

    function: Any
    required_inputs: tuple[str, ...]
    evaluation_order: tuple[str, ...]
    targets: tuple[str, ...]
    returns_mapping: bool
    module_ast: ast.Module
    config: EvalConfig


def _ensure_user_identifier(name: str, *, expression: str | None) -> str:
    safe_name = ensure_identifier(name, expression=expression)
    if safe_name.startswith(RESERVED_INTERNAL_PREFIX):
        raise InvalidNodeError(
            f"Identifier {safe_name!r} uses reserved internal prefix "
            f"{RESERVED_INTERNAL_PREFIX!r}",
            expression=expression,
            node=None,
        )
    return safe_name


def _normalise_inputs(inputs: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in inputs:
        if not isinstance(raw, str):
            raise ExpressionError("Input identifiers must be strings")
        name = _ensure_user_identifier(raw, expression=None)
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
        name = _ensure_user_identifier(raw_name, expression=None)
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


def _normalise_targets(target: str | Sequence[str]) -> tuple[tuple[str, ...], bool]:
    if isinstance(target, str):
        return (_ensure_user_identifier(target, expression=None),), False
    if isinstance(target, (bytes, bytearray)) or not isinstance(target, Sequence):
        raise ExpressionError("Target must be a string or sequence of strings")

    seen: set[str] = set()
    targets: list[str] = []
    for raw in target:
        if not isinstance(raw, str):
            raise ExpressionError("Target identifiers must be strings")
        name = _ensure_user_identifier(raw, expression=None)
        if name in seen:
            raise ExpressionError(f"Duplicate target identifier: {name}")
        seen.add(name)
        targets.append(name)
    if not targets:
        raise ExpressionError("At least one target identifier is required")
    return tuple(targets), True


def _dependency_closure(
    targets: Sequence[str],
    dependency_map: Mapping[str, set[str]],
    expression_names: set[str],
) -> set[str]:
    needed: set[str] = set()
    stack = list(targets)
    while stack:
        current = stack.pop()
        if current in needed:
            continue
        needed.add(current)
        stack.extend(dep for dep in dependency_map[current] if dep in expression_names)
    return needed


def _build_function_ast(  # noqa: PLR0913
    *,
    evaluation_order: tuple[str, ...],
    expressions: Mapping[str, Mapping[str, Any]],
    required_inputs: tuple[str, ...],
    allowed_inputs: tuple[str, ...],
    targets: tuple[str, ...],
    returns_mapping: bool,
) -> ast.Module:
    scope_name = "__mj_scope"
    scope_keys_name = "__mj_scope_keys"
    missing_name = "__mj_missing"
    extra_name = "__mj_extra"
    error_name = "__mj_error"
    builder_cache: dict[str, MathJsAstBuilder] = {}

    def _builder(expr_name: str) -> MathJsAstBuilder:
        if expr_name not in builder_cache:
            builder_cache[expr_name] = MathJsAstBuilder(
                expression_name=expr_name,
                helper_names=HELPER_NAME_MAP,
                local_names=set(allowed_inputs) | set(expressions),
            )
        return builder_cache[expr_name]

    def _runtime_wrapped_assign(expr_name: str) -> ast.Try:
        return ast.Try(
            body=[
                ast.Assign(
                    targets=[ast.Name(id=expr_name, ctx=ast.Store())],
                    value=_builder(expr_name).build(expressions[expr_name]),
                ),
            ],
            handlers=[
                ast.ExceptHandler(
                    type=ast.Name(id="__mj_Exception", ctx=ast.Load()),
                    name=error_name,
                    body=[
                        ast.Raise(
                            exc=ast.Call(
                                func=ast.Name(
                                    id="__mj_RuntimeEvaluationError",
                                    ctx=ast.Load(),
                                ),
                                args=[
                                    ast.Constant(
                                        value=(
                                            f"Error evaluating expression "
                                            f"{expr_name!r}"
                                        ),
                                    ),
                                ],
                                keywords=[
                                    ast.keyword(
                                        arg="expression",
                                        value=ast.Constant(value=expr_name),
                                    ),
                                ],
                            ),
                            cause=ast.Name(id=error_name, ctx=ast.Load()),
                        ),
                    ],
                ),
            ],
            orelse=[],
            finalbody=[],
        )

    scope_arg = ast.arg(arg=scope_name)
    body: list[ast.stmt] = []

    isinstance_call = ast.Call(
        func=ast.Name(id="__mj_isinstance", ctx=ast.Load()),
        args=[
            ast.Name(id=scope_name, ctx=ast.Load()),
            ast.Name(id="__mj_Mapping", ctx=ast.Load()),
        ],
        keywords=[],
    )
    body.append(
        ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=isinstance_call),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="__mj_InputValidationError", ctx=ast.Load()),
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
        func=ast.Name(id="__mj_set", ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=scope_name, ctx=ast.Load()),
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
            targets=[ast.Name(id=scope_keys_name, ctx=ast.Store())],
            value=scope_key_call,
        ),
    )
    body.append(
        ast.Assign(
            targets=[ast.Name(id=missing_name, ctx=ast.Store())],
            value=ast.BinOp(
                left=required_set,
                op=ast.Sub(),
                right=ast.Name(id=scope_keys_name, ctx=ast.Load()),
            ),
        ),
    )

    missing_message = ast.JoinedStr(
        values=[
            ast.Constant(value="Missing required inputs: "),
            ast.FormattedValue(
                value=ast.Call(
                    func=ast.Name(id="__mj_sorted", ctx=ast.Load()),
                    args=[ast.Name(id=missing_name, ctx=ast.Load())],
                    keywords=[],
                ),
                conversion=-1,
                format_spec=None,
            ),
        ],
    )
    body.append(
        ast.If(
            test=ast.Name(id=missing_name, ctx=ast.Load()),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="__mj_InputValidationError", ctx=ast.Load()),
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
            targets=[ast.Name(id=extra_name, ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Name(id=scope_keys_name, ctx=ast.Load()),
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
                    func=ast.Name(id="__mj_sorted", ctx=ast.Load()),
                    args=[ast.Name(id=extra_name, ctx=ast.Load())],
                    keywords=[],
                ),
                conversion=-1,
                format_spec=None,
            ),
        ],
    )
    body.append(
        ast.If(
            test=ast.Name(id=extra_name, ctx=ast.Load()),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="__mj_InputValidationError", ctx=ast.Load()),
                        args=[extra_message],
                        keywords=[],
                    ),
                    cause=None,
                ),
            ],
            orelse=[],
        ),
    )

    body.extend(
        ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Subscript(
                value=ast.Name(id=scope_name, ctx=ast.Load()),
                slice=ast.Constant(value=name),
                ctx=ast.Load(),
            ),
        )
        for name in required_inputs
    )

    body.extend(_runtime_wrapped_assign(expr_name) for expr_name in evaluation_order)

    if returns_mapping:
        return_value: ast.expr = ast.Dict(
            keys=[ast.Constant(value=target) for target in targets],
            values=[ast.Name(id=target, ctx=ast.Load()) for target in targets],
        )
    else:
        return_value = ast.Name(id=targets[0], ctx=ast.Load())

    body.append(ast.Return(value=return_value))

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
        returns=None,
        type_comment=None,
        type_params=[],
    )

    module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(module)
    return module


def compile_to_callable(
    *,
    expressions: Mapping[str, Mapping[str, Any]],
    inputs: Iterable[str],
    target: str | Sequence[str],
    config: EvalConfig | Mapping[str, object] | None = None,
) -> CompilationResult:
    """Compile expressions into an executable callable with dependency metadata."""
    normalized_config = coerce_eval_config(config)
    targets, returns_mapping = _normalise_targets(target)

    normalised_inputs = _normalise_inputs(inputs)
    input_set = set(normalised_inputs)

    validated_exprs = _validate_expressions(expressions, input_set)

    for target_name in targets:
        if target_name not in validated_exprs:
            raise MissingTargetError(
                f"Expression {target_name!r} missing from expressions",
            )

    dependency_map: dict[str, set[str]] = {}
    for name, node in validated_exprs.items():
        collector = SymbolDependencyCollector(expression_name=name)
        deps = collector.collect(node)
        unknown = deps - input_set - set(validated_exprs) - set(MATHJS_BUILTIN_SYMBOLS)
        if unknown:
            identifier = sorted(unknown)[0]
            raise UnknownIdentifierError(
                f"Expression {name!r} references unknown identifier {identifier!r}",
                expression=name,
                identifier=identifier,
            )
        dependency_map[name] = deps

    closure = _dependency_closure(targets, dependency_map, set(validated_exprs))
    required_inputs = sorted(
        {dep for expr in closure for dep in dependency_map[expr] if dep in input_set},
    )

    sorter = TopologicalSorter()
    for expr in sorted(closure):
        deps = sorted(dep for dep in dependency_map[expr] if dep in closure)
        sorter.add(expr, *deps)

    try:
        order = tuple(sorter.static_order())
    except CycleError as exc:
        cycle = tuple(exc.args[1]) if len(exc.args) > 1 else ()
        raise CircularDependencyError(
            "Dependency cycle detected",
            cycle=cycle,
        ) from exc

    module_ast = _build_function_ast(
        evaluation_order=order,
        expressions=validated_exprs,
        required_inputs=tuple(required_inputs),
        allowed_inputs=normalised_inputs,
        targets=targets,
        returns_mapping=returns_mapping,
    )

    compiled = compile(module_ast, filename="<mathjs>", mode="exec")
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "__mj_Exception": Exception,
        "__mj_InputValidationError": InputValidationError,
        "__mj_Mapping": AbcMapping,
        "__mj_RuntimeEvaluationError": RuntimeEvaluationError,
        "__mj_isinstance": isinstance,
        "__mj_set": set,
        "__mj_sorted": sorted,
    }
    safe_globals.update(create_helper_functions(normalized_config))

    namespace: dict[str, Any] = {}
    exec(compiled, safe_globals, namespace)
    function = namespace["_compiled"]

    return CompilationResult(
        function=function,
        required_inputs=tuple(required_inputs),
        evaluation_order=order,
        targets=targets,
        returns_mapping=returns_mapping,
        module_ast=module_ast,
        config=normalized_config,
    )


__all__ = ["CompilationResult", "compile_to_callable"]
