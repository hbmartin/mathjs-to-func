"""Public API for mathjs-to-func."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, overload

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from .compiler import CompilationResult, compile_to_callable
from .errors import (
    CircularDependencyError,
    ExpressionError,
    InputValidationError,
    InvalidNodeError,
    MissingTargetError,
    UnknownIdentifierError,
)

__all__ = [
    "CircularDependencyError",
    "CompiledEvaluator",
    "CompiledEvaluatorWithSource",
    "ExpressionError",
    "InputValidationError",
    "InvalidNodeError",
    "MissingTargetError",
    "UnknownIdentifierError",
    "build_evaluator",
]


class CompiledEvaluator(Protocol):
    """Callable returned by :func:`build_evaluator` with metadata attributes."""

    __mathjs_required_inputs__: tuple[str, ...]
    __mathjs_evaluation_order__: tuple[str, ...]

    def __call__(self, scope: Mapping[str, Any]) -> Any:  # noqa: ANN401
        """Evaluate the compiled expression for a scope mapping."""
        ...


class CompiledEvaluatorWithSource(CompiledEvaluator, Protocol):
    """Compiled evaluator returned when source output is requested."""

    __mathjs_source__: str


def _extract_payload(
    expressions: Mapping[str, Any] | None,
    inputs: Iterable[str] | None,
    target: str | None,
    payload: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Iterable[str], str]:
    if payload is not None:
        if expressions is not None or inputs is not None or target is not None:
            raise ExpressionError(
                "payload cannot be combined with direct arguments",
                expression=None,
            )
        try:
            expressions = payload["expressions"]
            inputs = payload["inputs"]
            target = payload["target"]
        except KeyError as exc:
            missing = exc.args[0]
            raise ExpressionError(
                f"Payload missing required key: {missing}",
                expression=None,
            ) from exc
    if expressions is None or inputs is None or target is None:
        raise ExpressionError("Expressions, inputs, and target are required")
    return expressions, inputs, target


@overload
def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    include_source: Literal[True],
) -> CompiledEvaluatorWithSource: ...


@overload
def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    include_source: Literal[False] = False,
) -> CompiledEvaluator: ...


@overload
def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    include_source: bool,
) -> CompiledEvaluator | CompiledEvaluatorWithSource: ...


def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    include_source: bool = False,
) -> CompiledEvaluator | CompiledEvaluatorWithSource:
    """Compile math.js expressions into a reusable callable.

    Parameters may be supplied directly or via ``payload`` containing the keys
    ``expressions``, ``inputs``, and ``target``. The returned function expects a
    single mapping argument containing the input values and returns the computed
    target value.
    """
    expressions, inputs, target = _extract_payload(expressions, inputs, target, payload)

    result: CompilationResult = compile_to_callable(
        expressions=expressions,
        inputs=inputs,
        target=target,
    )

    func = result.function
    func.__mathjs_required_inputs__ = result.required_inputs
    func.__mathjs_evaluation_order__ = result.evaluation_order

    if include_source:
        source = ast.unparse(result.module_ast)
        func.__mathjs_source__ = source
        return cast("CompiledEvaluatorWithSource", func)

    return cast("CompiledEvaluator", func)
