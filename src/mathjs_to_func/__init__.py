"""Public API for mathjs-to-func."""

from __future__ import annotations

import ast
import math
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import FunctionType
from typing import Any, Literal, Protocol, cast, overload

from .compiler import CompilationResult, compile_to_callable, normalise_inputs
from .errors import (
    CircularDependencyError,
    ExpressionError,
    InputValidationError,
    InvalidNodeError,
    MissingTargetError,
    RuntimeEvaluationError,
    UnknownIdentifierError,
)
from .helpers import EvalConfig, coerce_eval_config, source_preamble
from .introspection import (
    inputs_referenced_per_target,
    to_dot,
    to_mermaid,
    to_string,
    to_tex,
)

__all__ = [
    "CircularDependencyError",
    "CompiledEvaluator",
    "CompiledEvaluatorWithSource",
    "EvalConfig",
    "ExpressionError",
    "InputValidationError",
    "InvalidNodeError",
    "MissingTargetError",
    "RuntimeEvaluationError",
    "UnknownIdentifierError",
    "build_evaluator",
    "inputs_referenced_per_target",
    "to_dot",
    "to_mermaid",
    "to_string",
    "to_tex",
]


class CompiledEvaluator(Protocol):
    """Callable returned by :func:`build_evaluator` with metadata attributes."""

    __code__: Any
    __globals__: dict[str, Any]
    __mathjs_config__: EvalConfig
    __mathjs_required_inputs__: tuple[str, ...]
    __mathjs_evaluation_order__: tuple[str, ...]
    __mathjs_inputs_referenced_per_target__: dict[str, tuple[str, ...]]
    __mathjs_targets__: tuple[str, ...]

    def __call__(self, scope: Mapping[str, Any]) -> Any:  # noqa: ANN401
        """Evaluate the compiled expression for a scope mapping."""
        ...


class CompiledEvaluatorWithSource(CompiledEvaluator, Protocol):
    """Compiled evaluator returned when source output is requested."""

    __mathjs_source__: str


def _extract_payload(
    expressions: Mapping[str, Any] | None,
    inputs: Iterable[str] | None,
    target: str | Sequence[str] | None,
    payload: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Iterable[str], str | Sequence[str]]:
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


def _cache_target(target: str | Sequence[str]) -> str | list[str]:
    if isinstance(target, str):
        return target
    try:
        return list(target)
    except TypeError:
        return cast("Any", target)


def _structural_cache_value(value: object) -> Hashable:
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        items = [
            (
                _structural_cache_value(key),
                _structural_cache_value(item),
            )
            for key, item in value.items()
        ]
        return (
            "dict",
            tuple(sorted(items, key=repr)),
        )
    if isinstance(value, Mapping):
        items = [
            (
                _structural_cache_value(key),
                _structural_cache_value(item),
            )
            for key, item in value.items()
        ]
        return (
            "mapping",
            tuple(sorted(items, key=repr)),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_structural_cache_value(item) for item in value))
    if isinstance(value, set):
        return (
            "set",
            tuple(sorted((_structural_cache_value(item) for item in value), key=repr)),
        )
    if isinstance(value, Hashable):
        return value
    return ("repr", type(value).__qualname__, repr(value))


def _canonical_cache_key(
    *,
    expressions: Mapping[str, Any],
    inputs: Iterable[str],
    target: str | Sequence[str],
    config: EvalConfig,
) -> Hashable:
    return _structural_cache_value(
        {
            "config": {
                "abs_tol": config.abs_tol,
                "comparison": config.comparison,
                "rel_tol": config.rel_tol,
                "result_dtype": config.result_dtype,
            },
            "expressions": expressions,
            "inputs": sorted(inputs),
            "target": _cache_target(target),
        },
    )


_COMPILE_CACHES: dict[int | None, OrderedDict[Hashable, CompilationResult]] = {}


def _cached_compile(  # noqa: PLR0913
    canonical_payload: Hashable,
    *,
    expressions: Mapping[str, Any],
    inputs: Iterable[str],
    target: str | Sequence[str],
    config: EvalConfig,
    maxsize: int | None,
) -> CompilationResult:
    cache = _COMPILE_CACHES.setdefault(maxsize, OrderedDict())
    if canonical_payload in cache:
        result = cache.pop(canonical_payload)
        cache[canonical_payload] = result
        return result

    result = compile_to_callable(
        expressions=expressions,
        inputs=inputs,
        target=target,
        config=config,
    )
    cache[canonical_payload] = result
    if maxsize is not None and maxsize > 0:
        while len(cache) > maxsize:
            cache.popitem(last=False)
    return result


def _clone_compilation_result(result: CompilationResult) -> CompilationResult:
    function = result.function
    cloned = FunctionType(
        function.__code__,
        dict(function.__globals__),
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = function.__kwdefaults__
    cloned.__annotations__ = dict(function.__annotations__)
    cloned.__qualname__ = function.__qualname__
    cloned.__module__ = function.__module__
    return CompilationResult(
        function=cloned,
        required_inputs=result.required_inputs,
        evaluation_order=result.evaluation_order,
        targets=result.targets,
        returns_mapping=result.returns_mapping,
        module_ast=result.module_ast,
        config=result.config,
        inputs_referenced_per_target=dict(result.inputs_referenced_per_target),
    )


def _attach_metadata(
    result: CompilationResult,
    *,
    include_source: bool,
) -> CompiledEvaluator | CompiledEvaluatorWithSource:
    func = result.function
    func.__mathjs_config__ = result.config
    func.__mathjs_required_inputs__ = result.required_inputs
    func.__mathjs_evaluation_order__ = result.evaluation_order
    func.__mathjs_inputs_referenced_per_target__ = result.inputs_referenced_per_target
    func.__mathjs_targets__ = result.targets

    if include_source:
        source = f"{source_preamble(result.config)}\n{ast.unparse(result.module_ast)}"
        func.__mathjs_source__ = source
        return cast("CompiledEvaluatorWithSource", func)

    return cast("CompiledEvaluator", func)


@overload
def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | Sequence[str] | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    config: EvalConfig | Mapping[str, object] | None = None,
    compile_cache: bool = False,
    compile_cache_maxsize: int | None = 128,
    include_source: Literal[True],
) -> CompiledEvaluatorWithSource: ...


@overload
def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | Sequence[str] | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    config: EvalConfig | Mapping[str, object] | None = None,
    compile_cache: bool = False,
    compile_cache_maxsize: int | None = 128,
    include_source: Literal[False] = False,
) -> CompiledEvaluator: ...


@overload
def build_evaluator(
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | Sequence[str] | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    config: EvalConfig | Mapping[str, object] | None = None,
    compile_cache: bool = False,
    compile_cache_maxsize: int | None = 128,
    include_source: bool,
) -> CompiledEvaluator | CompiledEvaluatorWithSource: ...


def build_evaluator(  # noqa: PLR0913
    expressions: Mapping[str, Any] | None = None,
    inputs: Iterable[str] | None = None,
    target: str | Sequence[str] | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
    config: EvalConfig | Mapping[str, object] | None = None,
    compile_cache: bool = False,
    compile_cache_maxsize: int | None = 128,
    include_source: bool = False,
) -> CompiledEvaluator | CompiledEvaluatorWithSource:
    """Compile math.js expressions into a reusable callable.

    Parameters may be supplied directly or via ``payload`` containing the keys
    ``expressions``, ``inputs``, and ``target``. The returned function expects a
    single mapping argument containing the input values and returns the computed
    target value. If ``target`` is a sequence, the callable returns a mapping
    keyed by target name.
    """
    expressions, inputs, target = _extract_payload(expressions, inputs, target, payload)
    normalized_config = coerce_eval_config(config)

    if compile_cache:
        if compile_cache_maxsize is not None and compile_cache_maxsize < 1:
            raise ValueError("compile_cache_maxsize must be a positive integer or None")
        input_list = normalise_inputs(inputs)
        canonical = _canonical_cache_key(
            expressions=expressions,
            inputs=input_list,
            target=target,
            config=normalized_config,
        )
        result = _clone_compilation_result(
            _cached_compile(
                canonical,
                expressions=expressions,
                inputs=input_list,
                target=target,
                config=normalized_config,
                maxsize=compile_cache_maxsize,
            ),
        )
    else:
        result = compile_to_callable(
            expressions=expressions,
            inputs=inputs,
            target=target,
            config=normalized_config,
        )

    return _attach_metadata(result, include_source=include_source)
