"""Helper functions exposed to generated evaluators."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

HELPER_NAME_MAP = {
    "abs": "_mj_abs",
    "ceil": "_mj_ceil",
    "exp": "_mj_exp",
    "floor": "_mj_floor",
    "log": "_mj_log",
    "mean": "_mj_mean",
    "median": "_mj_median",
    "min": "_mj_min",
    "max": "_mj_max",
    "round": "_mj_round",
    "sign": "_mj_sign",
    "sqrt": "_mj_sqrt",
    "sum": "_mj_sum",
    "ifnull": "_mj_ifnull",
}


def _expand_args(args: Sequence[object]) -> list[object]:
    """Unpack math.js variadic conventions."""
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        if isinstance(args[0], np.ndarray):
            return [args[0]]
        return list(args[0])
    return list(args)


def _maybe_scalar(value: object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return cast("Any", value).item()
    return value


def _is_array_like(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _maybe_bool(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _unary_numpy(func: Callable[..., Any], value: object) -> object:
    return _maybe_scalar(func(value))


def _binary_numpy(
    func: Callable[..., Any],
    left: object,
    right: object,
) -> object:
    return _maybe_scalar(func(left, right))


def _elementwise_reduce(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    values: Iterable[object],
) -> object:
    iterator = iter(values)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Function requires at least one argument") from exc
    result = np.asarray(first)
    for item in iterator:
        result = func(result, np.asarray(item))
    return _maybe_scalar(result)


def _mj_min(*args: object) -> object:
    raw_values = _expand_args(args)
    values: list[object] = []
    has_array = False
    for item in raw_values:
        if isinstance(item, np.ndarray):
            values.append(item)
            has_array = True
        elif isinstance(item, (list, tuple)):
            arr = np.asarray(item)
            values.append(arr)
            has_array = True
        else:
            values.append(item)
    if not values:
        raise ValueError("min requires at least one argument")
    if has_array:
        arrays = [np.asarray(v) for v in values]
        if len(arrays) == 1:
            return _maybe_scalar(np.min(arrays[0]))
        return _elementwise_reduce(np.minimum, arrays)
    return min(cast("Any", values))


def _mj_max(*args: object) -> object:
    raw_values = _expand_args(args)
    values: list[object] = []
    has_array = False
    for item in raw_values:
        if isinstance(item, np.ndarray):
            values.append(item)
            has_array = True
        elif isinstance(item, (list, tuple)):
            arr = np.asarray(item)
            values.append(arr)
            has_array = True
        else:
            values.append(item)
    if not values:
        raise ValueError("max requires at least one argument")
    if has_array:
        arrays = [np.asarray(v) for v in values]
        if len(arrays) == 1:
            return _maybe_scalar(np.max(arrays[0]))
        return _elementwise_reduce(np.maximum, arrays)
    return max(cast("Any", values))


def _mj_sum(*args: object) -> object:
    raw_values = _expand_args(args)
    values: list[object] = []
    has_array = False
    for item in raw_values:
        if isinstance(item, np.ndarray):
            values.append(item)
            has_array = True
        elif isinstance(item, (list, tuple)):
            arr = np.asarray(item)
            values.append(arr)
            has_array = True
        else:
            values.append(item)
    if not values:
        raise ValueError("sum requires at least one argument")

    if has_array:
        arrays = [np.asarray(v) for v in values]
        if len(arrays) == 1:
            return _maybe_scalar(np.sum(arrays[0]))
        result = reduce(np.add, arrays)
        return _maybe_scalar(result)

    if all(isinstance(v, (int, float, np.number, bool, np.bool_)) for v in values):
        return sum(values)  # type: ignore[arg-type]

    result = values[0]
    for item in values[1:]:
        result = cast("Any", result) + item
    return result


def _mj_ifnull(value: object, fallback: object) -> object:
    if value is None:
        return fallback
    if isinstance(value, float) and np.isnan(value):
        return fallback
    if isinstance(value, np.number):
        return fallback if np.isnan(value) else value
    if isinstance(value, np.ndarray):
        mask = np.isnan(value)
        if not mask.any():
            return value
        return np.where(mask, np.asarray(fallback), value)
    if isinstance(value, int):
        return value
    return value


def _mj_abs(value: object) -> object:
    return _unary_numpy(np.abs, value)


def _mj_sqrt(value: object) -> object:
    return _unary_numpy(np.sqrt, value)


def _mj_log(value: object) -> object:
    return _unary_numpy(np.log, value)


def _mj_exp(value: object) -> object:
    return _unary_numpy(np.exp, value)


def _mj_round(value: object, decimals: object = 0) -> object:
    return _maybe_scalar(
        np.round(cast("Any", value), int(cast("Any", decimals))),
    )


def _mj_floor(value: object) -> object:
    return _unary_numpy(np.floor, value)


def _mj_ceil(value: object) -> object:
    return _unary_numpy(np.ceil, value)


def _mj_sign(value: object) -> object:
    return _unary_numpy(np.sign, value)


def _mj_mean(*args: object) -> object:
    if len(args) == 1 and _is_array_like(args[0]):
        array_arg = np.asarray(args[0])
        if array_arg.size == 0:
            raise ValueError("mean requires at least one argument")
        return _maybe_scalar(np.mean(array_arg))
    values = _expand_args(args)
    if not values:
        raise ValueError("mean requires at least one argument")
    return _maybe_scalar(np.mean(np.asarray(values), axis=0))


def _mj_median(*args: object) -> object:
    if len(args) == 1 and _is_array_like(args[0]):
        array_arg = np.asarray(args[0])
        if array_arg.size == 0:
            raise ValueError("median requires at least one argument")
        return _maybe_scalar(np.median(array_arg))
    values = _expand_args(args)
    if not values:
        raise ValueError("median requires at least one argument")
    return _maybe_scalar(np.median(np.asarray(values), axis=0))


def _mj_larger(left: object, right: object) -> object:
    return _binary_numpy(np.greater, left, right)


def _mj_larger_eq(left: object, right: object) -> object:
    return _binary_numpy(np.greater_equal, left, right)


def _mj_smaller(left: object, right: object) -> object:
    return _binary_numpy(np.less, left, right)


def _mj_smaller_eq(left: object, right: object) -> object:
    return _binary_numpy(np.less_equal, left, right)


def _mj_equal(left: object, right: object) -> object:
    return _binary_numpy(np.equal, left, right)


def _mj_unequal(left: object, right: object) -> object:
    return _binary_numpy(np.not_equal, left, right)


def _mj_and(left: object, right: object) -> object:
    return _binary_numpy(np.logical_and, left, right)


def _mj_or(left: object, right: object) -> object:
    return _binary_numpy(np.logical_or, left, right)


def _mj_lazy_and(left: object, right: Callable[[], object]) -> object:
    if _is_array_like(left):
        return _mj_and(left, right())
    if not left:
        try:
            right_value = right()
        except ArithmeticError:
            return False
        if _is_array_like(right_value):
            return _mj_and(False if left is None else left, right_value)
        return False
    return _maybe_bool(_mj_and(left, right()))


def _mj_lazy_or(left: object, right: Callable[[], object]) -> object:
    if _is_array_like(left):
        return _mj_or(left, right())
    if left:
        try:
            right_value = right()
        except ArithmeticError:
            return True
        if _is_array_like(right_value):
            return _mj_or(left, right_value)
        return True
    return _maybe_bool(_mj_or(left, right()))


def _mj_xor(left: object, right: object) -> object:
    return _binary_numpy(np.logical_xor, left, right)


def _mj_not(value: object) -> object:
    return _unary_numpy(np.logical_not, value)


def _mj_where(condition: object, true_value: object, false_value: object) -> object:
    return _maybe_scalar(
        np.where(
            np.asarray(condition),
            np.asarray(true_value),
            np.asarray(false_value),
        ),
    )


def _mj_lazy_where(
    condition: object,
    true_value: Callable[[], object],
    false_value: Callable[[], object],
) -> object:
    if _is_array_like(condition):
        return _mj_where(condition, true_value(), false_value())
    if condition:
        return true_value()
    return false_value()


HELPER_FUNCTIONS = {
    "_mj_abs": _mj_abs,
    "_mj_and": _mj_and,
    "_mj_ceil": _mj_ceil,
    "_mj_equal": _mj_equal,
    "_mj_exp": _mj_exp,
    "_mj_floor": _mj_floor,
    "_mj_min": _mj_min,
    "_mj_max": _mj_max,
    "_mj_larger": _mj_larger,
    "_mj_larger_eq": _mj_larger_eq,
    "_mj_lazy_and": _mj_lazy_and,
    "_mj_lazy_or": _mj_lazy_or,
    "_mj_lazy_where": _mj_lazy_where,
    "_mj_log": _mj_log,
    "_mj_mean": _mj_mean,
    "_mj_median": _mj_median,
    "_mj_not": _mj_not,
    "_mj_or": _mj_or,
    "_mj_round": _mj_round,
    "_mj_sign": _mj_sign,
    "_mj_smaller": _mj_smaller,
    "_mj_smaller_eq": _mj_smaller_eq,
    "_mj_sqrt": _mj_sqrt,
    "_mj_sum": _mj_sum,
    "_mj_ifnull": _mj_ifnull,
    "_mj_unequal": _mj_unequal,
    "_mj_where": _mj_where,
    "_mj_xor": _mj_xor,
}

__all__ = [
    "HELPER_FUNCTIONS",
    "HELPER_NAME_MAP",
    "_mj_abs",
    "_mj_and",
    "_mj_ceil",
    "_mj_equal",
    "_mj_exp",
    "_mj_floor",
    "_mj_ifnull",
    "_mj_larger",
    "_mj_larger_eq",
    "_mj_lazy_and",
    "_mj_lazy_or",
    "_mj_lazy_where",
    "_mj_log",
    "_mj_max",
    "_mj_mean",
    "_mj_median",
    "_mj_min",
    "_mj_not",
    "_mj_or",
    "_mj_round",
    "_mj_sign",
    "_mj_smaller",
    "_mj_smaller_eq",
    "_mj_sqrt",
    "_mj_sum",
    "_mj_unequal",
    "_mj_where",
    "_mj_xor",
]
