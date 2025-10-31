"""Helper functions exposed to generated evaluators."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import reduce

import numpy as np

HELPER_NAME_MAP = {
    "min": "_mj_min",
    "max": "_mj_max",
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
        return value.item()
    return value


def _elementwise_reduce(func, values: Iterable[object]) -> object:
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
    return min(values)


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
    return max(values)


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
        result = result + item  # type: ignore[operator]
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


HELPER_FUNCTIONS = {
    "_mj_min": _mj_min,
    "_mj_max": _mj_max,
    "_mj_sum": _mj_sum,
    "_mj_ifnull": _mj_ifnull,
}

__all__ = [
    "HELPER_FUNCTIONS",
    "HELPER_NAME_MAP",
    "_mj_ifnull",
    "_mj_max",
    "_mj_min",
    "_mj_sum",
]
