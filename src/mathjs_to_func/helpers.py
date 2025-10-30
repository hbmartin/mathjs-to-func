"""Helper functions exposed to generated evaluators."""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

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


def _coerce(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


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
    values = [_coerce(v) for v in _expand_args(args)]
    if not values:
        raise ValueError("min requires at least one argument")
    if any(isinstance(v, np.ndarray) for v in values):
        return _elementwise_reduce(np.minimum, values)
    return min(values)


def _mj_max(*args: object) -> object:
    values = [_coerce(v) for v in _expand_args(args)]
    if not values:
        raise ValueError("max requires at least one argument")
    if any(isinstance(v, np.ndarray) for v in values):
        return _elementwise_reduce(np.maximum, values)
    return max(values)


def _mj_sum(*args: object) -> object:
    values = [_coerce(v) for v in _expand_args(args)]
    if not values:
        raise ValueError("sum requires at least one argument")

    if all(isinstance(v, (int, float, np.number)) for v in values):
        return sum(values)  # type: ignore[arg-type]

    if any(isinstance(v, np.ndarray) for v in values):
        stacked = [np.asarray(v) for v in values]
        result = reduce(np.add, stacked)
        return _maybe_scalar(result)

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
