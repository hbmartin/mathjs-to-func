"""Helper functions exposed to generated evaluators."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import reduce
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

HELPER_NAME_MAP = {
    "abs": "__mj_abs",
    "acos": "__mj_acos",
    "acosh": "__mj_acosh",
    "asin": "__mj_asin",
    "asinh": "__mj_asinh",
    "atan": "__mj_atan",
    "atan2": "__mj_atan2",
    "atanh": "__mj_atanh",
    "ceil": "__mj_ceil",
    "cbrt": "__mj_cbrt",
    "clamp": "__mj_clamp",
    "combinations": "__mj_combinations",
    "cos": "__mj_cos",
    "cosh": "__mj_cosh",
    "exp": "__mj_exp",
    "factorial": "__mj_factorial",
    "floor": "__mj_floor",
    "gcd": "__mj_gcd",
    "hypot": "__mj_hypot",
    "lcm": "__mj_lcm",
    "log": "__mj_log",
    "log1p": "__mj_log1p",
    "log2": "__mj_log2",
    "log10": "__mj_log10",
    "mean": "__mj_mean",
    "median": "__mj_median",
    "min": "__mj_min",
    "max": "__mj_max",
    "mode": "__mj_mode",
    "permutations": "__mj_permutations",
    "round": "__mj_round",
    "sign": "__mj_sign",
    "sin": "__mj_sin",
    "sinh": "__mj_sinh",
    "sqrt": "__mj_sqrt",
    "std": "__mj_std",
    "sum": "__mj_sum",
    "tan": "__mj_tan",
    "tanh": "__mj_tanh",
    "variance": "__mj_variance",
    "ifnull": "__mj_ifnull",
    "nullish": "__mj_ifnull",
}

MATHJS_REL_TOL = 1e-12
MATHJS_ABS_TOL = 1e-15


def _coerce_tolerance(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative finite number")
    try:
        tolerance = float(cast("Any", value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative finite number") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return tolerance


@dataclass(frozen=True)
class EvalConfig:
    """Runtime options used by compiled evaluators."""

    rel_tol: float = MATHJS_REL_TOL
    abs_tol: float = MATHJS_ABS_TOL
    epsilon: float | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalize and validate tolerance aliases."""
        rel_tol: object = self.rel_tol
        abs_tol: object = self.abs_tol
        if self.epsilon is not None:
            if self.rel_tol == MATHJS_REL_TOL:
                rel_tol = self.epsilon
            if self.abs_tol == MATHJS_ABS_TOL:
                abs_tol = self.epsilon

        object.__setattr__(
            self,
            "rel_tol",
            _coerce_tolerance(rel_tol, name="rel_tol"),
        )
        object.__setattr__(
            self,
            "abs_tol",
            _coerce_tolerance(abs_tol, name="abs_tol"),
        )
        object.__setattr__(self, "epsilon", None)


def coerce_eval_config(config: EvalConfig | Mapping[str, object] | None) -> EvalConfig:
    """Normalize a user-supplied evaluator config."""
    if config is None:
        return EvalConfig()
    if isinstance(config, EvalConfig):
        return config
    if isinstance(config, Mapping):
        allowed = {"rel_tol", "abs_tol", "epsilon"}
        unknown = set(config) - allowed
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ValueError(f"Unknown EvalConfig option(s): {names}")
        return EvalConfig(
            rel_tol=cast("float", config.get("rel_tol", MATHJS_REL_TOL)),
            abs_tol=cast("float", config.get("abs_tol", MATHJS_ABS_TOL)),
            epsilon=cast("float | None", config.get("epsilon")),
        )
    raise TypeError("config must be an EvalConfig, mapping, or None")


def _expand_args(args: Sequence[object]) -> list[object]:
    """Unpack math.js variadic conventions."""
    if len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
        if isinstance(args[0], np.ndarray):
            return [args[0]]
        return list(args[0])
    return list(args)


def _collect(args: Sequence[object]) -> tuple[list[object], bool]:
    values: list[object] = []
    has_array = False
    for item in _expand_args(args):
        if isinstance(item, np.ndarray):
            values.append(item)
            has_array = True
        elif isinstance(item, (list, tuple)):
            values.append(np.asarray(item))
            has_array = True
        else:
            values.append(item)
    return values, has_array


def _maybe_scalar(value: object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return cast("Any", value).item()
    if isinstance(value, np.generic):
        return cast("Any", value).item()
    return value


def _is_array_like(value: object) -> bool:
    return isinstance(value, (list, tuple, np.ndarray))


def _flatten_nested(value: object) -> Iterator[object]:
    if isinstance(value, np.ndarray):
        for item in value.flat:
            yield from _flatten_nested(cast("object", item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_nested(item)
    else:
        yield value


def _mode_key(value: object, *, nan_key: object) -> object:
    try:
        if bool(np.isnan(cast("Any", value))):
            return nan_key
    except (TypeError, ValueError):
        pass
    return value


def _maybe_bool(value: object) -> object:
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _as_integer(value: object, *, name: str) -> int:
    scalar = _maybe_scalar(value)
    if isinstance(scalar, bool):
        raise TypeError(f"{name} requires integer arguments")
    try:
        integer = int(cast("Any", scalar))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} requires integer arguments") from exc
    if scalar != integer:
        raise ValueError(f"{name} requires integer arguments")
    return integer


def _integer_array(value: object, *, name: str) -> np.ndarray:
    def as_integer(item: object) -> int:
        return _as_integer(item, name=name)

    return np.vectorize(as_integer, otypes=[int])(np.asarray(value, dtype=object))


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
    values, has_array = _collect(args)
    if not values:
        raise ValueError("min requires at least one argument")
    if has_array:
        arrays = [np.asarray(v) for v in values]
        if len(arrays) == 1:
            return _maybe_scalar(np.min(arrays[0]))
        return _elementwise_reduce(np.minimum, arrays)
    return min(cast("Any", values))


def _mj_max(*args: object) -> object:
    values, has_array = _collect(args)
    if not values:
        raise ValueError("max requires at least one argument")
    if has_array:
        arrays = [np.asarray(v) for v in values]
        if len(arrays) == 1:
            return _maybe_scalar(np.max(arrays[0]))
        return _elementwise_reduce(np.maximum, arrays)
    return max(cast("Any", values))


def _mj_sum(*args: object) -> object:
    values, has_array = _collect(args)
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


def _mj_lazy_ifnull(value: object, fallback: Callable[[], object]) -> object:
    if isinstance(value, np.ndarray):
        try:
            mask = np.isnan(value)
        except TypeError:
            return value
        if mask.any():
            return np.where(mask, np.asarray(fallback()), value)
        return value
    if value is None:
        return fallback()
    if isinstance(value, float) and np.isnan(value):
        return fallback()
    if isinstance(value, np.number):
        return fallback() if np.isnan(value) else value
    return value


def _mj_abs(value: object) -> object:
    return _unary_numpy(np.abs, value)


def _mj_sqrt(value: object) -> object:
    return _unary_numpy(np.sqrt, value)


def _mj_log(value: object, base: object | None = None) -> object:
    if base is None:
        return _unary_numpy(np.log, value)
    return _maybe_scalar(np.log(cast("Any", value)) / np.log(cast("Any", base)))


def _mj_log2(value: object) -> object:
    return _unary_numpy(np.log2, value)


def _mj_log10(value: object) -> object:
    return _unary_numpy(np.log10, value)


def _mj_log1p(value: object, base: object | None = None) -> object:
    if base is None:
        return _unary_numpy(np.log1p, value)
    return _maybe_scalar(np.log1p(cast("Any", value)) / np.log(cast("Any", base)))


def _mj_exp(value: object) -> object:
    return _unary_numpy(np.exp, value)


def _mj_sin(value: object) -> object:
    return _unary_numpy(np.sin, value)


def _mj_cos(value: object) -> object:
    return _unary_numpy(np.cos, value)


def _mj_tan(value: object) -> object:
    return _unary_numpy(np.tan, value)


def _mj_asin(value: object) -> object:
    return _unary_numpy(np.arcsin, value)


def _mj_acos(value: object) -> object:
    return _unary_numpy(np.arccos, value)


def _mj_atan(value: object) -> object:
    return _unary_numpy(np.arctan, value)


def _mj_atan2(left: object, right: object) -> object:
    return _binary_numpy(np.arctan2, left, right)


def _mj_sinh(value: object) -> object:
    return _unary_numpy(np.sinh, value)


def _mj_cosh(value: object) -> object:
    return _unary_numpy(np.cosh, value)


def _mj_tanh(value: object) -> object:
    return _unary_numpy(np.tanh, value)


def _mj_asinh(value: object) -> object:
    return _unary_numpy(np.arcsinh, value)


def _mj_acosh(value: object) -> object:
    return _unary_numpy(np.arccosh, value)


def _mj_atanh(value: object) -> object:
    return _unary_numpy(np.arctanh, value)


def _mj_cbrt(value: object) -> object:
    return _unary_numpy(np.cbrt, value)


def _mj_hypot(*args: object) -> object:
    values = _expand_args(args)
    if not values:
        raise ValueError("hypot requires at least one argument")
    result = np.abs(np.asarray(values[0]))
    for item in values[1:]:
        result = np.hypot(result, np.asarray(item))
    return _maybe_scalar(result)


def _mj_clamp(value: object, lower: object, upper: object) -> object:
    return _maybe_scalar(
        np.clip(cast("Any", value), cast("Any", lower), cast("Any", upper)),
    )


def _mj_round(value: object, decimals: object = 0) -> object:
    if _is_array_like(decimals):
        decimal_values = _integer_array(decimals, name="round")

        def round_item(item: object, places: object) -> object:
            return np.round(cast("Any", item), _as_integer(places, name="round"))

        return _maybe_scalar(np.vectorize(round_item)(value, decimal_values))

    decimal_places = _as_integer(decimals, name="round")
    return _maybe_scalar(
        np.round(cast("Any", value), decimal_places),
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


def _mj_variance(*args: object) -> object:
    if len(args) == 1 and _is_array_like(args[0]):
        array_arg = np.asarray(args[0])
        if array_arg.size < 2:
            raise ValueError("variance requires at least two values")
        return _maybe_scalar(np.var(array_arg, ddof=1))
    values = _expand_args(args)
    if len(values) < 2:
        raise ValueError("variance requires at least two values")
    return _maybe_scalar(np.var(np.asarray(values), axis=0, ddof=1))


def _mj_std(*args: object) -> object:
    if len(args) == 1 and _is_array_like(args[0]):
        array_arg = np.asarray(args[0])
        if array_arg.size < 2:
            raise ValueError("std requires at least two values")
        return _maybe_scalar(np.std(array_arg, ddof=1))
    values = _expand_args(args)
    if len(values) < 2:
        raise ValueError("std requires at least two values")
    return _maybe_scalar(np.std(np.asarray(values), axis=0, ddof=1))


def _mj_mode(*args: object) -> object:
    nan_key = object()
    values = _expand_args(args)
    counts: Counter[object] = Counter()
    originals: dict[object, object] = {}
    for item in values:
        for scalar in _flatten_nested(item):
            key = _mode_key(scalar, nan_key=nan_key)
            counts[key] += 1
            originals.setdefault(key, np.nan if key is nan_key else scalar)
    if not counts:
        raise ValueError("mode requires at least one argument")
    highest = max(counts.values())
    return [originals[key] for key, count in counts.items() if count == highest]


def _mj_factorial(value: object) -> object:
    def factorial(item: object) -> int:
        return math.factorial(_as_integer(item, name="factorial"))

    if _is_array_like(value):
        return _maybe_scalar(np.vectorize(factorial)(value))
    return factorial(value)


def _mj_gcd(*args: object) -> object:
    values = _expand_args(args)
    if not values:
        raise ValueError("gcd requires at least one argument")
    result = _integer_array(values[0], name="gcd")
    for item in values[1:]:
        result = np.gcd(result, _integer_array(item, name="gcd"))
    return _maybe_scalar(result)


def _mj_lcm(*args: object) -> object:
    values = _expand_args(args)
    if not values:
        raise ValueError("lcm requires at least one argument")
    result = _integer_array(values[0], name="lcm")
    for item in values[1:]:
        result = np.lcm(result, _integer_array(item, name="lcm"))
    return _maybe_scalar(result)


def _mj_combinations(total: object, choose: object) -> object:
    def combinations(left: object, right: object) -> int:
        return math.comb(
            _as_integer(left, name="combinations"),
            _as_integer(right, name="combinations"),
        )

    return _maybe_scalar(np.vectorize(combinations)(total, choose))


def _mj_permutations(total: object, choose: object | None = None) -> object:
    def permutations(left: object, right: object | None = None) -> int:
        total_value = _as_integer(left, name="permutations")
        if right is None:
            return math.factorial(total_value)
        return math.perm(total_value, _as_integer(right, name="permutations"))

    if choose is None:
        if _is_array_like(total):
            return _maybe_scalar(np.vectorize(permutations)(total))
        return permutations(total)
    return _maybe_scalar(np.vectorize(permutations)(total, choose))


def _mj_close_with_tolerances(
    left: object,
    right: object,
    *,
    rel_tol: float,
    abs_tol: float,
) -> object:
    try:
        return _maybe_scalar(
            np.isclose(
                cast("Any", left),
                cast("Any", right),
                rtol=rel_tol,
                atol=abs_tol,
            ),
        )
    except (TypeError, ValueError):
        return _binary_numpy(np.equal, left, right)


def _mj_close(left: object, right: object) -> object:
    return _mj_close_with_tolerances(
        left,
        right,
        rel_tol=MATHJS_REL_TOL,
        abs_tol=MATHJS_ABS_TOL,
    )


def _mj_larger(left: object, right: object) -> object:
    return _maybe_scalar(
        np.logical_and(
            np.greater(cast("Any", left), cast("Any", right)),
            np.logical_not(cast("Any", _mj_close(left, right))),
        ),
    )


def _mj_larger_eq(left: object, right: object) -> object:
    return _maybe_scalar(
        np.logical_or(
            np.greater(cast("Any", left), cast("Any", right)),
            cast("Any", _mj_close(left, right)),
        ),
    )


def _mj_smaller(left: object, right: object) -> object:
    return _maybe_scalar(
        np.logical_and(
            np.less(cast("Any", left), cast("Any", right)),
            np.logical_not(cast("Any", _mj_close(left, right))),
        ),
    )


def _mj_smaller_eq(left: object, right: object) -> object:
    return _maybe_scalar(
        np.logical_or(
            np.less(cast("Any", left), cast("Any", right)),
            cast("Any", _mj_close(left, right)),
        ),
    )


def _mj_equal(left: object, right: object) -> object:
    return _mj_close(left, right)


def _mj_unequal(left: object, right: object) -> object:
    return _maybe_scalar(np.logical_not(cast("Any", _mj_equal(left, right))))


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


_RELATIONAL_HELPERS = {
    "larger": _mj_larger,
    "largerEq": _mj_larger_eq,
    "smaller": _mj_smaller,
    "smallerEq": _mj_smaller_eq,
    "equal": _mj_equal,
    "unequal": _mj_unequal,
}


def _mj_relational(
    conditionals: Sequence[str],
    *terms: Callable[[], object],
) -> object:
    if len(conditionals) != len(terms) - 1:
        raise ValueError("RelationalNode requires one fewer conditional than params")

    left = terms[0]()
    result: object = True
    vector_mode = False

    for index, conditional in enumerate(conditionals):
        try:
            compare = _RELATIONAL_HELPERS[conditional]
        except KeyError as exc:
            msg = f"Unsupported relational conditional: {conditional!r}"
            raise ValueError(msg) from exc
        right = terms[index + 1]()
        comparison = compare(left, right)

        if _is_array_like(comparison):
            result = np.logical_and(np.asarray(result), np.asarray(comparison))
            vector_mode = True
        elif vector_mode:
            result = np.logical_and(np.asarray(result), cast("Any", comparison))
        elif not comparison:
            return False

        left = right

    return _maybe_scalar(result)


def _configured_comparison_helpers(  # noqa: C901
    config: EvalConfig,
) -> dict[str, Callable[..., object]]:
    def close(left: object, right: object) -> object:
        return _mj_close_with_tolerances(
            left,
            right,
            rel_tol=config.rel_tol,
            abs_tol=config.abs_tol,
        )

    def larger(left: object, right: object) -> object:
        return _maybe_scalar(
            np.logical_and(
                np.greater(cast("Any", left), cast("Any", right)),
                np.logical_not(cast("Any", close(left, right))),
            ),
        )

    def larger_eq(left: object, right: object) -> object:
        return _maybe_scalar(
            np.logical_or(
                np.greater(cast("Any", left), cast("Any", right)),
                cast("Any", close(left, right)),
            ),
        )

    def smaller(left: object, right: object) -> object:
        return _maybe_scalar(
            np.logical_and(
                np.less(cast("Any", left), cast("Any", right)),
                np.logical_not(cast("Any", close(left, right))),
            ),
        )

    def smaller_eq(left: object, right: object) -> object:
        return _maybe_scalar(
            np.logical_or(
                np.less(cast("Any", left), cast("Any", right)),
                cast("Any", close(left, right)),
            ),
        )

    def equal(left: object, right: object) -> object:
        return close(left, right)

    def unequal(left: object, right: object) -> object:
        return _maybe_scalar(np.logical_not(cast("Any", equal(left, right))))

    relational_helpers = {
        "larger": larger,
        "largerEq": larger_eq,
        "smaller": smaller,
        "smallerEq": smaller_eq,
        "equal": equal,
        "unequal": unequal,
    }

    def relational(
        conditionals: Sequence[str],
        *terms: Callable[[], object],
    ) -> object:
        if len(conditionals) != len(terms) - 1:
            raise ValueError(
                "RelationalNode requires one fewer conditional than params",
            )

        left = terms[0]()
        result: object = True
        vector_mode = False

        for index, conditional in enumerate(conditionals):
            try:
                compare = relational_helpers[conditional]
            except KeyError as exc:
                msg = f"Unsupported relational conditional: {conditional!r}"
                raise ValueError(msg) from exc
            right = terms[index + 1]()
            comparison = compare(left, right)

            if _is_array_like(comparison):
                result = np.logical_and(np.asarray(result), np.asarray(comparison))
                vector_mode = True
            elif vector_mode:
                result = np.logical_and(np.asarray(result), cast("Any", comparison))
            elif not comparison:
                return False

            left = right

        return _maybe_scalar(result)

    return {
        "__mj_equal": equal,
        "__mj_larger": larger,
        "__mj_larger_eq": larger_eq,
        "__mj_relational": relational,
        "__mj_smaller": smaller,
        "__mj_smaller_eq": smaller_eq,
        "__mj_unequal": unequal,
    }


def create_helper_functions(
    config: EvalConfig | Mapping[str, object] | None = None,
) -> dict[str, Callable[..., object]]:
    """Return generated-function helpers bound to an evaluator config."""
    normalized = coerce_eval_config(config)
    helpers = dict(HELPER_FUNCTIONS)
    helpers.update(_configured_comparison_helpers(normalized))
    return helpers


def source_preamble(config: EvalConfig | Mapping[str, object] | None = None) -> str:
    """Return imports and helper bindings for executable generated source."""
    normalized = coerce_eval_config(config)
    return (
        "from collections.abc import Mapping as __mj_Mapping\n"
        "from mathjs_to_func.errors import InputValidationError as "
        "__mj_InputValidationError\n"
        "from mathjs_to_func.errors import RuntimeEvaluationError as "
        "__mj_RuntimeEvaluationError\n"
        "from mathjs_to_func.helpers import EvalConfig as __mj_EvalConfig\n"
        "from mathjs_to_func.helpers import create_helper_functions as "
        "__mj_create_helper_functions\n\n"
        "__mj_Exception = Exception\n"
        "__mj_isinstance = isinstance\n"
        "__mj_set = set\n"
        "__mj_sorted = sorted\n"
        "globals().update(\n"
        "    __mj_create_helper_functions(\n"
        "        __mj_EvalConfig(\n"
        f"            rel_tol={normalized.rel_tol!r},\n"
        f"            abs_tol={normalized.abs_tol!r},\n"
        "        )\n"
        "    )\n"
        ")\n"
    )


def _mj_range(start: object, end: object, step: object = 1) -> np.ndarray:
    step_value = cast("Any", _maybe_scalar(step))
    if step_value == 0:
        raise ValueError("RangeNode step cannot be zero")
    start_value = cast("Any", _maybe_scalar(start))
    end_value = cast("Any", _maybe_scalar(end))
    values = np.arange(start_value, end_value + cast("Any", step_value), step_value)
    tolerance = abs(cast("Any", step_value)) * 1e-12
    if step_value > 0:
        return values[values <= end_value + tolerance]
    return values[values >= end_value - tolerance]


def _mj_index(value: object) -> int:
    index = _as_integer(value, name="IndexNode")
    if index < 1:
        raise ValueError("IndexNode indices must be >= 1")
    return index - 1


def _mj_range_index(value: object) -> int:
    index = _as_integer(value, name="RangeNode")
    if index < 1:
        raise ValueError("RangeNode indices must be >= 1")
    return index - 1


def _mj_index_range(start: object, end: object, step: object = 1) -> slice:
    step_value = _as_integer(step, name="RangeNode")
    if step_value == 0:
        raise ValueError("RangeNode step cannot be zero")
    start_index = _mj_range_index(start)
    end_index = _mj_range_index(end)
    if step_value > 0:
        stop: int | None = end_index + 1
    else:
        stop = None if end_index == 0 else end_index - 1
    return slice(start_index, stop, step_value)


def _mj_access(value: object, *dimensions: object) -> object:
    if not dimensions:
        return value
    index = cast("Any", dimensions[0] if len(dimensions) == 1 else tuple(dimensions))
    if isinstance(value, np.ndarray):
        return _maybe_scalar(cast("Any", value)[index])
    if len(dimensions) > 1 and any(isinstance(item, slice) for item in dimensions):
        try:
            return _maybe_scalar(np.asarray(value)[index])
        except (IndexError, TypeError, ValueError):
            pass
    current = value
    for dimension in dimensions:
        current = cast("Any", current)[dimension]
    return _maybe_scalar(current)


HELPER_FUNCTIONS: dict[str, Callable[..., object]] = {
    "__mj_abs": _mj_abs,
    "__mj_acos": _mj_acos,
    "__mj_acosh": _mj_acosh,
    "__mj_access": _mj_access,
    "__mj_and": _mj_and,
    "__mj_asin": _mj_asin,
    "__mj_asinh": _mj_asinh,
    "__mj_atan": _mj_atan,
    "__mj_atan2": _mj_atan2,
    "__mj_atanh": _mj_atanh,
    "__mj_cbrt": _mj_cbrt,
    "__mj_ceil": _mj_ceil,
    "__mj_clamp": _mj_clamp,
    "__mj_combinations": _mj_combinations,
    "__mj_cos": _mj_cos,
    "__mj_cosh": _mj_cosh,
    "__mj_equal": _mj_equal,
    "__mj_exp": _mj_exp,
    "__mj_factorial": _mj_factorial,
    "__mj_floor": _mj_floor,
    "__mj_gcd": _mj_gcd,
    "__mj_hypot": _mj_hypot,
    "__mj_index": _mj_index,
    "__mj_index_range": _mj_index_range,
    "__mj_lcm": _mj_lcm,
    "__mj_min": _mj_min,
    "__mj_max": _mj_max,
    "__mj_larger": _mj_larger,
    "__mj_larger_eq": _mj_larger_eq,
    "__mj_lazy_and": _mj_lazy_and,
    "__mj_lazy_or": _mj_lazy_or,
    "__mj_lazy_where": _mj_lazy_where,
    "__mj_log": _mj_log,
    "__mj_log1p": _mj_log1p,
    "__mj_log2": _mj_log2,
    "__mj_log10": _mj_log10,
    "__mj_mean": _mj_mean,
    "__mj_median": _mj_median,
    "__mj_mode": _mj_mode,
    "__mj_not": _mj_not,
    "__mj_or": _mj_or,
    "__mj_permutations": _mj_permutations,
    "__mj_range": _mj_range,
    "__mj_round": _mj_round,
    "__mj_sign": _mj_sign,
    "__mj_sin": _mj_sin,
    "__mj_sinh": _mj_sinh,
    "__mj_smaller": _mj_smaller,
    "__mj_smaller_eq": _mj_smaller_eq,
    "__mj_sqrt": _mj_sqrt,
    "__mj_std": _mj_std,
    "__mj_sum": _mj_sum,
    "__mj_tan": _mj_tan,
    "__mj_tanh": _mj_tanh,
    "__mj_ifnull": _mj_ifnull,
    "__mj_lazy_ifnull": _mj_lazy_ifnull,
    "__mj_relational": _mj_relational,
    "__mj_unequal": _mj_unequal,
    "__mj_variance": _mj_variance,
    "__mj_where": _mj_where,
    "__mj_xor": _mj_xor,
}

__all__ = [
    "HELPER_FUNCTIONS",
    "HELPER_NAME_MAP",
    "MATHJS_ABS_TOL",
    "MATHJS_REL_TOL",
    "EvalConfig",
    "_mj_abs",
    "_mj_access",
    "_mj_acos",
    "_mj_acosh",
    "_mj_and",
    "_mj_asin",
    "_mj_asinh",
    "_mj_atan",
    "_mj_atan2",
    "_mj_atanh",
    "_mj_cbrt",
    "_mj_ceil",
    "_mj_clamp",
    "_mj_combinations",
    "_mj_cos",
    "_mj_cosh",
    "_mj_equal",
    "_mj_exp",
    "_mj_factorial",
    "_mj_floor",
    "_mj_gcd",
    "_mj_hypot",
    "_mj_ifnull",
    "_mj_index",
    "_mj_index_range",
    "_mj_larger",
    "_mj_larger_eq",
    "_mj_lazy_and",
    "_mj_lazy_ifnull",
    "_mj_lazy_or",
    "_mj_lazy_where",
    "_mj_lcm",
    "_mj_log",
    "_mj_log1p",
    "_mj_log2",
    "_mj_log10",
    "_mj_max",
    "_mj_mean",
    "_mj_median",
    "_mj_min",
    "_mj_mode",
    "_mj_not",
    "_mj_or",
    "_mj_permutations",
    "_mj_range",
    "_mj_relational",
    "_mj_round",
    "_mj_sign",
    "_mj_sin",
    "_mj_sinh",
    "_mj_smaller",
    "_mj_smaller_eq",
    "_mj_sqrt",
    "_mj_std",
    "_mj_sum",
    "_mj_tan",
    "_mj_tanh",
    "_mj_unequal",
    "_mj_variance",
    "_mj_where",
    "_mj_xor",
    "coerce_eval_config",
    "create_helper_functions",
    "source_preamble",
]
