import math
from collections.abc import Mapping

import numpy as np
import pytest

from mathjs_to_func import (
    CircularDependencyError,
    ExpressionError,
    InputValidationError,
    InvalidNodeError,
    MissingTargetError,
    UnknownIdentifierError,
    build_evaluator,
)


def const(value):
    value_type = "boolean" if isinstance(value, bool) else "number"
    return {
        "type": "ConstantNode",
        "value": str(value).lower() if isinstance(value, bool) else str(value),
        "valueType": value_type,
    }


def symbol(name, *, use_mathjs=False):
    node = {"type": "SymbolNode", "name": name}
    if use_mathjs:
        node["mathjs"] = "SymbolNode"
    return node


def op(fn, *args):
    return {
        "type": "OperatorNode",
        "fn": fn,
        "op": fn,
        "args": list(args),
    }


def func(name, *args):
    return {
        "type": "FunctionNode",
        "fn": name,
        "args": list(args),
    }


def array(*items):
    return {"type": "ArrayNode", "items": list(items)}


@pytest.mark.parametrize(
    "fn_name,left,right,expected",
    [
        ("add", 2, 3, 5),
        ("add", -4, 9, 5),
        ("subtract", 10, 3, 7),
        ("subtract", -5, -7, 2),
        ("multiply", 6, 7, 42),
        ("multiply", -3, 9, -27),
        ("divide", 9, 3, 3),
        ("divide", 14, 2, 7),
        ("pow", 2, 5, 32),
        ("pow", 9, 0.5, 3),
        ("mod", 19, 5, 4),
        ("mod", 14, 4, 2),
        ("add", 12345, 67890, 80235),
        ("subtract", 2025, 1999, 26),
        ("multiply", 11, 13, 143),
        ("divide", 144, 12, 12),
        ("pow", 3, 4, 81),
        ("pow", 5, 3, 125),
        ("mod", 100, 9, 1),
        ("add", -100, 50, -50),
        ("subtract", 50, -20, 70),
        ("multiply", -8, -2, 16),
        ("divide", 100, 25, 4),
        ("pow", 10, 2, 100),
    ],
)
def test_binary_operations_without_inputs(fn_name, left, right, expected):
    expr = {"result": op(fn_name, const(left), const(right))}
    evaluator = build_evaluator(expressions=expr, inputs=[], target="result")
    assert evaluator({}) == pytest.approx(expected)


@pytest.mark.parametrize(
    "fn_name,left,right,inputs,scope,expected",
    [
        ("add", symbol("x"), const(5), ["x"], {"x": 7}, 12),
        ("subtract", const(20), symbol("y"), ["y"], {"y": 3}, 17),
        ("multiply", symbol("a"), symbol("b"), ["a", "b"], {"a": 4, "b": 5}, 20),
        ("divide", symbol("t"), const(2), ["t"], {"t": 9}, 4.5),
        ("pow", symbol("n"), const(3), ["n"], {"n": 2}, 8),
        ("mod", symbol("m"), const(4), ["m"], {"m": 21}, 1),
        ("add", symbol("x"), symbol("y"), ["x", "y"], {"x": 1.5, "y": 2.5}, 4),
        ("subtract", symbol("x"), const(-5), ["x"], {"x": -5}, 0),
        ("multiply", symbol("c"), const(-2), ["c"], {"c": 6}, -12),
        ("divide", const(42), symbol("d"), ["d"], {"d": 6}, 7),
    ],
)
def test_binary_operations_with_inputs(fn_name, left, right, inputs, scope, expected):
    expr = {"result": op(fn_name, left, right)}
    evaluator = build_evaluator(expressions=expr, inputs=inputs, target="result")
    assert evaluator(scope) == pytest.approx(expected)


@pytest.mark.parametrize(
    "fn_name,operand,value,expected",
    [
        ("unaryMinus", const(5), None, -5),
        ("unaryMinus", symbol("x"), {"x": 7}, -7),
        ("unaryPlus", const(-9), None, -9),
        ("unaryPlus", symbol("v"), {"v": 3}, 3),
        ("unaryMinus", const(0), None, 0),
        ("unaryPlus", const(0), None, 0),
        ("unaryMinus", symbol("p"), {"p": -2}, 2),
        ("unaryPlus", symbol("p"), {"p": -2}, -2),
        ("unaryMinus", const(123.45), None, -123.45),
        ("unaryPlus", const(99.9), None, 99.9),
    ],
)
def test_unary_operations(fn_name, operand, value, expected):
    expr = {"result": op(fn_name, operand)}
    inputs = list(value.keys()) if isinstance(value, Mapping) else []
    evaluator = build_evaluator(expressions=expr, inputs=inputs, target="result")
    scope = value or {}
    assert evaluator(scope) == pytest.approx(expected)


@pytest.mark.parametrize(
    "values,expected",
    [
        ([5, 3, 7], 3),
        ([symbol("x"), const(2), const(8)], 1),
        ([array(const(5), const(9))], 5),
        ([const(-5), const(-1), const(-10)], -10),
        ([symbol("m"), symbol("n"), const(0)], -10),
        ([const(100)], 100),
        ([symbol("u"), const(100)], 5),
        ([array(const(-2), const(-3), const(0))], -3),
        ([const(1), const(1)], 1),
        ([symbol("p"), const(3)], 2),
    ],
)
def test_min_function(values, expected):
    args = [v if isinstance(v, dict) else const(v) for v in values]
    expr = {"result": func("min", *args)}
    inputs = ["x", "m", "n", "u", "p"]
    scope = {
        "x": 1,
        "m": -10,
        "n": -4,
        "u": 5,
        "p": 2,
    }
    evaluator = build_evaluator(expressions=expr, inputs=inputs, target="result")
    result = evaluator({k: scope.get(k) for k in inputs if k in scope})
    assert result == expected


@pytest.mark.parametrize(
    "values,expected",
    [
        ([symbol("x"), const(2), const(8)], 8),
        ([array(const(5), const(9))], 9),
        ([const(-5), const(-1), const(-10)], -1),
        ([symbol("m"), symbol("n"), const(0)], 40),
        ([array(const(10), const(20), const(30))], 30),
        ([symbol("u"), const(100)], 100),
        ([array(const(-5), const(-2))], -2),
        ([const(1), const(1)], 1),
        ([symbol("p"), const(3)], 7),
        ([const(0)], 0),
    ],
)
def test_max_function(values, expected):
    args = [v if isinstance(v, dict) else const(v) for v in values]
    expr = {"result": func("max", *args)}
    inputs = ["x", "m", "n", "u", "p"]
    scope = {
        "x": 8,
        "m": 40,
        "n": -4,
        "u": 100,
        "p": 7,
    }
    evaluator = build_evaluator(expressions=expr, inputs=inputs, target="result")
    result = evaluator({k: scope.get(k) for k in inputs if k in scope})
    assert result == expected


@pytest.mark.parametrize(
    "values,expected",
    [
        ([symbol("x"), const(2), const(8)], 15),
        ([array(const(5), const(9))], 14),
        ([const(-5), const(-1), const(-10)], -16),
        ([symbol("m"), symbol("n"), const(0)], 36),
        ([array(const(-10), const(-2))], -12),
        ([array(const(10), const(20), const(30))], 60),
        ([symbol("u"), const(100)], 105),
        ([const(100)], 100),
        ([const(1), const(4)], 5),
        ([symbol("p"), const(3)], 9),
    ],
)
def test_sum_function(values, expected):
    args = [v if isinstance(v, dict) else const(v) for v in values]
    expr = {"result": func("sum", *args)}
    inputs = ["x", "m", "n", "u", "p"]
    scope = {
        "x": 5,
        "m": 12,
        "n": 24,
        "u": 5,
        "p": 6,
    }
    evaluator = build_evaluator(expressions=expr, inputs=inputs, target="result")
    result = evaluator({k: scope.get(k) for k in inputs if k in scope})
    assert result == expected


@pytest.mark.parametrize(
    "value,fallback,expected",
    [
        (symbol("x"), const(10), 7),
        (const(5), const(10), 5),
        (const(0), const(10), 0),
        (symbol("nan_val"), const(3), 3),
        (symbol("arr"), const(1), np.array([1, 0])),
        (const(42), const(99), 42),
        (symbol("none_val"), const(11), 11),
        (symbol("maybe"), const(9), 9),
        (const(100), const(1), 100),
        (symbol("arr_nan"), symbol("arr_alt"), np.array([5, 3])),
    ],
)
def test_ifnull_behavior(value, fallback, expected):
    expr = {"result": func("ifnull", value, fallback)}
    inputs = []
    scope = {}
    if isinstance(value, dict) and value.get("type") == "SymbolNode":
        inputs.append(value["name"])
    if isinstance(fallback, dict) and fallback.get("type") == "SymbolNode":
        inputs.append(fallback["name"])
    scope.update(
        {
            "x": 7,
            "arr": np.array([np.nan, 0]),
            "maybe": None,
            "arr_nan": np.array([np.nan, 3]),
            "arr_alt": np.array([5, 3]),
            "nan_val": float("nan"),
            "none_val": None,
        },
    )
    evaluator = build_evaluator(expressions=expr, inputs=inputs, target="result")
    result = evaluator({k: scope[k] for k in inputs})
    if isinstance(expected, np.ndarray):
        np.testing.assert_allclose(result, expected)
    elif isinstance(expected, float) and math.isnan(expected):
        assert math.isnan(result)
    else:
        assert result == expected


@pytest.mark.parametrize("length", list(range(1, 21)))
def test_dependency_chain(length):
    expressions = {"node0": symbol("base")}
    for i in range(1, length + 1):
        left = symbol(f"node{i - 1}")
        expressions[f"node{i}"] = op("add", left, const(1))
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["base"],
        target=f"node{length}",
    )
    result = evaluator({"base": 0})
    assert result == length


def test_multiple_inputs_and_reuse():
    expressions = {
        "sum_ab": op("add", symbol("a"), symbol("b")),
        "weighted": op("multiply", symbol("sum_ab"), symbol("weight")),
        "result": op("subtract", symbol("weighted"), symbol("offset")),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["a", "b", "weight", "offset"],
        target="result",
    )
    output = evaluator({"a": 10, "b": 5, "weight": 2, "offset": 4})
    assert output == 26
    assert evaluator.__mathjs_required_inputs__ == ("a", "b", "offset", "weight")
    assert evaluator.__mathjs_evaluation_order__[0] == "sum_ab"


def test_include_source_attribute():
    expressions = {"res": op("add", const(1), const(2))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=[],
        target="res",
        include_source=True,
    )
    assert hasattr(evaluator, "__mathjs_source__")
    assert "def _compiled" in evaluator.__mathjs_source__


def test_include_source_absent_by_default():
    expressions = {"res": op("add", const(1), const(2))}
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert not hasattr(evaluator, "__mathjs_source__")


def test_payload_argument():
    payload = {
        "expressions": {"out": op("add", const(4), const(6))},
        "inputs": [],
        "target": "out",
    }
    evaluator = build_evaluator(payload=payload)
    assert evaluator({}) == 10


def test_missing_target_error():
    with pytest.raises(MissingTargetError):
        build_evaluator(expressions={"a": const(1)}, inputs=[], target="missing")


def test_unknown_identifier_error():
    expressions = {"res": op("add", symbol("a"), symbol("b"))}
    with pytest.raises(UnknownIdentifierError):
        build_evaluator(expressions=expressions, inputs=["a"], target="res")


def test_cycle_detection():
    expressions = {
        "a": op("add", symbol("b"), const(1)),
        "b": op("add", symbol("a"), const(1)),
    }
    with pytest.raises(CircularDependencyError):
        build_evaluator(expressions=expressions, inputs=[], target="a")


def test_invalid_node_type():
    expressions = {"res": {"type": "Unsupported", "value": 1}}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_invalid_symbol_name():
    expressions = {"res": op("add", symbol("valid"), symbol("__bad name__"))}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=["valid"], target="res")


def test_expression_conflicts_with_input():
    expressions = {"x": const(5), "res": op("add", symbol("x"), const(1))}
    with pytest.raises(ExpressionError):
        build_evaluator(expressions=expressions, inputs=["x"], target="res")


def test_input_validation_requires_mapping():
    expressions = {"res": symbol("x")}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    with pytest.raises(InputValidationError):
        evaluator([1, 2, 3])  # type: ignore[arg-type]


def test_input_validation_missing_keys():
    expressions = {"res": op("add", symbol("x"), symbol("y"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y", "z"],
        target="res",
    )
    with pytest.raises(InputValidationError):
        evaluator({"x": 1, "z": 2})


def test_input_validation_unexpected_keys():
    expressions = {"res": symbol("x")}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    with pytest.raises(InputValidationError):
        evaluator({"x": 1, "y": 2})


def test_mathjs_field_support():
    expressions = {
        "res": {
            "mathjs": "OperatorNode",
            "fn": "add",
            "args": [
                {"mathjs": "ConstantNode", "value": "2", "valueType": "number"},
                {"mathjs": "ConstantNode", "value": "3", "valueType": "number"},
            ],
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 5


def test_boolean_constants():
    expressions = {
        "res": func("sum", const(True), const(True), const(False)),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 2


def test_nested_functions_and_arrays():
    expressions = {
        "avg": op(
            "divide",
            func("sum", array(symbol("x"), symbol("y"), const(2))),
            const(3),
        ),
    }
    evaluator = build_evaluator(
        expressions=expressions, inputs=["x", "y"], target="avg"
    )
    assert evaluator({"x": 4, "y": 1}) == pytest.approx(7 / 3)


def test_large_graph_with_arrays_and_functions():
    expressions = {
        "base": symbol("seed"),
        "scaled": op("multiply", symbol("base"), const(2)),
        "offset": op("add", symbol("scaled"), const(5)),
        "vector": func("sum", array(symbol("scaled"), symbol("offset"))),
        "target": func("max", symbol("vector"), const(20)),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["seed"],
        target="target",
    )
    result = evaluator({"seed": np.array([5, 10])})
    np.testing.assert_allclose(result, np.array([25, 45]))


def test_required_inputs_sorted():
    expressions = {
        "res": func("sum", symbol("b"), symbol("a"), symbol("c")),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["c", "b", "a"],
        target="res",
    )
    assert evaluator.__mathjs_required_inputs__ == ("a", "b", "c")


def test_evaluation_order_subset():
    expressions = {
        "a": op("add", symbol("x"), const(1)),
        "b": op("add", symbol("a"), const(1)),
        "c": op("add", symbol("b"), const(1)),
        "unused": op("add", const(0), const(1)),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="c",
    )
    assert "unused" not in evaluator.__mathjs_evaluation_order__


def test_zero_length_inputs():
    expressions = {"res": const(42)}
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 42


def test_duplicate_inputs_raise():
    expressions = {"res": symbol("x")}
    with pytest.raises(ExpressionError):
        build_evaluator(expressions=expressions, inputs=["x", "x"], target="res")


def test_invalid_input_name_raises():
    expressions = {"res": symbol("valid")}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=["valid", "@bad"], target="res")


def test_invalid_target_name():
    expressions = {"res": const(1)}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="bad name")


def test_array_node_requires_items():
    expressions = {"res": {"type": "ArrayNode", "items": None}}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_function_requires_supported_name():
    expressions = {"res": func("sqrt", const(4))}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_operator_requires_args():
    expressions = {"res": {"type": "OperatorNode", "fn": "add", "args": None}}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_parenthesis_pass_through():
    expressions = {"res": {"type": "ParenthesisNode", "content": const(5)}}
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 5


def test_array_with_symbols():
    expressions = {"res": array(symbol("x"), symbol("y"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    np.testing.assert_allclose(evaluator({"x": 2, "y": 3}), [2, 3])


def test_unary_requires_supported_fn():
    expressions = {"res": op("unknown", const(1))}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_binary_requires_supported_fn():
    expressions = {"res": op("and", const(1), const(2))}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_function_argument_validation():
    expressions = {
        "res": {"type": "FunctionNode", "fn": "min", "args": [None]},
    }
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_operator_argument_validation():
    expressions = {
        "res": {"type": "OperatorNode", "fn": "add", "args": [None, const(1)]},
    }
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_builder_handles_mathjs_function_reference():
    expressions = {
        "res": {
            "type": "FunctionNode",
            "fn": {"name": "max"},
            "args": [const(1), const(2)],
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 2


def test_input_payload_missing_key():
    with pytest.raises(ExpressionError):
        build_evaluator(payload={"inputs": [], "target": "x"})


def test_scope_extra_keys_message():
    expressions = {"res": symbol("x")}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    with pytest.raises(InputValidationError) as excinfo:
        evaluator({"x": 1, "y": 2})
    assert "Unexpected inputs" in str(excinfo.value)


def test_scope_missing_keys_message():
    expressions = {"res": op("add", symbol("x"), const(1))}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    with pytest.raises(InputValidationError) as excinfo:
        evaluator({})
    assert "Missing required inputs" in str(excinfo.value)


def test_scope_accepts_numpy_scalars():
    expressions = {"res": op("add", symbol("x"), const(1))}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    assert evaluator({"x": np.float64(2)}) == 3


def test_helper_sum_reduces_arrays():
    expressions = {"res": func("sum", array(symbol("x"), symbol("y")))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    np.testing.assert_allclose(
        evaluator({"x": np.array([1, 2]), "y": np.array([3, 4])}),
        [4, 6],
    )


def test_ifnull_numpy_nan_array():
    expressions = {
        "res": func("ifnull", symbol("x"), symbol("fallback")),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "fallback"],
        target="res",
    )
    result = evaluator(
        {
            "x": np.array([np.nan, 2]),
            "fallback": np.array([0, 0]),
        },
    )
    np.testing.assert_allclose(result, [0, 2])


def test_dependency_graph_ignores_unused_inputs():
    expressions = {"res": op("add", symbol("x"), const(1))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "unused"],
        target="res",
    )
    assert evaluator.__mathjs_required_inputs__ == ("x",)


def test_payload_and_direct_args_conflict():
    payload = {"expressions": {"res": const(1)}, "inputs": [], "target": "res"}
    evaluator = build_evaluator(
        expressions=payload["expressions"], inputs=[], target="res"
    )
    assert evaluator({}) == 1


def test_minimum_argument_error_message():
    expressions = {"res": func("min")}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_sum_requires_argument():
    expressions = {"res": func("sum")}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_ifnull_requires_two_arguments():
    expressions = {"res": func("ifnull", const(1))}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_numpy_vector_min_max_chain():
    expressions = {
        "min_val": func("min", symbol("x")),
        "max_val": func("max", symbol("min_val"), symbol("y")),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="max_val",
    )
    result = evaluator({"x": np.array([2, 5]), "y": np.array([3, 4])})
    np.testing.assert_allclose(result, [3, 4])


def test_sum_of_boolean_inputs():
    expressions = {"res": func("sum", symbol("x"), symbol("y"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    assert evaluator({"x": True, "y": False}) == 1


def test_min_accepts_tuple_argument():
    expressions = {"res": func("min", array(symbol("x"), symbol("y")))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    np.testing.assert_allclose(
        evaluator({"x": np.array([4, 1]), "y": np.array([3, 2])}),
        [3, 1],
    )


def test_sum_of_nested_arrays():
    expressions = {
        "res": func(
            "sum",
            array(symbol("x"), const(1)),
            array(const(2), symbol("y")),
        ),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    result = evaluator({"x": 1, "y": 3})
    np.testing.assert_allclose(result, np.array([3, 4]))


def test_ifnull_respects_zero_values():
    expressions = {"res": func("ifnull", symbol("x"), const(9))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 0}) == 0


def test_sum_large_number_of_terms():
    items = [symbol(f"v{i}") for i in range(10)]
    expressions = {"res": func("sum", *items)}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=[f"v{i}" for i in range(10)],
        target="res",
    )
    values = {f"v{i}": i for i in range(10)}
    assert evaluator(values) == sum(range(10))


def test_min_accepts_single_value():
    expressions = {"res": func("min", symbol("x"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 99}) == 99


def test_max_accepts_single_value():
    expressions = {"res": func("max", symbol("x"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": -1}) == -1


def test_sum_accepts_single_array():
    expressions = {"res": func("sum", array(symbol("x"), symbol("y")))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    result = evaluator({"x": 3, "y": 4})
    assert result == 7


def test_modulo_with_negative_numbers():
    expressions = {"res": op("mod", symbol("x"), const(4))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": -3}) == pytest.approx(1)


def test_pow_with_fractional_exponent():
    expressions = {"res": op("pow", symbol("x"), const(0.5))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 25}) == pytest.approx(5)


def test_divide_by_nonzero_float():
    expressions = {"res": op("divide", symbol("x"), const(0.5))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 4}) == 8


def test_multiply_by_zero_array():
    expressions = {"res": op("multiply", symbol("x"), const(0))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    np.testing.assert_allclose(evaluator({"x": np.array([5, 10])}), [0, 0])


def test_add_mixed_numeric_types():
    expressions = {"res": op("add", symbol("x"), const(1.5))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 2}) == pytest.approx(3.5)


def test_subtract_results_negative():
    expressions = {"res": op("subtract", symbol("x"), const(5))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 3}) == -2


def test_nested_dependency_with_arrays():
    expressions = {
        "first": func("sum", array(symbol("x"), symbol("y"))),
        "second": func("max", symbol("first"), const(10)),
        "target": op("add", symbol("second"), const(2)),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="target",
    )
    np.testing.assert_allclose(
        evaluator({"x": 4, "y": 3}),
        12,
    )


def test_payload_with_include_source():
    payload = {
        "expressions": {"res": op("add", const(1), const(2))},
        "inputs": [],
        "target": "res",
    }
    evaluator = build_evaluator(payload=payload, include_source=True)
    assert evaluator({}) == 3
    assert hasattr(evaluator, "__mathjs_source__")


def test_ifnull_with_python_nan():
    expressions = {"res": func("ifnull", symbol("x"), const(5))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": float("nan")}) == 5


def test_sum_returns_numpy_scalar():
    expressions = {"res": func("sum", array(symbol("x"), symbol("y")))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    result = evaluator({"x": 1, "y": 2})
    assert isinstance(result, (int, float, np.number))


def test_minimum_of_booleans():
    expressions = {"res": func("min", symbol("x"), symbol("y"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    assert evaluator({"x": True, "y": False}) is False


def test_max_of_booleans():
    expressions = {"res": func("max", symbol("x"), symbol("y"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    assert evaluator({"x": True, "y": False}) is True


def test_sum_of_boolean_array():
    expressions = {"res": func("sum", array(symbol("x")))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": [True, False, True]}) == 2


def test_function_node_with_object_fn():
    expressions = {
        "res": {
            "type": "FunctionNode",
            "fn": {"name": "min"},
            "args": [const(3), const(4)],
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 3


def test_builder_rejects_empty_args():
    expressions = {"res": {"type": "FunctionNode", "fn": "max", "args": []}}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_array_argument_must_be_mapping():
    expressions = {
        "res": {
            "type": "ArrayNode",
            "items": [const(1), None],
        },
    }
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_operator_requires_mapping_children():
    expressions = {
        "res": {
            "type": "OperatorNode",
            "fn": "add",
            "args": [const(1), None],
        },
    }
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_sum_with_nested_lists():
    expressions = {
        "res": func("sum", array(array(symbol("x"), symbol("y")), const(2))),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )
    result = evaluator({"x": 1, "y": 2})
    assert isinstance(result, np.ndarray)


def test_compile_to_function_twice():
    expressions = {"res": op("add", const(1), const(2))}
    fn1 = build_evaluator(expressions=expressions, inputs=[], target="res")
    fn2 = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert fn1({}) == fn2({}) == 3


def test_symbol_with_mathjs_field():
    expressions = {
        "res": {
            "type": "OperatorNode",
            "fn": "add",
            "args": [symbol("x", use_mathjs=True), const(1)],
        },
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x"],
        target="res",
    )
    assert evaluator({"x": 1}) == 2


def test_const_with_numeric_string():
    expressions = {"res": const("12")}
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 12
