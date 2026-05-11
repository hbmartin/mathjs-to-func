import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from mathjs_to_func import (
    CircularDependencyError,
    ExpressionError,
    InputValidationError,
    InvalidNodeError,
    MissingTargetError,
    RuntimeEvaluationError,
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


def relational(conditionals, *params):
    return {
        "type": "RelationalNode",
        "conditionals": list(conditionals),
        "params": list(params),
    }


def array(*items):
    return {"type": "ArrayNode", "items": list(items)}


def range_node(start, end, step=None):
    node = {"type": "RangeNode", "start": start, "end": end}
    if step is not None:
        node["step"] = step
    return node


def index(*dimensions):
    return {"type": "IndexNode", "dimensions": list(dimensions)}


def accessor(obj, *dimensions):
    return {"type": "AccessorNode", "object": obj, "index": index(*dimensions)}


def object_node(**properties: object):
    return {"type": "ObjectNode", "properties": properties}


def assert_runtime_cause(excinfo, cause_type, message: str):
    cause = excinfo.value.__cause__
    assert excinfo.value.expression == "res"
    assert isinstance(cause, cause_type)
    assert message in str(cause)


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
    inputs: list[str] = [str(k) for k in value] if isinstance(value, Mapping) else []
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
        cast("Any", evaluator)([1, 2, 3])


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
        expressions=expressions,
        inputs=["x", "y"],
        target="avg",
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
    expressions = {"res": func("distance", const(4))}
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
    expressions = {"res": op("bitAnd", const(1), const(2))}
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


def test_builder_handles_serialized_mathjs_symbol_function_reference():
    expressions = {
        "res": {
            "mathjs": "FunctionNode",
            "fn": {"mathjs": "SymbolNode", "name": "sqrt"},
            "args": [{"mathjs": "ConstantNode", "value": "81", "valueType": "number"}],
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == 9


def test_broad_scalar_functions():
    expressions = {
        "res": func(
            "sum",
            func("abs", const(-4)),
            func("sqrt", const(16)),
            func("log", func("exp", const(3))),
            func("round", const(2.6)),
            func("floor", const(2.9)),
            func("ceil", const(2.1)),
            func("sign", const(-12)),
            func("mean", array(const(2), const(4), const(6))),
            func("median", array(const(9), const(1), const(5))),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == pytest.approx(27)


def test_mathjs_builtin_constants_without_inputs():
    expressions = {
        "res": func(
            "sum",
            symbol("pi"),
            symbol("e"),
            symbol("tau"),
            symbol("SQRT2"),
            symbol("phi"),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")

    assert evaluator.__mathjs_required_inputs__ == ()
    assert evaluator({}) == pytest.approx(
        math.pi + math.e + math.tau + math.sqrt(2) + ((1 + math.sqrt(5)) / 2),
    )


def test_mathjs_builtin_nullish_values():
    expressions = {
        "nan_value": func("ifnull", symbol("NaN"), const(3)),
        "null_value": func("ifnull", symbol("null"), const(5)),
        "undefined_value": func("ifnull", symbol("undefined"), const(7)),
        "res": func(
            "sum",
            symbol("nan_value"),
            symbol("null_value"),
            symbol("undefined_value"),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")

    assert evaluator({}) == 15


def test_inputs_override_mathjs_builtin_constants():
    expressions = {"res": op("add", symbol("pi"), const(1))}
    evaluator = build_evaluator(expressions=expressions, inputs=["pi"], target="res")

    assert evaluator.__mathjs_required_inputs__ == ("pi",)
    assert evaluator({"pi": 2}) == 3


def test_mathjs_equal_uses_default_tolerances():
    expressions = {
        "value": op("add", const(0.1), const(0.2)),
        "res": op("equal", symbol("value"), const(0.3)),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")

    assert evaluator({}) is True


def test_mathjs_relational_operators_use_default_tolerances():
    expressions = {
        "value": op("add", const(0.1), const(0.2)),
        "larger": op("larger", symbol("value"), const(0.3)),
        "larger_eq": op("largerEq", symbol("value"), const(0.3)),
        "smaller": op("smaller", const(0.3), symbol("value")),
        "res": array(symbol("larger"), symbol("larger_eq"), symbol("smaller")),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")

    assert evaluator({}) == [False, True, False]


def test_mathjs_equal_tolerances_vectorize():
    expressions = {"res": op("equal", symbol("x"), const(0.3))}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")

    np.testing.assert_array_equal(
        evaluator({"x": np.array([0.1 + 0.2, 0.31])}),
        np.array([True, False]),
    )


def test_nullish_function_alias_and_operator():
    expressions = {
        "function_alias": func("nullish", symbol("maybe"), const(4)),
        "operator_value": op(
            "nullish",
            const(2),
            op("divide", const(1), const(0)),
        ),
        "res": func("sum", symbol("function_alias"), symbol("operator_value")),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["maybe"], target="res")

    assert evaluator({"maybe": None}) == 6


def test_relational_logical_and_conditional_nodes():
    expressions = {
        "condition": {
            "type": "OperatorNode",
            "fn": "and",
            "args": [
                {
                    "type": "OperatorNode",
                    "fn": "largerEq",
                    "args": [symbol("x"), const(10)],
                },
                {
                    "type": "OperatorNode",
                    "fn": "smaller",
                    "args": [symbol("x"), const(20)],
                },
            ],
        },
        "res": {
            "type": "ConditionalNode",
            "condition": symbol("condition"),
            "trueExpr": const(1),
            "falseExpr": const(0),
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    assert evaluator({"x": 12}) == 1
    assert evaluator({"x": 25}) == 0


def test_relational_node_scalar_chain():
    expressions = {
        "res": relational(
            ["smaller", "smallerEq"],
            const(10),
            symbol("x"),
            const(50),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")

    assert evaluator({"x": 25}) is True
    assert evaluator({"x": 10}) is False
    assert evaluator({"x": 60}) is False


def test_relational_node_vectorizes_chained_comparisons():
    expressions = {
        "res": relational(
            ["smaller", "smallerEq"],
            const(10),
            symbol("x"),
            const(50),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")

    np.testing.assert_array_equal(
        evaluator({"x": np.array([5, 25, 50, 55])}),
        np.array([False, True, True, False]),
    )


def test_relational_node_short_circuits_scalar_false_branch():
    expressions = {
        "res": relational(
            ["unequal", "larger"],
            symbol("x"),
            const(0),
            op("divide", const(1), symbol("x")),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")

    assert evaluator({"x": 0}) is False


def test_relational_and_conditional_nodes_vectorize():
    expressions = {
        "res": {
            "type": "ConditionalNode",
            "condition": {
                "type": "OperatorNode",
                "fn": "larger",
                "args": [symbol("x"), const(0)],
            },
            "trueExpr": symbol("x"),
            "falseExpr": const(0),
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    np.testing.assert_allclose(evaluator({"x": np.array([-2, 3, -1, 4])}), [0, 3, 0, 4])


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.array([2, 4, 9]), [5, 4]),
        (np.array(5), [5, 5]),
        ([[1, 2], [3, 4]], [2.5, 2.5]),
    ],
)
def test_mean_and_median_reduce_single_array_like(value, expected):
    expressions = {
        "mean": func("mean", symbol("x")),
        "median": func("median", symbol("x")),
        "res": array(symbol("mean"), symbol("median")),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    np.testing.assert_allclose(evaluator({"x": value}), expected)


@pytest.mark.parametrize("fn_name", ["mean", "median"])
def test_mean_and_median_reject_empty_single_array_like(fn_name):
    expressions = {"res": func(fn_name, symbol("x"))}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({"x": np.array([])})
    assert_runtime_cause(
        excinfo,
        ValueError,
        f"{fn_name} requires at least one argument",
    )


def test_scalar_and_short_circuits_right_operand():
    expressions = {
        "res": op(
            "and",
            op("unequal", symbol("x"), const(0)),
            op("larger", op("divide", const(1), symbol("x")), const(1)),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    assert evaluator({"x": 0}) is False
    assert evaluator({"x": 0.5}) is True


def test_scalar_or_short_circuits_right_operand():
    expressions = {
        "res": op(
            "or",
            op("equal", symbol("x"), const(0)),
            op("larger", op("divide", const(1), symbol("x")), const(1)),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    assert evaluator({"x": 0}) is True
    assert evaluator({"x": 2}) is False


@pytest.mark.parametrize(
    "fn_name,left,value,expected",
    [
        ("and", False, np.array([True, True]), np.array([False, False])),
        ("or", True, np.array([False, False]), np.array([True, True])),
    ],
)
def test_scalar_short_circuit_logical_nodes_preserve_array_right(
    fn_name,
    left,
    value,
    expected,
):
    expressions = {"res": op(fn_name, const(left), symbol("x"))}
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    np.testing.assert_array_equal(evaluator({"x": value}), expected)


def test_none_and_array_right_returns_false_array():
    expressions = {"res": op("and", symbol("x"), symbol("y"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["x", "y"],
        target="res",
    )

    result = evaluator({"x": None, "y": np.array([True, False])})

    np.testing.assert_array_equal(result, np.array([False, False]))


def test_scalar_conditional_short_circuits_dead_branch():
    expressions = {
        "res": {
            "type": "ConditionalNode",
            "condition": op("unequal", symbol("x"), const(0)),
            "trueExpr": op("divide", const(1), symbol("x")),
            "falseExpr": const(0),
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    assert evaluator({"x": 0}) == 0
    assert evaluator({"x": 2}) == pytest.approx(0.5)


def test_vector_logical_nodes_still_vectorize():
    expressions = {
        "res": op(
            "and",
            op("larger", symbol("x"), const(0)),
            op("smaller", symbol("x"), const(5)),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=["x"], target="res")
    np.testing.assert_array_equal(
        evaluator({"x": np.array([-1, 3, 8])}),
        np.array([False, True, False]),
    )


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
        expressions=payload["expressions"],
        inputs=[],
        target="res",
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


@pytest.mark.parametrize(
    "expression,expected",
    [
        (func("sin", op("divide", symbol("pi"), const(2))), 1),
        (func("cos", const(0)), 1),
        (func("tan", op("divide", symbol("pi"), const(4))), 1),
        (func("asin", const(1)), math.pi / 2),
        (func("acos", const(1)), 0),
        (func("atan", const(1)), math.pi / 4),
        (func("atan2", const(1), const(1)), math.pi / 4),
        (func("sinh", const(0)), 0),
        (func("cosh", const(0)), 1),
        (func("tanh", const(0)), 0),
        (func("asinh", const(0)), 0),
        (func("acosh", const(1)), 0),
        (func("atanh", const(0)), 0),
        (func("log2", const(8)), 3),
        (func("log10", const(100)), 2),
        (func("log", const(10000), const(10)), 4),
        (func("log1p", const(1)), math.log(2)),
        (func("log1p", const(9999), const(10)), 4),
        (func("cbrt", const(27)), 3),
        (func("hypot", const(3), const(4)), 5),
        (func("clamp", const(10), const(0), const(5)), 5),
        (func("factorial", const(5)), 120),
        (func("gcd", const(24), const(18)), 6),
        (func("lcm", const(4), const(6)), 12),
        (func("variance", array(const(1), const(2), const(3))), 1),
        (func("std", array(const(1), const(2), const(3))), 1),
        (func("combinations", const(5), const(2)), 10),
        (func("permutations", const(5), const(2)), 20),
        (func("permutations", const(4)), 24),
    ],
)
def test_v05_scalar_function_expansion(expression, expected):
    evaluator = build_evaluator(
        expressions={"res": expression},
        inputs=[],
        target="res",
    )
    assert evaluator({}) == pytest.approx(expected)


def test_hypot_single_negative_returns_magnitude():
    evaluator = build_evaluator(
        expressions={"res": func("hypot", const(-5))},
        inputs=[],
        target="res",
    )

    assert evaluator({}) == 5


def test_mode_returns_all_modes_in_input_order():
    expressions = {
        "res": func("mode", array(const(1), const(2), const(2), const(3), const(3))),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == [2, 3]


def test_mode_flattens_nested_arrays_and_coalesces_nan():
    evaluator = build_evaluator(
        expressions={"res": func("mode", symbol("x"))},
        inputs=["x"],
        target="res",
    )

    result = evaluator({"x": [[float("nan")], [float("nan")], [1]]})

    assert len(result) == 1
    assert math.isnan(result[0])


def test_gcd_and_lcm_accept_integer_compatible_float_arrays():
    expressions = {
        "gcds": func("gcd", symbol("gcd_values"), const(6)),
        "lcms": func("lcm", symbol("lcm_values"), const(6)),
        "res": object_node(gcds=symbol("gcds"), lcms=symbol("lcms")),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["gcd_values", "lcm_values"],
        target="res",
    )

    result = evaluator(
        {
            "gcd_values": np.array([6.0, 9.0]),
            "lcm_values": np.array([4.0, 6.0]),
        },
    )

    np.testing.assert_allclose(result["gcds"], [6, 3])
    np.testing.assert_allclose(result["lcms"], [12, 6])


def test_gcd_rejects_non_integer_floats():
    evaluator = build_evaluator(
        expressions={"res": func("gcd", symbol("x"), const(1))},
        inputs=["x"],
        target="res",
    )

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({"x": 1.5})
    assert_runtime_cause(excinfo, ValueError, "gcd requires integer arguments")


def test_integer_helpers_reject_infinity_with_value_error():
    evaluator = build_evaluator(
        expressions={"res": func("factorial", symbol("x"))},
        inputs=["x"],
        target="res",
    )

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({"x": float("inf")})
    assert_runtime_cause(excinfo, ValueError, "factorial requires integer arguments")


def test_v05_function_expansion_vectorizes_common_helpers():
    expressions = {
        "trig": func("sin", symbol("angles")),
        "logs": func("log2", symbol("values")),
        "roots": func("cbrt", symbol("cubes")),
        "clamped": func("clamp", symbol("raw"), const(0), const(5)),
        "gcds": func("gcd", symbol("ints"), const(6)),
        "combs": func("combinations", symbol("totals"), const(2)),
        "variance": func("variance", symbol("a"), symbol("b"), symbol("c")),
        "std": func("std", symbol("a"), symbol("b"), symbol("c")),
        "res": object_node(
            trig=symbol("trig"),
            logs=symbol("logs"),
            roots=symbol("roots"),
            clamped=symbol("clamped"),
            gcds=symbol("gcds"),
            combs=symbol("combs"),
            variance=symbol("variance"),
            std=symbol("std"),
        ),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["angles", "values", "cubes", "raw", "ints", "totals", "a", "b", "c"],
        target="res",
    )

    result = evaluator(
        {
            "angles": np.array([0, math.pi / 2]),
            "values": np.array([2, 8]),
            "cubes": np.array([-8, 27]),
            "raw": np.array([-1, 2, 10]),
            "ints": np.array([6, 9]),
            "totals": np.array([5, 6]),
            "a": np.array([1, 2]),
            "b": np.array([2, 4]),
            "c": np.array([3, 6]),
        },
    )

    np.testing.assert_allclose(result["trig"], [0, 1])
    np.testing.assert_allclose(result["logs"], [1, 3])
    np.testing.assert_allclose(result["roots"], [-2, 3])
    np.testing.assert_allclose(result["clamped"], [0, 2, 5])
    np.testing.assert_allclose(result["gcds"], [6, 3])
    np.testing.assert_allclose(result["combs"], [10, 15])
    np.testing.assert_allclose(result["variance"], [1, 4])
    np.testing.assert_allclose(result["std"], [1, 2])


@pytest.mark.parametrize(
    "expression,expected",
    [
        (func("add", const(2), const(3)), 5),
        (func("subtract", const(7), const(2)), 5),
        (func("multiply", const(4), const(3)), 12),
        (func("divide", const(10), const(2)), 5),
        (func("pow", const(2), const(5)), 32),
        (func("mod", const(17), const(5)), 2),
        (func("larger", const(4), const(3)), True),
        (func("smallerEq", const(3), const(3)), True),
        (func("equal", op("add", const(0.1), const(0.2)), const(0.3)), True),
        (func("unequal", const(4), const(3)), True),
        (func("xor", const(True), const(False)), True),
        (func("not", const(False)), True),
        (func("unaryMinus", const(5)), -5),
        (func("unaryPlus", const(-5)), -5),
    ],
)
def test_function_node_operator_aliases(expression, expected):
    evaluator = build_evaluator(
        expressions={"res": expression},
        inputs=[],
        target="res",
    )
    assert evaluator({}) == expected


def test_function_node_logical_aliases_preserve_lazy_semantics():
    expressions = {
        "and_result": func("and", const(False), op("divide", const(1), const(0))),
        "or_result": func("or", const(True), op("divide", const(1), const(0))),
        "nullish_result": func("nullish", const(2), op("divide", const(1), const(0))),
        "res": array(
            symbol("and_result"),
            symbol("or_result"),
            symbol("nullish_result"),
        ),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    assert evaluator({}) == [False, True, 2]


def test_accessor_node_uses_mathjs_one_based_indices():
    expressions = {"res": accessor(symbol("data"), const(1))}
    evaluator = build_evaluator(expressions=expressions, inputs=["data"], target="res")
    assert evaluator({"data": [10, 20, 30]}) == 10


def test_accessor_node_accepts_symbolic_index():
    expressions = {"res": accessor(symbol("vec"), symbol("i"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["vec", "i"],
        target="res",
    )
    assert evaluator({"vec": [10, 20, 30], "i": 2}) == 20


@pytest.mark.parametrize("bad_index", [0, -1])
def test_accessor_node_rejects_indices_below_one(bad_index):
    expressions = {"res": accessor(symbol("vec"), symbol("i"))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["vec", "i"],
        target="res",
    )

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({"vec": [10, 20, 30], "i": bad_index})
    assert_runtime_cause(excinfo, ValueError, "IndexNode indices must be >= 1")


def test_accessor_node_handles_numpy_multidimensional_indices():
    expressions = {"res": accessor(symbol("matrix"), const(2), const(3))}
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["matrix"],
        target="res",
    )
    assert evaluator({"matrix": np.array([[1, 2, 3], [4, 5, 6]])}) == 6


def test_accessor_node_handles_inclusive_range_indices():
    expressions = {"res": accessor(symbol("vec"), range_node(const(2), const(4)))}
    evaluator = build_evaluator(expressions=expressions, inputs=["vec"], target="res")
    assert evaluator({"vec": [10, 20, 30, 40, 50]}) == [20, 30, 40]


@pytest.mark.parametrize(
    "bad_range",
    [
        range_node(const(0), const(2)),
        range_node(const(1), const(0)),
    ],
)
def test_accessor_node_rejects_range_indices_below_one(bad_range):
    expressions = {"res": accessor(symbol("vec"), bad_range)}
    evaluator = build_evaluator(expressions=expressions, inputs=["vec"], target="res")

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({"vec": [10, 20, 30]})
    assert_runtime_cause(excinfo, ValueError, "RangeNode indices must be >= 1")


def test_range_node_materializes_inclusive_ranges():
    expressions = {
        "forward": range_node(const(1), const(3)),
        "backward": range_node(const(5), const(1), const(-2)),
        "res": object_node(forward=symbol("forward"), backward=symbol("backward")),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    result = evaluator({})
    np.testing.assert_allclose(result["forward"], [1, 2, 3])
    np.testing.assert_allclose(result["backward"], [5, 3, 1])


def test_range_node_rejects_zero_step():
    expressions = {"res": range_node(const(1), const(3), const(0))}
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")
    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({})
    assert_runtime_cause(excinfo, ValueError, "step cannot be zero")


@pytest.mark.parametrize(
    "expression",
    [
        func("sin", const(1), const(2)),
        func("factorial", const(5), const(6)),
        func("log"),
        func("log", const(1), const(2), const(3)),
        func("log1p", const(1), const(2), const(3)),
        func("log2", const(1), const(2)),
        func("round"),
        func("permutations"),
    ],
)
def test_function_arity_gaps_are_rejected_at_compile_time(expression):
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions={"res": expression}, inputs=[], target="res")


def test_object_node_and_new_nodes_collect_dependencies():
    expressions = {
        "res": object_node(
            picked=accessor(symbol("data"), symbol("i")),
            sequence=range_node(symbol("start"), symbol("end")),
            total=func("add", symbol("x"), const(1)),
        ),
    }
    evaluator = build_evaluator(
        expressions=expressions,
        inputs=["data", "i", "start", "end", "x", "unused"],
        target="res",
    )

    assert evaluator.__mathjs_required_inputs__ == ("data", "end", "i", "start", "x")
    result = evaluator({"data": [8, 9], "i": 2, "start": 2, "end": 4, "x": 10})
    assert result["picked"] == 9
    assert result["total"] == 11
    np.testing.assert_allclose(result["sequence"], [2, 3, 4])


def test_block_node_remains_unsupported():
    expressions = {"res": {"type": "BlockNode", "blocks": []}}
    with pytest.raises(InvalidNodeError):
        build_evaluator(expressions=expressions, inputs=[], target="res")


def test_log_optional_base_broadcasts_over_arrays():
    evaluator = build_evaluator(
        expressions={"res": func("log", symbol("x"), symbol("base"))},
        inputs=["x", "base"],
        target="res",
    )

    np.testing.assert_allclose(
        evaluator({"x": np.array([8, 27, 10000]), "base": np.array([2, 3, 10])}),
        [3, 3, 4],
    )


@pytest.mark.parametrize(
    "name",
    ["set", "Mapping", "scope", "_missing", "_extra", "_mj_sum"],
)
def test_collision_prone_input_identifiers_are_allowed(name):
    evaluator = build_evaluator(
        expressions={"res": symbol(name)},
        inputs=[name],
        target="res",
    )

    assert evaluator({name: 7}) == 7


@pytest.mark.parametrize(
    "name",
    ["set", "Mapping", "scope", "_missing", "_extra", "_mj_sum"],
)
def test_collision_prone_expression_identifiers_are_allowed(name):
    evaluator = build_evaluator(
        expressions={
            name: func("sum", const(1), const(2)),
            "res": op("add", symbol(name), const(4)),
        },
        inputs=[],
        target="res",
    )

    assert evaluator({}) == 7


def test_reserved_internal_prefix_is_rejected_for_inputs():
    with pytest.raises(InvalidNodeError, match="reserved internal prefix"):
        build_evaluator(
            expressions={"res": const(1)},
            inputs=["__mj_input"],
            target="res",
        )


def test_reserved_internal_prefix_is_rejected_for_expression_ids():
    with pytest.raises(InvalidNodeError, match="reserved internal prefix"):
        build_evaluator(
            expressions={"__mj_expr": const(1), "res": const(2)},
            inputs=[],
            target="res",
        )


def test_reserved_internal_prefix_is_rejected_for_targets():
    with pytest.raises(InvalidNodeError, match="reserved internal prefix"):
        build_evaluator(
            expressions={"res": const(1)},
            inputs=[],
            target="__mj_target",
        )


def test_round_supports_scalar_and_negative_decimals():
    expressions = {
        "one": func("round", const(1.2345), const(2)),
        "two": func("round", const(1234.5), const(-2)),
        "res": array(symbol("one"), symbol("two")),
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")

    np.testing.assert_allclose(evaluator({}), [1.23, 1200])


def test_round_broadcasts_array_decimals():
    evaluator = build_evaluator(
        expressions={"res": func("round", symbol("x"), symbol("places"))},
        inputs=["x", "places"],
        target="res",
    )

    np.testing.assert_allclose(
        evaluator(
            {
                "x": np.array([1.234, 5.678, 9.876]),
                "places": np.array([1, 2, -1]),
            },
        ),
        [1.2, 5.68, 10],
    )
    np.testing.assert_allclose(
        evaluator({"x": 1.2345, "places": np.array([1, 3])}),
        [1.2, 1.234],
    )


def test_round_rejects_non_integer_scalar_decimals_at_runtime():
    evaluator = build_evaluator(
        expressions={"res": func("round", symbol("x"), symbol("places"))},
        inputs=["x", "places"],
        target="res",
    )

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({"x": 1.2345, "places": 1.5})
    assert_runtime_cause(excinfo, ValueError, "round requires integer arguments")


def test_non_finite_raw_number_constants_are_supported():
    evaluator = build_evaluator(
        expressions={
            "pos": {
                "type": "ConstantNode",
                "value": float("inf"),
                "valueType": "number",
            },
            "neg": {
                "type": "ConstantNode",
                "value": float("-inf"),
                "valueType": "number",
            },
            "nan": {
                "type": "ConstantNode",
                "value": float("nan"),
                "valueType": "number",
            },
            "res": array(symbol("pos"), symbol("neg"), symbol("nan")),
        },
        inputs=[],
        target="res",
    )

    result = evaluator({})
    assert math.isinf(result[0])
    assert result[0] > 0
    assert math.isinf(result[1])
    assert result[1] < 0
    assert math.isnan(result[2])


def test_non_finite_string_number_constants_are_supported():
    evaluator = build_evaluator(
        expressions={
            "pos": const("Infinity"),
            "neg": const("-inf"),
            "nan": const("NaN"),
            "res": array(symbol("pos"), symbol("neg"), symbol("nan")),
        },
        inputs=[],
        target="res",
    )

    result = evaluator({})
    assert math.isinf(result[0])
    assert result[0] > 0
    assert math.isinf(result[1])
    assert result[1] < 0
    assert math.isnan(result[2])


def test_runtime_errors_preserve_failing_expression_name_and_cause():
    evaluator = build_evaluator(
        expressions={
            "bad": op("divide", const(1), const(0)),
            "res": op("add", symbol("bad"), const(1)),
        },
        inputs=[],
        target="res",
    )

    with pytest.raises(RuntimeEvaluationError) as excinfo:
        evaluator({})
    assert excinfo.value.expression == "bad"
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_compiled_globals_keep_builtin_escape_hatches_unreachable():
    evaluator = build_evaluator(expressions={"res": const(1)}, inputs=[], target="res")

    assert evaluator.__globals__["__builtins__"] == {}
    for name in ("__import__", "eval", "getattr"):
        assert name not in evaluator.__globals__
    assert "_mj_sum" not in evaluator.__globals__
    assert "__mj_sum" in evaluator.__globals__


@pytest.mark.parametrize("fn_name", ["__import__", "eval", "getattr"])
def test_builtin_escape_function_calls_are_rejected(fn_name):
    with pytest.raises(InvalidNodeError):
        build_evaluator(
            expressions={"res": func(fn_name, const(1))},
            inputs=[],
            target="res",
        )
