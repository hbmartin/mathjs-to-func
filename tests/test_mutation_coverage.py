# ruff: noqa: SLF001
import math

import pytest

from mathjs_to_func import InvalidNodeError, build_evaluator, to_string
from mathjs_to_func.ast_builder import MathJsAstBuilder
from mathjs_to_func.parse import expression_json_schema


def _builder() -> MathJsAstBuilder:
    return MathJsAstBuilder(
        expression_name="res",
        helper_names={},
        local_names=set(),
    )


@pytest.mark.parametrize(
    "fn,value,expected",
    [
        ("not", True, False),
        ("percentage", 12, 0.12),
        ("unaryMinus", 5, -5),
        ("unaryPlus", -5, -5),
    ],
)
def test_compile_time_unary_operator_evaluation(fn, value, expected):
    assert _builder()._evaluate_unary_operator(fn, value) == expected


def test_compile_time_unary_operator_rejects_unknown_operator():
    with pytest.raises(InvalidNodeError, match="Unsupported unary operator"):
        _builder()._evaluate_unary_operator("bitNot", 1)


@pytest.mark.parametrize(
    "fn,left,right,expected",
    [
        ("add", 2, 3, 5),
        ("subtract", 7, 2, 5),
        ("multiply", 4, 3, 12),
        ("divide", 9, 2, 4.5),
        ("pow", 2, 5, 32),
        ("mod", 17, 5, 2),
        ("larger", 4, 3, True),
        ("largerEq", 3, 3, True),
        ("smaller", 2, 3, True),
        ("smallerEq", 3, 3, True),
        ("equal", 3, 3, True),
        ("unequal", 3, 4, True),
        ("xor", True, False, True),
    ],
)
def test_compile_time_binary_operator_evaluation(fn, left, right, expected):
    assert _builder()._evaluate_binary_operator(fn, left, right) == expected


@pytest.mark.parametrize("fn", [None, "bitAnd"])
def test_compile_time_binary_operator_rejects_unknown_operator(fn):
    with pytest.raises(InvalidNodeError, match="Unsupported binary operator"):
        _builder()._evaluate_binary_operator(fn, 1, 2)


def test_standalone_index_node_is_rejected_by_builder():
    with pytest.raises(
        InvalidNodeError,
        match="IndexNode is only supported inside AccessorNode",
    ):
        build_evaluator(
            expressions={
                "res": {
                    "type": "IndexNode",
                    "dimensions": [
                        {"type": "ConstantNode", "value": "1", "valueType": "number"},
                    ],
                },
            },
            inputs=[],
            target="res",
        )


@pytest.mark.parametrize(
    "node,expected",
    [
        ({"type": "ConstantNode", "value": "true", "valueType": "boolean"}, "true"),
        ({"type": "ConstantNode", "value": False, "valueType": "boolean"}, "false"),
        ({"type": "ConstantNode", "value": None, "valueType": "null"}, "null"),
        ({"type": "ConstantNode", "value": None}, "null"),
        ({"type": "ConstantNode", "value": "label", "valueType": "string"}, '"label"'),
        ({"type": "ConstantNode", "value": "plain"}, '"plain"'),
        ({"type": "ConstantNode", "value": "1e3"}, "1e3"),
        ({"type": "ConstantNode", "value": "not-a-number"}, '"not-a-number"'),
        ({"type": "ConstantNode", "value": 4.5, "valueType": "number"}, "4.5"),
    ],
)
def test_introspection_constant_text_variants(node, expected):
    assert to_string(node) == expected


def test_expression_json_schema_uses_mathjs_aliases():
    schema = expression_json_schema()
    constant_schema = schema["$defs"]["ConstantNode"]

    assert constant_schema["properties"]["type"]["const"] == "ConstantNode"
    assert "valueType" in constant_schema["properties"]
    assert "value_type" not in constant_schema["properties"]


def test_format_exponential_notation_normalizes_exponent():
    evaluator = build_evaluator(
        expressions={
            "res": {
                "type": "FunctionNode",
                "fn": "format",
                "args": [
                    {"type": "SymbolNode", "name": "value"},
                    {
                        "type": "ObjectNode",
                        "properties": {
                            "notation": {
                                "type": "ConstantNode",
                                "value": "exponential",
                                "valueType": "string",
                            },
                            "precision": {
                                "type": "ConstantNode",
                                "value": "3",
                                "valueType": "number",
                            },
                        },
                    },
                ],
            },
        },
        inputs=["value"],
        target="res",
    )

    assert evaluator({"value": 12345}) == "1.23e+4"
    assert evaluator({"value": 0.00123}) == "1.23e-3"


def test_format_engineering_notation_uses_exponent_formatter():
    evaluator = build_evaluator(
        expressions={
            "res": {
                "type": "FunctionNode",
                "fn": "format",
                "args": [
                    {"type": "SymbolNode", "name": "value"},
                    {
                        "type": "ObjectNode",
                        "properties": {
                            "notation": {
                                "type": "ConstantNode",
                                "value": "engineering",
                                "valueType": "string",
                            },
                            "precision": {
                                "type": "ConstantNode",
                                "value": "4",
                                "valueType": "number",
                            },
                        },
                    },
                ],
            },
        },
        inputs=["value"],
        target="res",
    )

    assert evaluator({"value": 12345}) == "12.35e+3"
    assert evaluator({"value": -0.00123}) == "-1.23e-3"


def test_format_exponential_handles_zero_without_log10_domain_error():
    evaluator = build_evaluator(
        expressions={
            "res": {
                "type": "FunctionNode",
                "fn": "format",
                "args": [
                    {"type": "SymbolNode", "name": "value"},
                    {
                        "type": "ObjectNode",
                        "properties": {
                            "notation": {
                                "type": "ConstantNode",
                                "value": "exponential",
                                "valueType": "string",
                            },
                            "precision": {
                                "type": "ConstantNode",
                                "value": "2",
                                "valueType": "number",
                            },
                        },
                    },
                ],
            },
        },
        inputs=["value"],
        target="res",
    )

    assert evaluator({"value": 0}) == "0e+0"
    assert evaluator({"value": math.nan}) == "NaN"
