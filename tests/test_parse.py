import json
import math

import pytest

from mathjs_to_func import build_evaluator
from mathjs_to_func.parse import (
    ConstantNode,
    OperatorNode,
    SymbolNode,
    parse,
    parse_payload,
)


def test_parse_constant_node():
    payload = '{"type": "ConstantNode", "value": "5", "valueType": "number"}'
    result = parse(payload)
    assert result == {
        "type": "ConstantNode",
        "value": "5",
        "valueType": "number",
    }


def test_parse_string_constant_node():
    payload = '{"type": "ConstantNode", "value": "fixed", "valueType": "string"}'
    result = parse(payload)
    assert result == {
        "type": "ConstantNode",
        "value": "fixed",
        "valueType": "string",
    }


def test_parse_symbol_node_from_mathjs_alias():
    payload = json.dumps({"mathjs": "SymbolNode", "name": "x"})
    result = parse(payload)
    assert result == {"type": "SymbolNode", "name": "x"}


def test_parse_operator_node_supports_nested_args():
    payload = json.dumps(
        {
            "type": "OperatorNode",
            "fn": "add",
            "args": [
                {"type": "ConstantNode", "value": "1", "valueType": "number"},
                {
                    "type": "ParenthesisNode",
                    "content": {
                        "type": "OperatorNode",
                        "fn": "multiply",
                        "args": [
                            {
                                "type": "ConstantNode",
                                "value": "2",
                                "valueType": "number",
                            },
                            {
                                "type": "ConstantNode",
                                "value": "3",
                                "valueType": "number",
                            },
                        ],
                    },
                },
            ],
        },
    )
    result = parse(payload)
    assert result["type"] == "OperatorNode"
    assert result["fn"] == "add"
    assert len(result["args"]) == 2
    inner = result["args"][1]["content"]
    assert inner["type"] == "OperatorNode"


def test_parse_function_node_works_with_build_evaluator():
    payload = json.dumps(
        {
            "type": "FunctionNode",
            "fn": "sum",
            "args": [
                {"type": "SymbolNode", "name": "a"},
                {"type": "ConstantNode", "value": "5", "valueType": "number"},
            ],
        },
    )
    expression = parse(payload)

    evaluator = build_evaluator(
        expressions={"result": expression},
        inputs=["a"],
        target="result",
    )

    assert evaluator({"a": 7}) == 12


def test_parse_conditional_node_works_with_build_evaluator():
    payload = json.dumps(
        {
            "type": "ConditionalNode",
            "condition": {
                "type": "OperatorNode",
                "fn": "larger",
                "args": [
                    {"type": "SymbolNode", "name": "x"},
                    {"type": "ConstantNode", "value": "0", "valueType": "number"},
                ],
            },
            "trueExpr": {"type": "SymbolNode", "name": "x"},
            "falseExpr": {"type": "ConstantNode", "value": "0", "valueType": "number"},
        },
    )
    expression = parse(payload)

    evaluator = build_evaluator(
        expressions={"result": expression},
        inputs=["x"],
        target="result",
    )

    assert evaluator({"x": 5}) == 5
    assert evaluator({"x": -1}) == 0


def test_parse_relational_node_works_with_build_evaluator():
    payload = json.dumps(
        {
            "type": "RelationalNode",
            "conditionals": ["smaller", "smallerEq"],
            "params": [
                {"type": "ConstantNode", "value": "10", "valueType": "number"},
                {"type": "SymbolNode", "name": "x"},
                {"type": "ConstantNode", "value": "20", "valueType": "number"},
            ],
        },
    )
    expression = parse(payload)

    evaluator = build_evaluator(
        expressions={"result": expression},
        inputs=["x"],
        target="result",
    )

    assert evaluator({"x": 15}) is True
    assert evaluator({"x": 25}) is False


def test_parse_invalid_payload_raises_value_error():
    with pytest.raises(ValueError, match=r"Invalid math.js JSON payload"):
        parse('{"type": "UnknownNode", "value": 1}')


def test_parse_accessor_index_and_range_nodes():
    payload = json.dumps(
        {
            "type": "AccessorNode",
            "object": {"type": "SymbolNode", "name": "data"},
            "index": {
                "type": "IndexNode",
                "dimensions": [
                    {
                        "type": "RangeNode",
                        "start": {
                            "type": "ConstantNode",
                            "value": "1",
                            "valueType": "number",
                        },
                        "end": {
                            "type": "ConstantNode",
                            "value": "3",
                            "valueType": "number",
                        },
                    },
                ],
            },
        },
    )

    result = parse(payload)

    assert result["type"] == "AccessorNode"
    assert result["index"]["type"] == "IndexNode"
    assert result["index"]["dimensions"][0]["type"] == "RangeNode"


def test_parse_object_node_properties():
    payload = json.dumps(
        {
            "mathjs": "ObjectNode",
            "properties": {
                "total": {
                    "mathjs": "FunctionNode",
                    "fn": "add",
                    "args": [
                        {"mathjs": "SymbolNode", "name": "x"},
                        {"mathjs": "ConstantNode", "value": "1", "valueType": "number"},
                    ],
                },
            },
        },
    )

    result = parse(payload)

    assert result["type"] == "ObjectNode"
    assert result["properties"]["total"]["type"] == "FunctionNode"


def test_parse_rejects_standalone_index_node():
    payload = json.dumps(
        {
            "type": "IndexNode",
            "dimensions": [
                {"type": "SymbolNode", "name": "i"},
            ],
        },
    )

    with pytest.raises(ValueError, match=r"Invalid math.js JSON payload"):
        parse(payload)


def test_parse_non_finite_constants_work_with_build_evaluator():
    expressions = {
        "pos": parse('{"type":"ConstantNode","value":Infinity,"valueType":"number"}'),
        "nan": parse('{"type":"ConstantNode","value":NaN,"valueType":"number"}'),
        "res": {
            "type": "ArrayNode",
            "items": [
                {"type": "SymbolNode", "name": "pos"},
                {"type": "SymbolNode", "name": "nan"},
            ],
        },
    }
    evaluator = build_evaluator(expressions=expressions, inputs=[], target="res")

    result = evaluator({})
    assert math.isinf(result[0])
    assert math.isnan(result[1])


def test_parse_payload_validates_complete_envelope_with_multiple_targets():
    payload = json.dumps(
        {
            "expressions": {
                "total": {
                    "type": "OperatorNode",
                    "fn": "add",
                    "args": [
                        {"type": "SymbolNode", "name": "x"},
                        {"type": "ConstantNode", "value": "2", "valueType": "number"},
                    ],
                },
                "double": {
                    "type": "OperatorNode",
                    "fn": "multiply",
                    "args": [
                        {"type": "SymbolNode", "name": "total"},
                        {"type": "ConstantNode", "value": "2", "valueType": "number"},
                    ],
                },
            },
            "inputs": ["x"],
            "target": ["total", "double"],
        },
    )

    parsed = parse_payload(payload)
    evaluator = build_evaluator(payload=parsed)

    assert evaluator({"x": 40}) == {"total": 42, "double": 84}


def test_public_pydantic_models_can_build_expression_payloads():
    expression = OperatorNode(
        fn="add",
        args=[
            SymbolNode(name="x"),
            ConstantNode(value="2", valueType="number"),
        ],
    )

    evaluator = build_evaluator(
        expressions={"res": expression.as_ast()},
        inputs=["x"],
        target="res",
    )

    assert evaluator({"x": 40}) == 42


@pytest.mark.parametrize("mathjs_type", ["BigNumber", "Complex", "Fraction", "Unit"])
def test_parse_rejects_mathjs_replacer_values_with_clear_error(mathjs_type):
    payload = json.dumps({"mathjs": mathjs_type, "value": "2"})

    with pytest.raises(
        ValueError,
        match=f"Unsupported math.js serialized value type: {mathjs_type}",
    ):
        parse(payload)


def test_parse_payload_rejects_nested_mathjs_replacer_values_with_clear_error():
    payload = json.dumps(
        {
            "expressions": {
                "res": {
                    "type": "ConstantNode",
                    "value": {"mathjs": "Complex", "re": 2, "im": 3},
                    "valueType": "number",
                },
            },
            "inputs": [],
            "target": "res",
        },
    )

    with pytest.raises(
        ValueError,
        match=r"Unsupported math.js serialized value type: Complex",
    ):
        parse_payload(payload)
