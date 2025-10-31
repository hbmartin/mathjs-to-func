import json

import pytest

from mathjs_to_func import build_evaluator
from mathjs_to_func.parse import parse


def test_parse_constant_node():
    payload = '{"type": "ConstantNode", "value": "5", "valueType": "number"}'
    result = parse(payload)
    assert result == {
        "type": "ConstantNode",
        "value": "5",
        "valueType": "number",
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


def test_parse_invalid_payload_raises_value_error():
    with pytest.raises(ValueError, match=r"Invalid math.js JSON payload"):
        parse('{"type": "UnknownNode", "value": 1}')
