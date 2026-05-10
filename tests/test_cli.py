import json

from mathjs_to_func.__main__ import main


def test_cli_compile_emit_source(tmp_path, capsys):
    payload = {
        "expressions": {
            "z": {
                "type": "OperatorNode",
                "fn": "add",
                "args": [
                    {"type": "SymbolNode", "name": "x"},
                    {"type": "ConstantNode", "value": "2", "valueType": "number"},
                ],
            },
        },
        "inputs": ["x"],
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["compile", str(payload_path), "--target", "z", "--emit-source"]) == 0

    output = capsys.readouterr().out
    assert "def _compiled" in output
    assert "return z" in output


def test_cli_compile_metadata(tmp_path, capsys):
    payload = {
        "expressions": {
            "z": {
                "type": "OperatorNode",
                "fn": "multiply",
                "args": [
                    {"type": "SymbolNode", "name": "x"},
                    {"type": "ConstantNode", "value": "3", "valueType": "number"},
                ],
            },
        },
        "inputs": ["x"],
        "target": "z",
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["compile", str(payload_path)]) == 0

    metadata = json.loads(capsys.readouterr().out)
    assert metadata == {
        "evaluation_order": ["z"],
        "required_inputs": ["x"],
        "target": "z",
    }


def test_cli_schema_export(tmp_path):
    output_path = tmp_path / "schema" / "mathjs-to-func.schema.json"

    assert main(["schema", "--output", str(output_path)]) == 0

    schema = json.loads(output_path.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "MathjsPayload"
    assert set(schema["properties"]) >= {"expressions", "inputs", "target"}
