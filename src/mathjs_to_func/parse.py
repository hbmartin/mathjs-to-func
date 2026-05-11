"""Optional math.js JSON parser powered by Pydantic."""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
)

__all__ = [
    "AccessorNode",
    "ArrayNode",
    "ConditionalNode",
    "ConstantNode",
    "FunctionNode",
    "IndexNode",
    "MathjsExpression",
    "MathjsPayload",
    "ObjectNode",
    "OperatorNode",
    "ParenthesisNode",
    "RangeNode",
    "RelationalNode",
    "SymbolNode",
    "expression_json_schema",
    "parse",
    "parse_payload",
    "payload_json_schema",
]


class _MathjsBaseModel(BaseModel):
    """Base model that preserves unknown fields and aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def as_ast(self) -> dict[str, Any]:
        """Return the validated node as a plain mapping."""
        return self.model_dump(by_alias=True, exclude_none=True)


class ConstantNode(_MathjsBaseModel):
    """math.js constant literal node."""

    type: Literal["ConstantNode"] = Field(
        default="ConstantNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    value: bool | int | float | str | None
    valueType: Literal["number", "boolean", "null"] | None = None


class SymbolNode(_MathjsBaseModel):
    """math.js symbol reference node."""

    type: Literal["SymbolNode"] = Field(
        default="SymbolNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    name: str


class ParenthesisNode(_MathjsBaseModel):
    """math.js parenthesized expression node."""

    type: Literal["ParenthesisNode"] = Field(
        default="ParenthesisNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    content: MathjsExpression = Field(
        validation_alias=AliasChoices("content", "expr"),
        serialization_alias="content",
    )


class OperatorNode(_MathjsBaseModel):
    """math.js operator expression node."""

    type: Literal["OperatorNode"] = Field(
        default="OperatorNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    args: list[MathjsExpression]
    fn: str | MathjsExpression
    op: str | None = None


class FunctionNode(_MathjsBaseModel):
    """math.js function call node."""

    type: Literal["FunctionNode"] = Field(
        default="FunctionNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    fn: str | MathjsExpression
    args: list[MathjsExpression] = Field(default_factory=list)


class ArrayNode(_MathjsBaseModel):
    """math.js array literal node."""

    type: Literal["ArrayNode"] = Field(
        default="ArrayNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    items: list[MathjsExpression] = Field(default_factory=list)


class RangeNode(_MathjsBaseModel):
    """math.js range expression node."""

    type: Literal["RangeNode"] = Field(
        default="RangeNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    start: MathjsExpression
    end: MathjsExpression
    step: MathjsExpression | None = None


class IndexNode(_MathjsBaseModel):
    """math.js accessor index node."""

    type: Literal["IndexNode"] = Field(
        default="IndexNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    dimensions: list[MathjsExpression] = Field(default_factory=list)
    dotNotation: bool | None = None


class AccessorNode(_MathjsBaseModel):
    """math.js read-only accessor node."""

    type: Literal["AccessorNode"] = Field(
        default="AccessorNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    object: MathjsExpression
    index: IndexNode


class ObjectNode(_MathjsBaseModel):
    """math.js object literal node."""

    type: Literal["ObjectNode"] = Field(
        default="ObjectNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    properties: dict[str, MathjsExpression] = Field(default_factory=dict)


class ConditionalNode(_MathjsBaseModel):
    """math.js ternary conditional node."""

    type: Literal["ConditionalNode"] = Field(
        default="ConditionalNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    condition: MathjsExpression
    trueExpr: MathjsExpression
    falseExpr: MathjsExpression


class RelationalNode(_MathjsBaseModel):
    """math.js chained relational comparison node."""

    type: Literal["RelationalNode"] = Field(
        default="RelationalNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    conditionals: list[
        Literal["smaller", "larger", "smallerEq", "largerEq", "equal", "unequal"]
    ]
    params: list[MathjsExpression]


MathjsExpression = (
    ConstantNode
    | SymbolNode
    | ParenthesisNode
    | OperatorNode
    | FunctionNode
    | ArrayNode
    | RangeNode
    | AccessorNode
    | ObjectNode
    | ConditionalNode
    | RelationalNode
)


class MathjsPayload(_MathjsBaseModel):
    """Evaluator payload containing expressions, inputs, and target metadata."""

    expressions: dict[str, MathjsExpression]
    inputs: list[str]
    target: str | list[str]


for model in (
    ConstantNode,
    SymbolNode,
    ParenthesisNode,
    OperatorNode,
    FunctionNode,
    ArrayNode,
    RangeNode,
    IndexNode,
    AccessorNode,
    ObjectNode,
    ConditionalNode,
    RelationalNode,
    MathjsPayload,
):
    model.model_rebuild()


_NODE_ADAPTER = TypeAdapter(MathjsExpression)
_UNSUPPORTED_MATHJS_VALUE_TYPES = frozenset(
    {
        "BigNumber",
        "Complex",
        "Fraction",
        "Unit",
    },
)


def _decode_json(source: str) -> Any:  # noqa: ANN401
    try:
        return json.loads(source)
    except JSONDecodeError as exc:
        raise ValueError("Invalid math.js JSON payload") from exc


def _reject_unsupported_replacer_values(value: Any) -> None:  # noqa: ANN401
    if isinstance(value, dict):
        marker = value.get("mathjs")
        if marker in _UNSUPPORTED_MATHJS_VALUE_TYPES:
            raise ValueError(f"Unsupported math.js serialized value type: {marker}")
        for child in value.values():
            _reject_unsupported_replacer_values(child)
        return
    if isinstance(value, list):
        for child in value:
            _reject_unsupported_replacer_values(child)


def expression_json_schema() -> dict[str, Any]:
    """Return a JSON Schema for a single supported math.js expression tree."""
    return _NODE_ADAPTER.json_schema(by_alias=True)


def payload_json_schema() -> dict[str, Any]:
    """Return a JSON Schema for a complete evaluator payload."""
    return MathjsPayload.model_json_schema(by_alias=True)


def parse(source: str) -> dict[str, Any]:
    """Parse math.js JSON for a single expression.

    Parameters
    ----------
    source:
        JSON string representing a math.js AST node.

    Returns
    -------
    dict[str, Any]
        A validated mapping compatible with ``build_evaluator`` expressions.

    Raises
    ------
    ValueError
        If the input cannot be decoded or does not match the supported schema.

    """
    data = _decode_json(source)
    _reject_unsupported_replacer_values(data)
    try:
        node = _NODE_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise ValueError("Invalid math.js JSON payload") from exc
    return node.as_ast()


def parse_payload(source: str) -> dict[str, Any]:
    """Parse and validate a complete evaluator payload JSON string."""
    data = _decode_json(source)
    _reject_unsupported_replacer_values(data)
    try:
        payload = MathjsPayload.model_validate(data)
    except ValidationError as exc:
        raise ValueError("Invalid math.js JSON payload") from exc
    return payload.as_ast()
