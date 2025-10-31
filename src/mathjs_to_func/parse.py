"""Optional math.js JSON parser powered by Pydantic."""

from __future__ import annotations

from json import JSONDecodeError
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

__all__ = ["parse"]


class _MathjsBaseModel(BaseModel):
    """Base model that preserves unknown fields and aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    def as_ast(self) -> dict[str, Any]:
        """Return the validated node as a plain mapping."""
        return self.model_dump(by_alias=True, exclude_none=True)


class ConstantNode(_MathjsBaseModel):
    type: Literal["ConstantNode"] = Field(
        default="ConstantNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    value: bool | int | float | str | None
    valueType: Literal["number", "boolean", "null"] | None = None


class SymbolNode(_MathjsBaseModel):
    type: Literal["SymbolNode"] = Field(
        default="SymbolNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    name: str


class ParenthesisNode(_MathjsBaseModel):
    type: Literal["ParenthesisNode"] = Field(
        default="ParenthesisNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    content: MathjsExpression | None = None
    expr: MathjsExpression | None = None

    @model_validator(mode="after")
    def _ensure_child(self) -> ParenthesisNode:
        if self.content is None and self.expr is None:
            raise ValueError("ParenthesisNode requires 'content' or 'expr'")
        return self


class OperatorNode(_MathjsBaseModel):
    type: Literal["OperatorNode"] = Field(
        default="OperatorNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    args: list[MathjsExpression]
    fn: str | MathjsExpression
    op: str | None = None


class FunctionNode(_MathjsBaseModel):
    type: Literal["FunctionNode"] = Field(
        default="FunctionNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    fn: str | MathjsExpression
    args: list[MathjsExpression] = Field(default_factory=list)


class ArrayNode(_MathjsBaseModel):
    type: Literal["ArrayNode"] = Field(
        default="ArrayNode",
        validation_alias=AliasChoices("type", "mathjs"),
        serialization_alias="type",
    )
    items: list[MathjsExpression] = Field(default_factory=list)


MathjsExpression = (
    ConstantNode
    | SymbolNode
    | ParenthesisNode
    | OperatorNode
    | FunctionNode
    | ArrayNode
)


for model in (
    ConstantNode,
    SymbolNode,
    ParenthesisNode,
    OperatorNode,
    FunctionNode,
    ArrayNode,
):
    model.model_rebuild()


_NODE_ADAPTER = TypeAdapter(MathjsExpression)


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
    try:
        node = _NODE_ADAPTER.validate_json(source)
    except (ValidationError, JSONDecodeError) as exc:
        raise ValueError("Invalid math.js JSON payload") from exc
    if not isinstance(node, _MathjsBaseModel):
        raise TypeError("Unexpected node type produced by parser")
    return node.as_ast()
