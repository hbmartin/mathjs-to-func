"""Custom exceptions for the mathjs-to-func package."""

from __future__ import annotations

from typing import Any, Mapping


class ExpressionError(RuntimeError):
    """Base class for expression related failures."""

    def __init__(self, message: str, *, expression: str | None = None):
        self.expression = expression
        super().__init__(message)


class MissingTargetError(ExpressionError):
    """Raised when the requested target expression is absent."""


class UnknownIdentifierError(ExpressionError):
    """Raised when an expression references an unknown symbol."""

    def __init__(
        self,
        message: str,
        *,
        expression: str | None,
        identifier: str,
    ):
        super().__init__(message, expression=expression)
        self.identifier = identifier


class CircularDependencyError(ExpressionError):
    """Raised when a dependency cycle is detected."""

    def __init__(self, message: str, cycle: tuple[str, ...]):
        super().__init__(message)
        self.cycle = cycle


class InvalidNodeError(ExpressionError):
    """Raised when the math.js AST contains an unsupported node."""

    def __init__(
        self,
        message: str,
        *,
        expression: str | None,
        node: Mapping[str, Any] | None = None,
    ):
        super().__init__(message, expression=expression)
        self.node = node


class InputValidationError(ExpressionError):
    """Raised when inputs passed to the compiled function are invalid."""

