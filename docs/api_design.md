# mathjs-to-func API Design

## Public Function

```
from mathjs_to_func import build_evaluator

func, source = build_evaluator(
    expressions=...,  # dict[str, dict]
    inputs=...,       # Iterable[str]
    target=...,       # str
    include_source=True,
)
```

- Returns a Python callable plus optional source preview string.
- Callable accepts a mapping of input variable names to numeric/array values and returns the computed `target` value.
- Intermediate expressions evaluate once in dependency order.

## Responsibilities

- Validate identifiers, inputs, and required expressions.
- Translate math.js AST JSON into safe Python `ast.AST` nodes.
- Build dependency graph, detect missing references and cycles.
- Generate Python function that reuses NumPy for math ops.
- Provide clear exceptions for invalid payloads.

## Supported math.js Nodes

- `ConstantNode` (numeric) with `valueType` support (number, boolean, null).
- `SymbolNode` (references to inputs/expressions/common math.js constants) with regex validation.
- `OperatorNode` for binary ops (`add`, `subtract`, `multiply`, `divide`, `pow`, `mod`), unary ops (`unaryPlus`, `unaryMinus`, `not`), logical ops (`and`, `or`, `xor`), relational ops (`larger`, `largerEq`, `smaller`, `smallerEq`, `equal`, `unequal`), and `nullish`.
- `ParenthesisNode` (delegates to child).
- `FunctionNode` for whitelisted numeric/statistical helpers, nullish helpers, and operator aliases such as `add(a, b)`.
- `ArrayNode` for list literals.
- `AccessorNode`/`IndexNode` for read-only numeric indexing with math.js 1-based indices translated to Python 0-based indices.
- `RangeNode` for inclusive ranges with optional non-zero step.
- `ObjectNode` for dict literals with string keys.
- `ConditionalNode` for lazy ternary expressions.
- `RelationalNode` for chained comparisons.

## Errors

Custom exception hierarchy under `ExpressionError` with subclasses:
- `MissingTargetError`
- `UnknownIdentifierError`
- `CircularDependencyError`
- `InvalidNodeError`
- `InputValidationError`

Each captures context (expression id, offending field) for UI surfacing.

## Generated Source

Example shape:

```
def compiled(inputs):
    allowed_keys = {...}
    data = {name: inputs[name] for name in allowed_keys}
    expr_a = (data["x"] + data["y"]) * 0.5
    expr_b = np.maximum(expr_a, data["z"])
    return expr_b
```

- Imports NumPy as `np` when needed.
- Maintains deterministic ordering.
- Optionally included in return tuple when `include_source=True`.
```
