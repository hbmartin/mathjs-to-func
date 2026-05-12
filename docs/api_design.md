# mathjs-to-func API Design

## Public Function

```
from mathjs_to_func import build_evaluator

func = build_evaluator(
    expressions=...,  # dict[str, dict]
    inputs=...,       # Iterable[str]
    target=...,       # str | Sequence[str]
    config=...,       # optional EvalConfig or mapping
    compile_cache=...,  # optional canonical JSON LRU cache
    include_source=True,
)
source = func.__mathjs_source__
```

- Returns a Python callable.
- When `include_source=True`, the generated source preview is attached to the callable as `__mathjs_source__`.
- Callable accepts a mapping of input variable names to numeric/array values and returns the computed `target` value, or a `dict[str, Any]` for multiple targets.
- Intermediate expressions evaluate once in dependency order.

## Responsibilities

- Validate identifiers, inputs, and required expressions.
- Translate math.js AST JSON into safe Python `ast.AST` nodes.
- Build dependency graph, detect missing references and cycles.
- Generate Python function that reuses NumPy for math ops.
- Bind per-evaluator comparison tolerances through the helper bundle.
- Provide clear exceptions for invalid payloads.

## Supported math.js Nodes

- `ConstantNode` (numeric, including non-finite math.js literals) with `valueType` support (number, boolean, null).
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
- `RuntimeEvaluationError`

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

- Includes a preamble importing the generated helper bindings when source output is requested.
- Maintains deterministic ordering.
- Optionally attached as `__mathjs_source__` when `include_source=True`.
```
