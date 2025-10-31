# mathjs-to-func

A tiny Python library that compiles serialized [math.js](https://mathjs.org/) expression trees into fast, reusable Python callables. The generated function respects dependency ordering, validates inputs, and mirrors a subset of math.js operators (`+`, `-`, `*`, `/`, `^`, `%`, unary plus/minus) and functions (`min`, `max`, `sum`, `ifnull`).

## Why

Front-ends often rely on math.js for authoring formulas. Shipping those formulas to the backend as plain strings forces the server to reparse and interpret unsafe text. mathjs-to-func lets you send the math.js **serialized AST** instead, then compiles it into a safe Python function that can be executed repeatedly with minimal overhead.

Key goals:
- Execute without reparsing or repeatedly walking the JSON graph.
- Detect dependency cycles and missing identifiers early.
- Keep execution sandboxed by compiling a controlled Python AST.
- Work well with scalars or NumPy arrays for vectorised workloads.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency and virtualenv management. From the repository root:

```bash
uv sync  # create the virtual environment declared in uv.lock
```

All examples below assume commands are wrapped with `uv run ...` to execute inside the managed environment.

## Public API

```python
from mathjs_to_func import build_evaluator

def main():
    mathjs_payload = {
        "expressions": {
            # z = (x + y) / 2
            "sum_xy": {
                "type": "OperatorNode",
                "fn": "add",
                "args": [
                    {"type": "SymbolNode", "name": "x"},
                    {"type": "SymbolNode", "name": "y"},
                ],
            },
            "mean": {
                "type": "OperatorNode",
                "fn": "divide",
                "args": [
                    {"type": "SymbolNode", "name": "sum_xy"},
                    {"type": "ConstantNode", "value": "2", "valueType": "number"},
                ],
            },
        },
        "inputs": ["x", "y"],
        "target": "mean",
    }

    evaluator = build_evaluator(**mathjs_payload, include_source=True)

    result = evaluator({"x": 10, "y": 6})
    print(result)  # -> 8.0

    # Introspection helpers
    print(evaluator.__mathjs_required_inputs__)     # ('x', 'y')
    print(evaluator.__mathjs_evaluation_order__)    # ('sum_xy', 'mean')
    print(evaluator.__mathjs_source__)              # Generated Python source
```

### Parameters

`build_evaluator` accepts keyword parameters (or a single `payload` mapping containing the same keys):

| Argument      | Type                               | Description |
|---------------|------------------------------------|-------------|
| `expressions` | `Mapping[str, Mapping[str, Any]]`   | math.js AST JSON keyed by expression id. Each id becomes a local variable in the compiled function. |
| `inputs`      | `Iterable[str]`                     | Whitelisted identifiers that may be supplied when the function is invoked. |
| `target`      | `str`                               | Name of the expression whose computed value should be returned. |
| `include_source` | `bool` (optional)                 | Attach the generated Python source code as `__mathjs_source__` on the returned callable. |

The returned callable always expects a single mapping argument with the provided inputs. It returns the evaluated `target` value and may be reused across invocations.

### Supported math.js nodes

| Node                     | Notes |
|-------------------------|-------|
| `ConstantNode`          | numeric (`number`), boolean, or `null` literals |
| `SymbolNode`            | validated identifiers; must be alphanumeric/underscore, starting with a letter/underscore |
| `OperatorNode`          | `add`, `subtract`, `multiply`, `divide`, `pow`, `mod`, unary `unaryPlus`, `unaryMinus` |
| `FunctionNode`          | `min`, `max`, `sum`, `ifnull` |
| `ParenthesisNode`       | forwards to the wrapped expression |
| `ArrayNode`             | materialised to Python lists/NumPy arrays |

Unknown node types, invalid identifiers, or disallowed functions raise `InvalidNodeError` during compilation.

### Error handling

- `ExpressionError`: base class for configuration mistakes.
- `MissingTargetError`: requested target id does not exist.
- `UnknownIdentifierError`: an expression references a symbol that is neither an input nor another expression.
- `CircularDependencyError`: dependency graph contains a cycle.
- `InvalidNodeError`: AST contains unsupported structures or invalid literals.
- `InputValidationError`: the compiled function received inputs that are missing, unexpected, or not a mapping.

All exceptions provide enough context (`expression` name, offending identifier, cycle list, etc.) to surface descriptive UI errors.

## Implementation Notes

1. **AST translation** – `MathJsAstBuilder` walks the math.js JSON and emits Python `ast.AST` nodes. Identifiers are validated via a strict regex to prevent sneaky names like `__import__`.
2. **Dependency graph** – A topological sorter (`graphlib.TopologicalSorter`) runs over expression references to produce a safe evaluation order while catching cycles and missing references upfront.
3. **Code generation** – The generated function validates the provided scope, binds required inputs to local variables, evaluates expressions in order, and returns the target. Intermediate values are stored as local variables named after their expression id.
4. **Execution sandbox** – The compiled module is executed with a tightly scoped globals dictionary: helper math functions, NumPy, and a few safe built-ins only. There is no ambient `__builtins__` exposure.
5. **Helper functions** – math.js functions map onto small Python helpers (`_mj_min`, `_mj_max`, `_mj_sum`, `_mj_ifnull`) that understand scalars and NumPy arrays.

## Testing

Run the full suite (178 tests) with:

```bash
uv run pytest
```

The tests cover operator translation, helper semantics, dependency validation, error conditions, numpy-friendly behaviour, and public API ergonomics.

## Project Structure

```
src/mathjs_to_func/
├── __init__.py          # build_evaluator public API and export list
├── ast_builder.py       # math.js JSON → Python AST translation
├── compiler.py          # dependency graph, code generation, compilation
├── errors.py            # structured exception hierarchy
├── helpers.py           # runtime helpers for min/max/sum/ifnull
└── py.typed             # PEP 561 marker for type-aware consumers
```

Additional documentation lives in `docs/api_design.md`, outlining the initial design considerations.

## Limitations & Future Work

- Only a subset of math.js functions/operators are implemented today.
- Units, user-defined functions, and incremental recomputation are intentionally out of scope for this milestone.
- Arrays are handled via NumPy; if you need bigints, complex numbers, or matrices, the helper layer will require extension.

Contributions and bug reports are welcome!
