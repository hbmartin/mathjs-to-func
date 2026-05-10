# mathjs-to-func

[![PyPI](https://img.shields.io/pypi/v/mathjs-to-func.svg)](https://pypi.org/project/mathjs-to-func/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![CI](https://github.com/hbmartin/mathjs-to-func/actions/workflows/ci.yml/badge.svg)](https://github.com/hbmartin/mathjs-to-func/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/hbmartin/mathjs-to-func)

A tiny Python library that compiles serialized [math.js](https://mathjs.org/) expression trees into fast, reusable Python callables. The generated function respects dependency ordering, validates inputs, and mirrors a practical subset of math.js operators, constants, comparisons, conditionals, and numeric functions.

## Key Features
- Execute without reparsing or repeatedly walking the JSON graph.
- Detect dependency cycles and missing identifiers early.
- Keep execution sandboxed by compiling a controlled Python AST.
- Work well with scalars or NumPy arrays for vectorised workloads.
- Resolve common math.js constants like `pi`, `e`, `tau`, `NaN`, and `Infinity`.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency and virtualenv management. From the repository root:

```bash
uv add mathjs-to-func
```

An optional `parse` extra installs a JSON-to-math.js parser powered by Pydantic:

```bash
uv add mathjs-to-func --extra parse
```

## Compiling A Function

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
| `SymbolNode`            | inputs, expression references, and common built-in constants; identifiers must be alphanumeric/underscore, starting with a letter/underscore |
| `OperatorNode`          | `add`, `subtract`, `multiply`, `divide`, `pow`, `mod`, unary `unaryPlus`, `unaryMinus`, `not`, `and`, `or`, `xor`, comparisons, and `nullish` |
| `FunctionNode`          | Common math.js numeric/statistical helpers, including trig, logs, `clamp`, `hypot`, integer combinatorics, `variance`, `std`, `mode`, `ifnull`, and operator aliases such as `add(a, b)` |
| `ParenthesisNode`       | forwards to the wrapped expression |
| `ArrayNode`             | materialised to Python lists/NumPy arrays |
| `AccessorNode`/`IndexNode` | read-only indexing with math.js 1-based indices translated to Python 0-based indices |
| `RangeNode`             | materialised to inclusive NumPy ranges with optional non-zero step |
| `ObjectNode`            | materialised to Python dict literals with string keys |
| `ConditionalNode`       | lazy scalar ternary evaluation, vectorised NumPy `where` for arrays |
| `RelationalNode`        | chained comparisons like `10 < x <= 50`, with scalar short-circuiting |

Unknown node types, invalid identifiers, or disallowed functions raise `InvalidNodeError` during compilation.

See [docs/compatibility.md](docs/compatibility.md) for the fuller math.js compatibility matrix and known gaps.

### Error handling

- `ExpressionError`: base class for configuration mistakes.
- `MissingTargetError`: requested target id does not exist.
- `UnknownIdentifierError`: an expression references a symbol that is neither an input nor another expression.
- `CircularDependencyError`: dependency graph contains a cycle.
- `InvalidNodeError`: AST contains unsupported structures or invalid literals.
- `InputValidationError`: the compiled function received inputs that are missing, unexpected, or not a mapping.

All exceptions provide enough context (`expression` name, offending identifier, cycle list, etc.) to surface descriptive UI errors.

## Parsing math.js JSON

With the extra installed you can turn serialized math.js nodes into evaluator-ready mappings:

```python
from mathjs_to_func import build_evaluator
from mathjs_to_func.parse import parse

expression = parse(
    """{
    "type": "OperatorNode",
    "fn": "add",
    "args": [
        {"type": "SymbolNode", "name": "x"},
        {"type": "ConstantNode", "value": "2", "valueType": "number"}
    ]
}"""
)

evaluator = build_evaluator(
    expressions={"total": expression},
    inputs=["x"],
    target="total",
)

result = evaluator({"x": 40})  # -> 42
```

All examples below assume commands are wrapped with `uv run ...` to execute inside the managed environment.

## CLI

Compile a payload file and inspect the generated Python source without writing a script:

```bash
uv run python -m mathjs_to_func compile payload.json --target z --emit-source
```

Without `--emit-source`, the command validates the payload and prints metadata JSON containing the target, required inputs, and evaluation order. Use `-` as the payload path to read JSON from stdin.

## JSON Schema

Export JSON Schema for frontend validation of serialized math.js payloads:

```bash
uv run python -m mathjs_to_func schema --output dist/mathjs-to-func.schema.json
```

The default schema covers a complete evaluator payload (`expressions`, `inputs`, and `target`). Use `--kind expression` to export the schema for a single math.js expression tree.

## Implementation Notes

1. **AST translation** – `MathJsAstBuilder` walks the math.js JSON and emits Python `ast.AST` nodes. Identifiers are validated via a strict regex to prevent sneaky names like `__import__`.
2. **Dependency graph** – A topological sorter (`graphlib.TopologicalSorter`) runs over expression references to produce a safe evaluation order while catching cycles and missing references upfront.
3. **Code generation** – The generated function validates the provided scope, binds required inputs to local variables, evaluates expressions in order, and returns the target. Intermediate values are stored as local variables named after their expression id.
4. **Execution sandbox** – The compiled module is executed with a tightly scoped globals dictionary: helper math functions, NumPy, and a few safe built-ins only. There is no ambient `__builtins__` exposure.
5. **Helper functions** – math.js functions map onto small Python helpers for arithmetic, comparison, logical, nullish, and statistics behavior. Equality and ordering use math.js-style default tolerances for numeric round-off.

## Testing

Run the full suite with:

```bash
uv run pytest
```

Run the benchmark suite locally with:

```bash
npm ci --prefix bench/js
uv run python -m bench
```

CI runs `uv run python -m bench --check` as a relative perf regression gate. The benchmark compares reusable `build_evaluator` call performance with Python `eval`, `simpleeval`, and a Node math.js parse → JSON round-trip → compile path across scalar arithmetic, conditional, helper-heavy, and NumPy payloads.

Run mutation testing with:

```bash
uv run mutmut run
uv run mutmut results
```

The GitHub mutation workflow runs on source and test changes, records the full mutmut result set, and emits a warning when any mutants survive.

The tests cover operator translation, helper semantics, dependency validation, error conditions, numpy-friendly behaviour, and public API ergonomics.

## Project Structure

```
src/mathjs_to_func/
├── __init__.py          # build_evaluator public API and export list
├── ast_builder.py       # math.js JSON → Python AST translation
├── compiler.py          # dependency graph, code generation, compilation
├── errors.py            # structured exception hierarchy
├── helpers.py           # runtime helpers for math.js-compatible functions/operators
└── py.typed             # PEP 561 marker for type-aware consumers
```

Additional documentation lives in `docs/api_design.md`, outlining the initial design considerations.

## Limitations & Future Work

- Only a subset of math.js functions/operators are implemented today; see the compatibility matrix for specifics.
- Units, user-defined functions, and incremental recomputation are intentionally out of scope for this milestone.
- Arrays are handled via NumPy; if you need bigints, complex numbers, or matrices, the helper layer will require extension.

Contributions and bug reports are welcome!
