# math.js Compatibility Matrix

This project targets serialized math.js expression trees, not the whole math.js runtime. The supported surface is intentionally smaller than math.js and optimized for safe, reusable Python callables over scalar and NumPy-array inputs.

Primary math.js references:

- https://mathjs.org/docs/expressions/syntax.html
- https://mathjs.org/docs/expressions/expression_trees.html
- https://mathjs.org/docs/reference/functions.html
- https://mathjs.org/docs/reference/constants.html

## Node Support

| Node | Status | Notes |
|------|--------|-------|
| `ConstantNode` | Supported | `number`, `boolean`, and `null`. Numeric constants may include `Infinity`, `-Infinity`, and `NaN`. String constants are not supported. |
| `SymbolNode` | Supported | Resolves inputs, other expression ids, and common built-in constants. Inputs/expressions override built-ins. |
| `OperatorNode` | Supported subset | Arithmetic, unary plus/minus, logical, relational, and `nullish`. See operator table below. |
| `FunctionNode` | Supported subset | Whitelisted numeric/statistical/nullish helpers plus operator function aliases. Custom functions and raw argument functions are not supported. |
| `ParenthesisNode` | Supported | Delegates to child content. |
| `ArrayNode` | Supported | Produces Python lists; helpers convert to NumPy arrays when needed. |
| `ConditionalNode` | Supported | Scalar branches are lazy. Array conditions evaluate both branches and use NumPy `where`. |
| `RelationalNode` | Supported | Chained comparisons such as `10 < x <= 50`; scalar comparisons short-circuit. |
| `AccessorNode` | Supported subset | Read-only numeric indexing via `IndexNode`; math.js 1-based indices are translated to Python 0-based indices. Dot/property access is not supported. |
| `AssignmentNode` | Not supported | Evaluators are pure and do not mutate scope. |
| `BlockNode` | Not supported | Multi-statement result sets are out of scope. |
| `FunctionAssignmentNode` | Not supported | User-defined functions are out of scope. |
| `IndexNode` | Supported subset | Supported inside `AccessorNode` with scalar numeric dimensions and `RangeNode` dimensions. |
| `ObjectNode` | Supported | Produces Python dict literals with string keys and supported expression values. |
| `RangeNode` | Supported | Materializes inclusive ranges via NumPy; optional non-zero step is supported. |

## Operators

| math.js function name | Status | Notes |
|-----------------------|--------|-------|
| `add`, `subtract`, `multiply`, `divide`, `pow`, `mod` | Supported | Uses Python/NumPy arithmetic. Also accepted in `FunctionNode` form. |
| `unaryPlus`, `unaryMinus` | Supported | Uses Python unary operators. |
| `not`, `and`, `or`, `xor` | Supported | Scalar `and`/`or` are lazy; arrays vectorize. Also accepted in `FunctionNode` form. |
| `equal`, `unequal`, `larger`, `largerEq`, `smaller`, `smallerEq` | Supported | Numeric comparisons use math.js-style default tolerances: `relTol=1e-12`, `absTol=1e-15`; override per evaluator with `EvalConfig` or `{"epsilon": ...}`. Also accepted in `FunctionNode` form. |
| `nullish` | Supported | Lazily evaluates fallback for scalar values in both operator and `FunctionNode` alias form. |
| `dotMultiply`, `dotDivide`, `dotPow` | Not supported | NumPy broadcasting covers many array use cases, but math.js dot operator semantics are not separately modeled. |
| `bitAnd`, `bitOr`, `bitXor`, `bitNot`, shifts | Not supported | Bitwise operators are out of scope. |
| Postfix factorial, transpose, percentage postfix | Not supported | These parser operators are not translated today. Function-style `factorial(n)` is supported. |
| `to`, `in` | Not supported | Units/conversions are out of scope. |

## Functions

| Function | Status | Notes |
|----------|--------|-------|
| `abs`, `ceil`, `exp`, `floor`, `log`, `log1p`, `log2`, `log10`, `round`, `sign`, `sqrt`, `cbrt` | Supported | Uses NumPy unary helpers. |
| `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | Supported | Uses NumPy trigonometric and hyperbolic helpers. |
| `hypot`, `clamp`, `factorial`, `gcd`, `lcm`, `combinations`, `permutations` | Supported | Scalars and NumPy-friendly inputs are supported where practical. |
| `min`, `max`, `sum`, `mean`, `median`, `variance`, `std`, `mode` | Supported | Scalars and arrays are supported. `variance` and `std` use unbiased sample normalization (`ddof=1`). |
| `ifnull` | Supported | Project helper; treats `None` and `NaN` as nullish. |
| `nullish` | Supported | Alias for the lazy `nullish` operator in `FunctionNode` form. |
| Probability, linear algebra, set, bitwise, unit, complex helper functions | Not supported | Add selectively as use cases appear. |
| `evaluate`, `parse`, `simplify`, `derivative`, `resolve`, `import`, `createUnit`, `reviver` | Not supported | Deliberately excluded from generated evaluators for safety and scope control. |

## Constants

| Constant | Status | Python value |
|----------|--------|--------------|
| `e`, `E` | Supported | `math.e` |
| `pi`, `PI` | Supported | `math.pi` |
| `tau` | Supported | `math.tau` |
| `phi` | Supported | Golden ratio |
| `LN2`, `LN10`, `LOG2E`, `LOG10E` | Supported | `math.log`/`math.log2`/`math.log10` equivalents |
| `SQRT1_2`, `SQRT2` | Supported | Square-root constants |
| `Infinity` | Supported | `math.inf`; supported as either a symbol or numeric constant literal |
| `NaN` | Supported | `math.nan`; supported as either a symbol or numeric constant literal |
| `i` | Supported | Python `1j` |
| `null`, `undefined` | Supported | `None` |
| `version` | Not supported | A compiled Python evaluator does not know the source math.js runtime version. |

## Data Types

| Type | Status | Notes |
|------|--------|-------|
| Python `int`/`float`/`bool`/`None` | Supported | Primary scalar types. |
| NumPy arrays/scalars | Supported | Used for vectorized workloads. |
| Python `complex` | Partial | Built-in `i` works for Python arithmetic; complex-specific math.js functions are not implemented. |
| math.js `BigNumber`, `Fraction`, `Unit`, `Matrix`, sparse matrix | Not supported | `parse`/`parse_payload` reject math.js replacer values such as `BigNumber`, `Fraction`, and `Unit` with explicit unsupported-value errors. Add via helper-layer extensions if needed. |

## Evaluation Model

- Evaluators are pure functions over a provided mapping; math.js assignment and parser scope mutation are not implemented.
- Unknown symbols fail at compile time unless they are declared inputs, expression ids, or supported built-in constants.
- Generated code executes with a deliberately small globals dictionary and no ambient builtins.
- User-provided identifiers may not use the reserved `__mj_` prefix, which is used for generated runtime internals.
- `include_source=True` emits a Python source string with the import preamble needed to re-execute `_compiled` when `mathjs_to_func` is installed.
