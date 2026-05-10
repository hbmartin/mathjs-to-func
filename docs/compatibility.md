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
| `ConstantNode` | Supported | `number`, `boolean`, and `null`. String constants are not supported. |
| `SymbolNode` | Supported | Resolves inputs, other expression ids, and common built-in constants. Inputs/expressions override built-ins. |
| `OperatorNode` | Supported subset | Arithmetic, unary plus/minus, logical, relational, and `nullish`. See operator table below. |
| `FunctionNode` | Supported subset | Whitelisted numeric/statistical/nullish helpers only. Custom functions and raw argument functions are not supported. |
| `ParenthesisNode` | Supported | Delegates to child content. |
| `ArrayNode` | Supported | Produces Python lists; helpers convert to NumPy arrays when needed. |
| `ConditionalNode` | Supported | Scalar branches are lazy. Array conditions evaluate both branches and use NumPy `where`. |
| `RelationalNode` | Supported | Chained comparisons such as `10 < x <= 50`; scalar comparisons short-circuit. |
| `AccessorNode` | Not supported | Indexing/property access is intentionally out of scope. |
| `AssignmentNode` | Not supported | Evaluators are pure and do not mutate scope. |
| `BlockNode` | Not supported | Multi-statement result sets are out of scope. |
| `FunctionAssignmentNode` | Not supported | User-defined functions are out of scope. |
| `IndexNode` | Not supported | Related to unsupported accessors/subsets. |
| `ObjectNode` | Not supported | Object literals are out of scope. |
| `RangeNode` | Not supported | Ranges are not materialized today. |

## Operators

| math.js function name | Status | Notes |
|-----------------------|--------|-------|
| `add`, `subtract`, `multiply`, `divide`, `pow`, `mod` | Supported | Uses Python/NumPy arithmetic. |
| `unaryPlus`, `unaryMinus` | Supported | Uses Python unary operators. |
| `not`, `and`, `or`, `xor` | Supported | Scalar `and`/`or` are lazy; arrays vectorize. |
| `equal`, `unequal`, `larger`, `largerEq`, `smaller`, `smallerEq` | Supported | Numeric comparisons use math.js-style default tolerances: `relTol=1e-12`, `absTol=1e-15`. |
| `nullish` | Supported | Operator form lazily evaluates fallback for scalar values. |
| `dotMultiply`, `dotDivide`, `dotPow` | Not supported | NumPy broadcasting covers many array use cases, but math.js dot operator semantics are not separately modeled. |
| `bitAnd`, `bitOr`, `bitXor`, `bitNot`, shifts | Not supported | Bitwise operators are out of scope. |
| `factorial`, transpose, percentage postfix | Not supported | These parser operators are not translated today. |
| `to`, `in` | Not supported | Units/conversions are out of scope. |

## Functions

| Function | Status | Notes |
|----------|--------|-------|
| `abs`, `ceil`, `exp`, `floor`, `log`, `round`, `sign`, `sqrt` | Supported | Uses NumPy unary helpers. |
| `min`, `max`, `sum`, `mean`, `median` | Supported | Scalars and arrays are supported. |
| `ifnull` | Supported | Project helper; treats `None` and `NaN` as nullish. |
| `nullish` | Supported | Alias for `ifnull` in `FunctionNode` form. |
| Trigonometry, probability, linear algebra, set, bitwise, unit, complex helper functions | Not supported | Add selectively as use cases appear. |
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
| `Infinity` | Supported | `math.inf` |
| `NaN` | Supported | `math.nan` |
| `i` | Supported | Python `1j` |
| `null`, `undefined` | Supported | `None` |
| `version` | Not supported | A compiled Python evaluator does not know the source math.js runtime version. |

## Data Types

| Type | Status | Notes |
|------|--------|-------|
| Python `int`/`float`/`bool`/`None` | Supported | Primary scalar types. |
| NumPy arrays/scalars | Supported | Used for vectorized workloads. |
| Python `complex` | Partial | Built-in `i` works for Python arithmetic; complex-specific math.js functions are not implemented. |
| math.js `BigNumber`, `Fraction`, `Unit`, `Matrix`, sparse matrix | Not supported | Add via helper-layer extensions if needed. |

## Evaluation Model

- Evaluators are pure functions over a provided mapping; math.js assignment and parser scope mutation are not implemented.
- Unknown symbols fail at compile time unless they are declared inputs, expression ids, or supported built-in constants.
- Generated code executes with a deliberately small globals dictionary and no ambient builtins.
