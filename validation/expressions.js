// Curated math.js expressions used to seed oracle fixtures.
//
// Each entry is { name, expression, scope }. `name` becomes part of the
// pytest test ID. `scope` is the input mapping passed to `node.evaluate()`
// and to the generated Python evaluator.
//
// NaN / Infinity literals are rejected by the Python library at compile
// time (ast_builder._to_number), so non-finite values are surfaced through
// SymbolNode references (the bare keywords `NaN` and `Infinity` resolve to
// math.js / Python built-in constants on both sides).

export default {
  arithmetic: [
    { name: "add_simple", expression: "2 + 3", scope: {} },
    { name: "subtract_with_inputs", expression: "a - b", scope: { a: 10, b: 4 } },
    { name: "multiply_simple", expression: "4 * 5", scope: {} },
    { name: "divide_fractional", expression: "10 / 4", scope: {} },
    { name: "pow_integer", expression: "2 ^ 8", scope: {} },
    { name: "mod_simple", expression: "19 mod 5", scope: {} },
  ],

  unary: [
    { name: "unary_minus", expression: "-x", scope: { x: 7 } },
    { name: "unary_plus", expression: "+x", scope: { x: -3 } },
    { name: "logical_not", expression: "not flag", scope: { flag: true } },
  ],

  logical: [
    { name: "logical_and_truthy", expression: "a and b", scope: { a: true, b: true } },
    { name: "logical_or_short_circuit", expression: "a or b", scope: { a: false, b: true } },
    { name: "logical_xor", expression: "a xor b", scope: { a: true, b: false } },
  ],

  nullish: [
    { name: "ifnull_function", expression: "ifnull(a, 5)", scope: { a: null } },
    { name: "nullish_function", expression: "nullish(a, 5)", scope: { a: 7 } },
  ],

  relational: [
    { name: "smaller", expression: "a < b", scope: { a: 1, b: 2 } },
    { name: "smallerEq", expression: "a <= b", scope: { a: 5, b: 5 } },
    { name: "larger", expression: "a > b", scope: { a: 9, b: 2 } },
    { name: "largerEq", expression: "a >= b", scope: { a: 5, b: 5 } },
    { name: "equal", expression: "a == b", scope: { a: 4, b: 4 } },
    { name: "unequal", expression: "a != b", scope: { a: 1, b: 2 } },
  ],

  relational_chained: [
    { name: "between_inclusive", expression: "1 < x <= 10", scope: { x: 7 } },
  ],

  conditional: [
    { name: "abs_via_ternary", expression: "x > 0 ? x : -x", scope: { x: -4 } },
  ],

  array_aggregates: [
    { name: "sum_array_literal", expression: "sum([1, 2, 3])", scope: {} },
    { name: "mean_array_inputs", expression: "mean([a, b, c])", scope: { a: 2, b: 4, c: 6 } },
    { name: "min_variadic", expression: "min(1, 2, 3)", scope: {} },
    { name: "max_array_literal", expression: "max([5, 9])", scope: {} },
    { name: "median_array_literal", expression: "median([1, 2, 3, 4])", scope: {} },
  ],

  functions_unary: [
    { name: "abs_negative", expression: "abs(-7)", scope: {} },
    { name: "ceil_simple", expression: "ceil(1.2)", scope: {} },
    { name: "floor_simple", expression: "floor(1.8)", scope: {} },
    { name: "round_two_decimals", expression: "round(2.345, 2)", scope: {} },
    { name: "sqrt_two", expression: "sqrt(2)", scope: {} },
    { name: "exp_one", expression: "exp(1)", scope: {} },
    { name: "log_e", expression: "log(e)", scope: {} },
    { name: "sign_negative", expression: "sign(-3)", scope: {} },
  ],

  constants: [
    { name: "pi_double", expression: "pi * 2", scope: {} },
    { name: "tau_quarter", expression: "tau / 4", scope: {} },
    { name: "e_to_one", expression: "e ^ 1", scope: {} },
    { name: "ln2_plus_ln10", expression: "LN2 + LN10", scope: {} },
    { name: "sqrt_of_sqrt2", expression: "sqrt(SQRT2)", scope: {} },
  ],

  mixed: [
    {
      name: "compound_arithmetic",
      expression: "(a + b) * sqrt(c) - log(d)",
      scope: { a: 1, b: 3, c: 16, d: Math.E },
    },
    {
      name: "min_plus_max",
      expression: "min(a, b) + max(a, b)",
      scope: { a: 7, b: 3 },
    },
  ],
};
