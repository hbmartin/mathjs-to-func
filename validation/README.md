# Oracle validation against math.js

This subproject cross-validates `mathjs-to-func` against the upstream
[math.js](https://mathjs.org/) library by using math.js as the ground-truth
oracle. A curated list of expressions is parsed with `math.parse()`, serialized
with `node.toJSON()` (the native AST↔JSON form), and evaluated with
`node.evaluate(scope)`. Both the AST JSON and the expected value are written to
`fixtures/*.json`. The Python test module `tests/test_oracle.py` loads those
fixtures and asserts that `build_evaluator(...)` produces matching results
within math.js' default numeric tolerances.

The Python library accepts math.js' native discriminator field name (`mathjs`)
in addition to its own (`type`), so no translation layer is needed —
`JSON.stringify(math.parse(expr))` is consumable directly.

## Regenerating fixtures

```bash
cd validation
npm ci
node generate.js
```

This writes one JSON file per expression group into `validation/fixtures/`.
Commit any changes alongside `package-lock.json` so CI (which is Python-only)
can run the oracle tests without Node.

## Adding new cases

Edit `expressions.js` and add an entry under the appropriate group. Each entry
is `{ name, expression, scope }`:

- `name` — short identifier used in pytest IDs (e.g. `"add_simple"`).
- `expression` — the math.js expression string.
- `scope` — input bindings (object). Use `null`/`Number.NaN`/`Infinity` as
  literals; the generator encodes them in JSON-safe form for round-trip.

Then re-run `node generate.js` and commit the updated fixture.

## Attribution

Test inputs and the oracle library are from
[math.js](https://github.com/josdejong/mathjs) by Jos de Jong, licensed under
the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
