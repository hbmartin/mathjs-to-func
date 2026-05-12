// Reads expressions.js, parses each entry with math.js, and writes one
// JSON fixture file per group containing { name, expression, scope,
// mathjs_json, expected }. Run via `node generate.js`.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { all, create } from "mathjs";

import groups from "./expressions.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURE_DIR = join(HERE, "fixtures");

const math = create(all);
math.config({ number: "number", matrix: "Array" });

// `ifnull` / `nullish` are project-specific helpers in mathjs-to-func, not
// native math.js functions. Register matching semantics so math.js can serve
// as the evaluation oracle: return the first argument unless it is null,
// undefined, or NaN.
function _isNullish(value) {
  if (value === null || value === undefined) return true;
  if (typeof value === "number" && Number.isNaN(value)) return true;
  return false;
}
const ifnull = (a, b) => (_isNullish(a) ? b : a);
math.import({ ifnull, nullish: ifnull }, { override: true });

function encodeNumber(value) {
  if (Number.isNaN(value)) return { __special__: "nan" };
  if (value === Infinity) return { __special__: "inf" };
  if (value === -Infinity) return { __special__: "-inf" };
  return value;
}

function encode(value) {
  if (typeof value === "number") return encodeNumber(value);
  if (Array.isArray(value)) return { __array__: value.map(encode) };
  if (value && typeof value.valueOf === "function" && value.valueOf() !== value) {
    return encode(value.valueOf());
  }
  return value;
}

mkdirSync(FIXTURE_DIR, { recursive: true });

for (const [groupName, items] of Object.entries(groups)) {
  const out = items.map(({ name, expression, scope }) => {
    const node = math.parse(expression);
    const mathjsJson = JSON.parse(JSON.stringify(node));
    const evaluated = node.evaluate(scope ?? {});
    return {
      name,
      expression,
      scope: scope ?? {},
      mathjs_json: mathjsJson,
      expected: encode(evaluated),
    };
  });
  const target = join(FIXTURE_DIR, `${groupName}.json`);
  writeFileSync(target, `${JSON.stringify(out, null, 2)}\n`);
  console.log(`wrote ${target} (${out.length} cases)`);
}
