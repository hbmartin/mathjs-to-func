import { create, all } from "mathjs";
import { performance } from "node:perf_hooks";

const math = create(all);

function median(values) {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return sorted[middle];
  }
  return (sorted[middle - 1] + sorted[middle]) / 2;
}

function roundTripCompile(expression) {
  const parsed = math.parse(expression);
  const serialized = JSON.stringify(parsed, math.replacer);
  const revived = JSON.parse(serialized, math.reviver);
  return revived.compile();
}

function measure(iterations, repeats, callback) {
  const timings = [];
  let result;
  for (let repeat = 0; repeat < repeats; repeat += 1) {
    const start = performance.now();
    for (let index = 0; index < iterations; index += 1) {
      result = callback();
    }
    timings.push((performance.now() - start) / 1000 / iterations);
  }
  return { secondsPerOp: median(timings), result };
}

let input = "";
for await (const chunk of process.stdin) {
  input += chunk;
}

const config = JSON.parse(input);
const repeats = config.repeats ?? 5;
const results = [];

for (const item of config.cases) {
  if (!item.mathjsExpression) {
    continue;
  }

  const build = measure(item.buildIterations, repeats, () =>
    roundTripCompile(item.mathjsExpression),
  );
  const compiled = roundTripCompile(item.mathjsExpression);
  const evaluate = measure(item.iterations, repeats, () =>
    compiled.evaluate(item.scope),
  );

  results.push({
    case: item.name,
    phase: "roundtrip_compile",
    runner: "mathjs",
    seconds_per_op: build.secondsPerOp,
  });
  results.push({
    case: item.name,
    phase: "reusable_call",
    runner: "mathjs",
    seconds_per_op: evaluate.secondsPerOp,
  });
}

process.stdout.write(`${JSON.stringify(results)}\n`);
