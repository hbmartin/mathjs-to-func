"""Cross-validate `mathjs-to-func` against upstream math.js fixtures.

Fixtures under `validation/fixtures/*.json` are produced by running each
curated expression through math.js's native `node.toJSON()` (the AST shape)
and `node.evaluate(scope)` (the ground-truth result). This module loads
every fixture and asserts that `build_evaluator` produces matching values
within math.js's default numeric tolerances.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mathjs_to_func import build_evaluator
from mathjs_to_func.helpers import MATHJS_ABS_TOL, MATHJS_REL_TOL

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "validation" / "fixtures"


_SPECIAL_VALUES = {
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}


def _decode(value: object) -> object:
    if isinstance(value, dict):
        if "__special__" in value:
            return _SPECIAL_VALUES[value["__special__"]]
        if "__array__" in value:
            return [_decode(v) for v in value["__array__"]]
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _load_cases() -> list:
    if not FIXTURE_DIR.is_dir():
        return []
    cases: list = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        cases.extend(
            pytest.param(item, id=f"{path.stem}::{item['name']}")
            for item in json.loads(path.read_text())
        )
    return cases


@pytest.mark.parametrize("case", _load_cases())
def test_matches_mathjs_oracle(case):
    scope = {key: _decode(value) for key, value in case["scope"].items()}
    evaluator = build_evaluator(
        expressions={"r": case["mathjs_json"]},
        inputs=list(scope.keys()),
        target="r",
    )
    actual = evaluator(scope)
    expected = _decode(case["expected"])

    if isinstance(expected, list) or isinstance(actual, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=MATHJS_REL_TOL,
            atol=MATHJS_ABS_TOL,
        )
        return

    if isinstance(expected, float) and math.isnan(expected):
        assert isinstance(actual, float)
        assert math.isnan(actual)
        return

    if isinstance(expected, bool) or isinstance(actual, bool):
        assert bool(actual) is bool(expected)
        return

    assert actual == pytest.approx(
        expected,
        rel=MATHJS_REL_TOL,
        abs=MATHJS_ABS_TOL,
    )
