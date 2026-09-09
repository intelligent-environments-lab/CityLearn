from typing import Optional

from scripts.ci import perf_smoke


def _case(avg_step_ms: float, p95_step_ms: Optional[float] = None, executed_steps: int = 10):
    p95_step_ms = avg_step_ms if p95_step_ms is None else p95_step_ms
    return {
        "executed_steps": executed_steps,
        "avg_step_ms": avg_step_ms,
        "p95_step_ms": p95_step_ms,
    }


def _report(none_avg: float, end_avg: float, entity_avg: float, entity_p95: Optional[float] = None):
    return {
        "cases": {
            "none": _case(none_avg),
            "end": _case(end_avg),
            "entity_none": _case(entity_avg, p95_step_ms=entity_p95),
        }
    }


def test_entity_ratio_does_not_fail_when_absolute_entity_latency_is_ok():
    report = _report(none_avg=4.0, end_avg=4.2, entity_avg=5.5)

    errors = perf_smoke._validate_absolute_thresholds(
        report,
        none_max_ms=30.0,
        end_max_ms=45.0,
        entity_max_ms=30.0,
        ratio_max=2.0,
        entity_overhead_ratio_max=1.08,
    )

    assert errors == []


def test_entity_ratio_fails_when_entity_latency_is_also_too_high():
    report = _report(none_avg=4.0, end_avg=4.2, entity_avg=35.0, entity_p95=36.0)

    errors = perf_smoke._validate_absolute_thresholds(
        report,
        none_max_ms=30.0,
        end_max_ms=45.0,
        entity_max_ms=30.0,
        ratio_max=2.0,
        entity_overhead_ratio_max=1.08,
    )

    assert any("entity avg_step_ms too high" in error for error in errors)
    assert any("entity/flat ratio too high" in error for error in errors)


def test_baseline_comparison_includes_entity_case():
    current = _report(none_avg=4.0, end_avg=4.1, entity_avg=20.0, entity_p95=21.0)
    baseline = _report(none_avg=4.0, end_avg=4.1, entity_avg=5.0, entity_p95=5.5)

    errors = perf_smoke._compare_to_baseline(
        current,
        baseline,
        regression_ratio=2.0,
        slack_ms=1.0,
    )

    assert any("entity_none avg_step_ms regression" in error for error in errors)
    assert any("entity_none p95_step_ms regression" in error for error in errors)
