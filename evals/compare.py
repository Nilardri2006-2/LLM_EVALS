"""
compare.py -- Regression decision engine.

Reads two snapshots produced by run_suite.py:

    baseline.json
          │
          │ compare
          ↓
    candidate.json

Every metric is resolved through metric_registry.rule_for().

Each metric is classified as:

    improved
    flat
    regressed
    blocked
    new
    dropped
    info

Overall verdict:

    PASS
        No gate blocked and no guardrail regressed.

    REVIEW
        At least one guardrail regressed beyond tolerance.

    FAIL
        At least one hard gate regressed.

Exit codes:

    PASS   = 0
    FAIL   = 1
    REVIEW = 2

Usage:

    python -m evals.compare

    python -m evals.compare \
        --baseline baselines/baseline.json \
        --candidate baselines/candidate.json

    python -m evals.compare --all
"""

import argparse
import json
import math
import sys
from collections import Counter

from evals.metric_registry import rule_for


# ============================================================
# 1. CONFIG
# ============================================================

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"

EXIT_CODE = {
    "PASS": 0,
    "FAIL": 1,
    "REVIEW": 2,
}


# ============================================================
# 2. LOAD
# ============================================================

def load(path: str) -> dict:
    """
    Load an evaluation snapshot from JSON.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# 3. VALUE HELPERS
# ============================================================

def _is_number(value) -> bool:
    """
    True only for finite numeric values.

    bool is deliberately excluded because bool is a subclass
    of int in Python.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not (
            isinstance(value, float)
            and (
                math.isnan(value)
                or math.isinf(value)
            )
        )
    )


# ============================================================
# 4. CLASSIFY ONE METRIC
# ============================================================

def classify(
    metric_id: str,
    baseline_value,
    candidate_value,
    rule: dict,
) -> dict:
    """
    Compare one baseline/candidate metric pair.

    Returns a structured classification row.
    """

    row = {
        "id": metric_id,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "kind": rule["kind"],
        "direction": rule["direction"],
        "delta": None,
        "tolerance": None,
        "status": None,
    }

    # ========================================================
    # STRUCTURAL CHANGES
    # ========================================================

    # Metric exists only in candidate.
    if baseline_value is None:

        row["status"] = "new"

        return row

    # Metric existed in baseline but disappeared from candidate.
    if candidate_value is None:

        row["status"] = "dropped"

        return row

    # ========================================================
    # INFO METRICS
    # ========================================================

    # Informational metrics are never allowed to influence
    # the release verdict.

    if rule["kind"] == "info":

        if (
            _is_number(baseline_value)
            and _is_number(candidate_value)
        ):

            row["delta"] = (
                candidate_value
                - baseline_value
            )

        row["status"] = "info"

        return row

    # ========================================================
    # BOOLEAN METRICS
    # ========================================================

    if (
        rule.get("bool")
        or isinstance(baseline_value, bool)
        or isinstance(candidate_value, bool)
    ):

        if baseline_value == candidate_value:

            row["status"] = "flat"

        elif (
            baseline_value is False
            and candidate_value is True
        ):

            row["status"] = "improved"

        else:

            if rule["kind"] == "gate":
                row["status"] = "blocked"
            else:
                row["status"] = "regressed"

        return row

    # ========================================================
    # INVALID / NON-NUMERIC VALUES
    # ========================================================

    if not (
        _is_number(baseline_value)
        and _is_number(candidate_value)
    ):

        # Don't crash the entire comparison because a metric
        # contains an unexpected value.

        row["status"] = "info"

        return row

    # ========================================================
    # NUMERIC COMPARISON
    # ========================================================

    delta = (
        candidate_value
        - baseline_value
    )

    row["delta"] = delta

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    higher_is_better = (
        rule["direction"] == "higher"
    )

    if higher_is_better:

        improved = delta > 0

    else:

        improved = delta < 0

    # An improvement is always safe.
    if improved:

        row["status"] = "improved"

        return row

    # --------------------------------------------------------
    # Tolerance
    # --------------------------------------------------------

    worsening = abs(delta)

    absolute_tolerance = rule.get(
        "tol",
        0.0,
    )

    relative_tolerance = (
        rule.get("rel_tol", 0.0)
        * abs(baseline_value)
    )

    tolerance = max(
        absolute_tolerance,
        relative_tolerance,
    )

    row["tolerance"] = tolerance

    # Within tolerance = measurement noise / flat.
    if worsening <= tolerance:

        row["status"] = "flat"

        return row

    # Beyond tolerance = meaningful regression.
    if rule["kind"] == "gate":

        row["status"] = "blocked"

    else:

        row["status"] = "regressed"

    return row


# ============================================================
# 5. COMPARE TWO SNAPSHOTS
# ============================================================

def compare(
    baseline: dict,
    candidate: dict,
):
    """
    Compare all metrics present in either snapshot.

    Returns:

        verdict, rows
    """

    baseline_metrics = baseline.get(
        "metrics",
        {},
    )

    candidate_metrics = candidate.get(
        "metrics",
        {},
    )

    # Union means structural changes are visible.
    metric_ids = sorted(
        set(baseline_metrics)
        | set(candidate_metrics)
    )

    rows = []

    for metric_id in metric_ids:

        baseline_value = baseline_metrics.get(
            metric_id
        )

        candidate_value = candidate_metrics.get(
            metric_id
        )

        rule = rule_for(
            metric_id
        )

        rows.append(
            classify(
                metric_id,
                baseline_value,
                candidate_value,
                rule,
            )
        )

    # ========================================================
    # OVERALL VERDICT
    # ========================================================

    # Hard safety/quality gates win over everything.
    if any(
        row["status"] == "blocked"
        for row in rows
    ):

        verdict = "FAIL"

    # Otherwise a guardrail regression requires review.
    elif any(
        row["status"] == "regressed"
        for row in rows
    ):

        verdict = "REVIEW"

    else:

        verdict = "PASS"

    return verdict, rows


# ============================================================
# 6. FORMATTING
# ============================================================

def _fmt(value) -> str:
    """
    Format values for the terminal report.
    """

    if isinstance(value, bool):

        return str(value)

    if isinstance(value, float):

        if math.isnan(value):
            return "nan"

        if math.isinf(value):
            return "inf"

        return f"{value:.4g}"

    if value is None:

        return "-"

    return str(value)


# ============================================================
# 7. REPORT
# ============================================================

def print_report(
    verdict: str,
    rows: list,
    show_all: bool = False,
) -> None:
    """
    Print a human-readable regression report.
    """

    status_order = {
        "blocked": 0,
        "regressed": 1,
        "dropped": 2,
        "new": 3,
        "improved": 4,
        "flat": 5,
        "info": 6,
    }

    # --------------------------------------------------------
    # Hide informational metrics unless --all was requested.
    # --------------------------------------------------------

    shown_rows = [
        row
        for row in rows
        if (
            show_all
            or row["kind"] != "info"
        )
    ]

    shown_rows.sort(
        key=lambda row: (
            status_order.get(
                row["status"],
                99,
            ),
            row["id"],
        )
    )

    # ========================================================
    # TABLE
    # ========================================================

    print("=" * 110)

    print(
        f"{'metric':<48}"
        f"{'baseline':>12}"
        f"{'candidate':>12}"
        f"{'delta':>12}"
        f"{'tolerance':>12}"
        f"  status"
    )

    print("-" * 110)

    for row in shown_rows:

        delta = row["delta"]

        if _is_number(delta):

            delta_text = f"{delta:+.4g}"

        else:

            delta_text = ""

        tolerance = row["tolerance"]

        if _is_number(tolerance):

            tolerance_text = f"{tolerance:.4g}"

        else:

            tolerance_text = ""

        if row["status"] in (
            "blocked",
            "regressed",
        ):

            marker = "  <<<"

        else:

            marker = ""

        print(
            f"{row['id']:<48}"
            f"{_fmt(row['baseline']):>12}"
            f"{_fmt(row['candidate']):>12}"
            f"{delta_text:>12}"
            f"{tolerance_text:>12}"
            f"  {row['status']}{marker}"
        )

    # ========================================================
    # COUNTS
    # ========================================================

    counts = Counter(
        row["status"]
        for row in rows
    )

    print("-" * 110)

    ordered_counts = [
        "blocked",
        "regressed",
        "improved",
        "flat",
        "new",
        "dropped",
        "info",
    ]

    summary = "  ".join(
        f"{status}={counts[status]}"
        for status in ordered_counts
        if counts[status]
    )

    print(summary)

    # --------------------------------------------------------
    # Hidden info metrics
    # --------------------------------------------------------

    info_count = counts["info"]

    if (
        not show_all
        and info_count
    ):

        print(
            f"({info_count} info metrics hidden; "
            f"use --all to show them)"
        )

    print("=" * 110)

    # ========================================================
    # VERDICT
    # ========================================================

    banners = {
        "PASS": (
            "PASS   -- no gate blocked and "
            "no guardrail regressed. Safe to promote."
        ),

        "REVIEW": (
            "REVIEW -- a guardrail regressed "
            "beyond tolerance. Human review required."
        ),

        "FAIL": (
            "FAIL   -- a hard gate regressed. "
            "Promotion blocked."
        ),
    }

    print(
        f"VERDICT: {banners[verdict]}"
    )

    print("=" * 110)


# ============================================================
# 8. METADATA REPORT
# ============================================================

def print_metadata(
    baseline: dict,
    candidate: dict,
    baseline_path: str,
    candidate_path: str,
) -> None:
    """
    Print the provenance of both snapshots.
    """

    baseline_meta = baseline.get(
        "metadata",
        {},
    )

    candidate_meta = candidate.get(
        "metadata",
        {},
    )

    print(
        f"baseline  : "
        f"{baseline_meta.get('label') or baseline_path}"
    )

    print(
        f"            git={baseline_meta.get('git_sha', '?')} "
        f"prompt={baseline_meta.get('prompt_hash', '?')}"
    )

    print(
        f"candidate : "
        f"{candidate_meta.get('label') or candidate_path}"
    )

    print(
        f"            git={candidate_meta.get('git_sha', '?')} "
        f"prompt={candidate_meta.get('prompt_hash', '?')}"
    )

    # --------------------------------------------------------
    # Pipeline configuration
    # --------------------------------------------------------

    baseline_pipeline = baseline_meta.get(
        "pipeline",
        {},
    )

    candidate_pipeline = candidate_meta.get(
        "pipeline",
        {},
    )

    if (
        baseline_pipeline
        or candidate_pipeline
    ):

        print("\nPipeline configuration:")

        keys = sorted(
            set(baseline_pipeline)
            | set(candidate_pipeline)
        )

        for key in keys:

            bv = baseline_pipeline.get(
                key,
                "?",
            )

            cv = candidate_pipeline.get(
                key,
                "?",
            )

            marker = (
                "  <-- CHANGED"
                if bv != cv
                else ""
            )

            print(
                f"  {key:<20}"
                f"{str(bv):<30}"
                f"-> {str(cv):<30}"
                f"{marker}"
            )


# ============================================================
# 9. ENTRYPOINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare two RAG evaluation snapshots "
            "and return a regression verdict."
        )
    )

    parser.add_argument(
        "--baseline",
        default=BASELINE_PATH,
        help=(
            f"baseline snapshot "
            f"(default: {BASELINE_PATH})"
        ),
    )

    parser.add_argument(
        "--candidate",
        default=CANDIDATE_PATH,
        help=(
            f"candidate snapshot "
            f"(default: {CANDIDATE_PATH})"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "also display informational metrics"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD
    # ========================================================

    baseline = load(
        args.baseline
    )

    candidate = load(
        args.candidate
    )

    # ========================================================
    # METADATA
    # ========================================================

    print_metadata(
        baseline,
        candidate,
        args.baseline,
        args.candidate,
    )

    # ========================================================
    # COMPARE
    # ========================================================

    verdict, rows = compare(
        baseline,
        candidate,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_report(
        verdict,
        rows,
        show_all=args.all,
    )

    # ========================================================
    # CI EXIT CODE
    # ========================================================

    sys.exit(
        EXIT_CODE[verdict]
    )


if __name__ == "__main__":
    main()