"""
run_suite.py -- Full evaluation-suite orchestrator.

Runs the complete evaluation suite against ONE RAG pipeline:

    Quality
      ├── Retriever
      ├── Generator
      ├── RAG Pipeline
      └── Application

    Safety
      ├── Toxicity
      ├── Leakage
      ├── Scope
      └── Safety aggregation

    Operational
      ├── Latency
      ├── Cost
      └── Reliability

The pipeline is constructed ONCE and injected into every evaluation
that needs it.

The resulting snapshot is the atomic unit used later for regression
comparison.

Usage:

    python -m evals.run_suite
        -> baselines/candidate.json

    python -m evals.run_suite --baseline
        -> baselines/baseline.json

    python -m evals.run_suite --out my_snapshot.json

    python -m evals.run_suite --label "top_k=3"

    python -m evals.run_suite --quiet

    python -m evals.run_suite --full
        -> include informational metrics
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.reranker import RerankingRetriever

from evals import (
    eval_retriever,
    eval_generator,
    eval_rag_pipeline,
    eval_application,
    eval_safety,
    eval_ops,
)

from evals.metric_registry import rule_for


load_dotenv()


# ============================================================
# 1. CONFIG
# ============================================================

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"


# ============================================================
# 2. HELPERS
# ============================================================

def _slug(name: str) -> str:
    """
    Convert metric names into stable metric-id components.

    Example:
        "Contextual Recall"
        ->
        "contextual_recall"
    """
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def flatten_nested(namespace: str, summary: dict) -> dict:
    """
    Convert nested evaluation output into dotted metric IDs.

    Input:

        {
            "Contextual Recall": {
                "mean": 0.91,
                "pass_rate": 1.0
            }
        }

    Output:

        {
            "retriever.contextual_recall.mean": 0.91,
            "retriever.contextual_recall.pass_rate": 1.0
        }
    """

    flattened = {}

    for metric, stats in summary.items():

        metric_slug = _slug(metric)

        for stat, value in stats.items():

            metric_id = (
                f"{namespace}."
                f"{metric_slug}."
                f"{stat}"
            )

            flattened[metric_id] = value

    return flattened


def prefix_flat(namespace: str, flat: dict) -> dict:
    """
    Prefix already-flat metrics with their namespace.

    Example:

        {
            "latency.e2e_p95_ms": 1200
        }

        ->
        {
            "ops.latency.e2e_p95_ms": 1200
        }
    """

    return {
        f"{namespace}.{key}": value
        for key, value in flat.items()
    }


# ============================================================
# 3. PROVENANCE
# ============================================================

def _git_sha() -> str:
    """
    Return the current short git commit SHA.

    If the project is not inside a git repository, return
    'unknown' instead of failing the evaluation.
    """

    try:

        return subprocess.check_output(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except Exception:

        return "unknown"


def _prompt_hash() -> str:
    """
    Hash the actual generator prompt.

    This means changing the prompt produces a different
    provenance hash even if the code otherwise remains identical.
    """

    try:

        from src.generator import prompt

        prompt_text = str(prompt)

        return hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest()[:12]

    except Exception:

        return "unknown"


def _pipeline_config(rag) -> dict:
    """
    Extract the important pipeline configuration.

    This is intentionally defensive because the exact attributes
    exposed by RagPipeline may evolve.

    Missing values are recorded as 'unknown' rather than guessed.
    """

    config = {}

    config["fetch_k"] = "unknown"
    config["top_k"] = "unknown"

    config["generator_model"] = "unknown"
    config["embedding_model"] = "unknown"
    config["reranker_model"] = "unknown"

    # --------------------------------------------------------
    # Retriever configuration
    # --------------------------------------------------------

    retriever = getattr(rag, "retriever", None)

    if retriever is not None:

        if hasattr(retriever, "fetch_k"):
            config["fetch_k"] = retriever.fetch_k

        if hasattr(retriever, "top_k"):
            config["top_k"] = retriever.top_k

        if hasattr(retriever, "reranker"):

            reranker = retriever.reranker

            model_name = getattr(
                reranker,
                "model",
                None,
            )

            if model_name:
                config["reranker_model"] = str(model_name)

    # --------------------------------------------------------
    # Generator configuration
    # --------------------------------------------------------

    try:

        from src.generator import llm

        model_name = getattr(
            llm,
            "model",
            None,
        )

        if model_name:
            config["generator_model"] = str(model_name)

    except Exception:
        pass

    # --------------------------------------------------------
    # Reranker fallback
    # --------------------------------------------------------

    if config["reranker_model"] == "unknown":

        try:

            from src.reranker import CROSS_ENCODER

            config["reranker_model"] = CROSS_ENCODER

        except Exception:
            pass

    return config


def build_metadata(label: str, rag, full: bool) -> dict:
    """
    Build provenance information for the snapshot.

    A snapshot should tell us:

      - when it ran
      - what code produced it
      - what prompt was used
      - what pipeline configuration was used
      - what experiment label was attached
    """

    pipeline_config = _pipeline_config(rag)

    return {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),

        "git_sha": _git_sha(),

        "prompt_hash": _prompt_hash(),

        "label": label,

        "pipeline": pipeline_config,

        "full": full,
    }


# ============================================================
# 4. RUN FULL SUITE
# ============================================================

def run_suite(
    label: str = "",
    quiet: bool = False,
    full: bool = False,
) -> dict:
    """
    Build ONE pipeline and run the complete evaluation suite.

    Returns:
        snapshot dictionary
    """

    verbose = not quiet

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Build pipeline ONCE
    # --------------------------------------------------------

    print("Building pipeline (once)...")

    rag = RagPipeline()

    # The retriever evaluation must ideally measure the exact
    # retriever used by this RAG pipeline.
    #
    # Only construct a fallback retriever if RagPipeline does not
    # expose one.

    retriever = getattr(
        rag,
        "retriever",
        None,
    )

    if retriever is None:

        print(
            "Warning: RagPipeline does not expose "
            "its retriever; constructing fallback retriever."
        )

        retriever = RerankingRetriever()

    metrics = {}

    # ========================================================
    # QUALITY
    # ========================================================

    print("\n[1/6] Retriever evaluation...")

    retriever_result = eval_retriever.run(
        retriever
    )

    metrics.update(
        flatten_nested(
            "retriever",
            retriever_result,
        )
    )

    # --------------------------------------------------------

    print("\n[2/6] Generator evaluation...")

    generator_result = eval_generator.run()

    metrics.update(
        flatten_nested(
            "generator",
            generator_result,
        )
    )

    # --------------------------------------------------------

    print("\n[3/6] RAG pipeline evaluation...")

    pipeline_result = eval_rag_pipeline.run(
        rag
    )

    metrics.update(
        flatten_nested(
            "pipeline",
            pipeline_result,
        )
    )

    # --------------------------------------------------------

    print("\n[4/6] Application quality evaluation...")

    application_result = eval_application.run(
        rag
    )

    metrics.update(
        flatten_nested(
            "application",
            application_result,
        )
    )

    # ========================================================
    # SAFETY
    # ========================================================

    print("\n[5/6] Safety evaluation...")

    safety_result = eval_safety.run_safety(
        rag,
        verbose=verbose,
    )

    metrics.update(
        prefix_flat(
            "safety",
            safety_result,
        )
    )

    # ========================================================
    # OPERATIONAL
    # ========================================================

    print("\n[6/6] Operational evaluation...")

    ops_result = eval_ops.run_ops(
        rag,
        verbose=verbose,
    )

    metrics.update(
        prefix_flat(
            "ops",
            ops_result,
        )
    )

    # ========================================================
    # FILTER INFO METRICS
    # ========================================================

    elapsed = time.perf_counter() - start_time

    if not full:

        filtered_metrics = {}

        for metric_id, value in metrics.items():

            rule = rule_for(metric_id)

            if rule["kind"] != "info":
                filtered_metrics[metric_id] = value

        metrics = filtered_metrics

    # ========================================================
    # SNAPSHOT
    # ========================================================

    metadata = build_metadata(
        label=label,
        rag=rag,
        full=full,
    )

    metadata["suite_seconds"] = round(
        elapsed,
        1,
    )

    metadata["n_metrics"] = len(
        metrics
    )

    snapshot = {
        "metadata": metadata,
        "metrics": metrics,
    }

    return snapshot


# ============================================================
# 5. WRITE SNAPSHOT
# ============================================================

def write_snapshot(
    snapshot: dict,
    path: str,
) -> str:
    """
    Write snapshot JSON to disk.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            snapshot,
            f,
            indent=2,
            sort_keys=True,
        )

    return path


# ============================================================
# 6. PRINT SNAPSHOT
# ============================================================

def print_snapshot(
    snapshot: dict,
) -> None:
    """
    Print a human-readable snapshot summary.
    """

    metadata = snapshot["metadata"]
    metrics = snapshot["metrics"]

    print("\n" + "=" * 78)
    print("EVALUATION SNAPSHOT")
    print("=" * 78)

    print(
        f"label        : "
        f"{metadata.get('label') or '(none)'}"
    )

    print(
        f"created_at   : "
        f"{metadata['created_at']}"
    )

    print(
        f"git_sha      : "
        f"{metadata['git_sha']}"
    )

    print(
        f"prompt_hash  : "
        f"{metadata['prompt_hash']}"
    )

    pipeline = metadata.get(
        "pipeline",
        {},
    )

    print(
        f"fetch_k      : "
        f"{pipeline.get('fetch_k', 'unknown')}"
    )

    print(
        f"top_k        : "
        f"{pipeline.get('top_k', 'unknown')}"
    )

    print(
        f"generator    : "
        f"{pipeline.get('generator_model', 'unknown')}"
    )

    print(
        f"reranker     : "
        f"{pipeline.get('reranker_model', 'unknown')}"
    )

    print(
        f"suite_time   : "
        f"{metadata['suite_seconds']}s"
    )

    print(
        f"metrics      : "
        f"{len(metrics)}"
    )

    print("-" * 78)

    for key in sorted(metrics):

        value = metrics[key]

        if isinstance(value, float):

            shown = f"{value:.4f}"

        else:

            shown = str(value)

        print(
            f"  {key:<48} {shown}"
        )

    print("=" * 78)


# ============================================================
# 7. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete RAG evaluation suite "
            "and write a regression snapshot."
        )
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            f"write to {BASELINE_PATH} "
            "(bless this run as baseline)"
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help="custom snapshot output path",
    )

    parser.add_argument(
        "--label",
        default="",
        help=(
            "human-readable description of "
            "the pipeline change"
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "suppress detailed safety/operational output"
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "include informational metrics such as "
            "avg/min/max/n and secondary operational metrics"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Decide output path
    # --------------------------------------------------------

    if args.out:

        output_path = args.out

    elif args.baseline:

        output_path = BASELINE_PATH

    else:

        output_path = CANDIDATE_PATH

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    snapshot = run_suite(
        label=args.label,
        quiet=args.quiet,
        full=args.full,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_snapshot(
        snapshot
    )

    # --------------------------------------------------------
    # Persist
    # --------------------------------------------------------

    path = write_snapshot(
        snapshot,
        output_path,
    )

    print(
        f"\nWrote snapshot -> {path}"
    )


if __name__ == "__main__":
    main()