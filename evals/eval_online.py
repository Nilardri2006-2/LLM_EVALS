"""
online/triad_worker.py

Online evaluation worker.

Reads recent RagPipeline traces from LangSmith,
evaluates the RAG triad:

    1. Faithfulness
    2. Answer Relevancy
    3. Contextual Relevancy

and writes the scores + reasons back to LangSmith
as feedback.

Run:

    python -m online.triad_worker
"""

import os
import time

from dotenv import load_dotenv
from langsmith import Client

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.cohere_judge import CohereJudge


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

# Put this in .env:
#
# LANGCHAIN_PROJECT=cx doubt solver
#
PROJECT = os.getenv(
    "LANGCHAIN_PROJECT",
    "doubt solver",
)

JUDGE_MODEL = "command-a-03-2025"

THRESHOLD = 0.7

# 0.30 = approximately 30% of production traces
# 1.00 = evaluate every trace
SAMPLE_RATE = 0.30

POLL_SECONDS = 60


# ============================================================
# CLIENTS
# ============================================================

client = Client()

judge = CohereJudge(
    model=JUDGE_MODEL
)


# ============================================================
# SAMPLING
# ============================================================

def _sampled(run) -> bool:
    """
    Deterministic sampling.

    The same LangSmith run will always produce the same
    sampling decision.
    """

    bucket = hash(str(run.id)) % 100

    return bucket < SAMPLE_RATE * 100


# ============================================================
# EXISTING FEEDBACK
# ============================================================

def _existing_keys(run) -> set:
    """
    Return feedback keys already attached to this run.

    This prevents re-evaluating the same metric repeatedly.
    """

    feedback = client.list_feedback(
        run_ids=[run.id]
    )

    return {
        item.key
        for item in feedback
    }


# ============================================================
# SCORE ONE TRACE
# ============================================================

def score_recent_traces():
    """
    Find recent RagPipeline traces and evaluate missing
    RAG-triad metrics.
    """

    runs = client.list_runs(
        project_name=PROJECT,
        is_root=True,
        run_type="chain",
    )

    for run in runs:

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        if not _sampled(run):
            continue

        # ----------------------------------------------------
        # Extract trace data
        # ----------------------------------------------------

        outputs = run.outputs or {}
        inputs = run.inputs or {}

        answer = outputs.get(
            "answer"
        )

        context = outputs.get(
            "context"
        )

        query = inputs.get(
            "query",
            ""
        )

        # Nothing useful to evaluate.
        if not answer or not context or not query:
            continue

        # ----------------------------------------------------
        # Already evaluated metrics
        # ----------------------------------------------------

        already = _existing_keys(run)

        # ----------------------------------------------------
        # Build metrics
        # ----------------------------------------------------

        jobs = [

            (
                "faithfulness",

                FaithfulnessMetric(
                    threshold=THRESHOLD,
                    model=judge,
                    include_reason=True,
                ),

                {
                    "input": query,
                    "actual_output": answer,
                    "retrieval_context": context,
                },
            ),

            (
                "answer_relevancy",

                AnswerRelevancyMetric(
                    threshold=THRESHOLD,
                    model=judge,
                    include_reason=True,
                ),

                {
                    "input": query,
                    "actual_output": answer,
                },
            ),

            (
                "contextual_relevancy",

                ContextualRelevancyMetric(
                    threshold=THRESHOLD,
                    model=judge,
                    include_reason=True,
                ),

                {
                    "input": query,
                    "actual_output": answer,
                    "retrieval_context": context,
                },
            ),
        ]

        # ----------------------------------------------------
        # Evaluate each metric independently
        # ----------------------------------------------------

        for key, metric, test_case_kwargs in jobs:

            # Don't judge a metric that already exists.
            if key in already:
                continue

            try:

                test_case = LLMTestCase(
                    **test_case_kwargs
                )

                metric.measure(
                    test_case
                )

                # ------------------------------------------------
                # Push result back to LangSmith
                # ------------------------------------------------

                client.create_feedback(
                    run_id=run.id,
                    key=key,
                    score=metric.score,
                    comment=metric.reason,
                )

                print(
                    f"[{key}] "
                    f"{run.id} -> "
                    f"{metric.score:.4f}"
                )

            except Exception as e:

                # One failed metric should not prevent
                # the remaining metrics from running.

                print(
                    f"[{key}] "
                    f"failed on run {run.id}: {e}"
                )


# ============================================================
# WORKER
# ============================================================

if __name__ == "__main__":

    print(
        f"Online triad worker started | "
        f"project={PROJECT} | "
        f"judge={JUDGE_MODEL} | "
        f"sample_rate={SAMPLE_RATE}"
    )

    while True:

        try:

            score_recent_traces()

        except Exception as e:

            print(
                f"[worker] evaluation pass failed: {e}"
            )

        time.sleep(
            POLL_SECONDS
        )