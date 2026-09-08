"""
evals/eval_reliability.py

Operational RAG evaluation: RELIABILITY.

Measures:
    - Success rate
    - Error rate
    - Retry rate

No golden dataset.
No LLM judge.

Reliability checks whether the complete RAG pipeline can
successfully serve requests.

Run:
    python -m evals.eval_reliability
"""

import sys
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline


load_dotenv()


# ============================================================
# CONFIG
# ============================================================

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5

# Number of retries AFTER the initial attempt
MAX_RETRIES = 2

# Exponential backoff:
# retry 1 → 0.5s
# retry 2 → 1.0s
BACKOFF_BASE_S = 0.5


# ============================================================
# RAG CONFIGURATION
# ============================================================

FETCH_K = 10
TOP_K = 5


# ============================================================
# RELIABILITY TRACKER
# ============================================================

class Reliability:

    def __init__(self):

        # User-level requests
        self.calls = 0

        # Requests that eventually succeeded
        self.successes = 0

        # Requests that failed even after retries
        self.failures = 0

        # Number of retry attempts
        self.retries = 0

        # Total attempts including first attempts + retries
        self.attempts = 0


# ============================================================
# RETRY WRAPPER
# ============================================================

def call_with_retries(
    fn,
    reliability,
):

    # One user request
    reliability.calls += 1

    for attempt in range(
        MAX_RETRIES + 1
    ):

        reliability.attempts += 1

        try:

            result = fn()

            reliability.successes += 1

            return result

        except Exception as exc:

            # Retry if attempts remain
            if attempt < MAX_RETRIES:

                reliability.retries += 1

                wait_time = (
                    BACKOFF_BASE_S
                    * (2 ** attempt)
                )

                print(
                    f"  Attempt {attempt + 1} failed. "
                    f"Retrying in {wait_time:.1f}s..."
                )

                time.sleep(
                    wait_time
                )

            else:

                reliability.failures += 1

                print(
                    f"  FAILED after "
                    f"{MAX_RETRIES} retries: {exc}"
                )

                return None


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(pipeline):

    reliability = Reliability()

    total_requests = (
        len(QUESTIONS)
        * REPEATS
    )

    print(
        f"Measuring reliability "
        f"({total_requests} requests)..."
    )

    for question in QUESTIONS:

        for _ in range(REPEATS):

            call_with_retries(
                lambda q=question: pipeline.invoke(q),
                reliability,
            )

    return reliability


# ============================================================
# REPORT
# ============================================================

def report(rel):

    # --------------------------------------------------------
    # SUCCESS RATE
    # --------------------------------------------------------

    success_rate = (
        100
        * rel.successes
        / rel.calls
        if rel.calls
        else 0
    )


    # --------------------------------------------------------
    # ERROR RATE
    # --------------------------------------------------------

    error_rate = (
        100
        * rel.failures
        / rel.calls
        if rel.calls
        else 0
    )


    # --------------------------------------------------------
    # RETRY RATE
    # --------------------------------------------------------

    retry_rate = (
        100
        * rel.retries
        / rel.calls
        if rel.calls
        else 0
    )


    # --------------------------------------------------------
    # ATTEMPTS PER REQUEST
    # --------------------------------------------------------

    avg_attempts = (
        rel.attempts
        / rel.calls
        if rel.calls
        else 0
    )


    # ========================================================
    # PRINT REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("RAG RELIABILITY EVALUATION")
    print("=" * 70)

    print(
        f"Retriever       : RerankingRetriever"
    )

    print(
        f"fetch_k         : {FETCH_K}"
    )

    print(
        f"top_k           : {TOP_K}"
    )

    print(
        f"Total requests  : {rel.calls}"
    )

    print(
        f"Total attempts  : {rel.attempts}"
    )

    print(
        f"Successful      : {rel.successes}"
    )

    print(
        f"Failed          : {rel.failures}"
    )

    print("-" * 70)

    print(
        f"Success rate    : {success_rate:.2f}%"
    )

    print(
        f"Error rate      : {error_rate:.2f}%"
    )

    print(
        f"Retry rate      : {retry_rate:.2f}%"
    )

    print(
        f"Avg attempts    : {avg_attempts:.2f} / request"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("RAG RELIABILITY BENCHMARK")
    print("========================================")

    pipeline = RagPipeline(
        fetch_k=FETCH_K,
        top_k=TOP_K,
    )

    reliability = benchmark(
        pipeline
    )

    report(
        reliability
    )


if __name__ == "__main__":
    main()