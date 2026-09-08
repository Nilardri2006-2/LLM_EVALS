"""
evals/eval_latency.py

Operational RAG evaluation: LATENCY + TTFT.

Measures:
    - End-to-end latency
    - Retrieval + reranking latency
    - Generation latency
    - Time-to-first-token (TTFT)
    - p50 / p95 / p99 latency
    - SLO pass/fail

No golden dataset.
No LLM judge.
No DeepEval metric.

Run:
    python -m evals.eval_latency
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

import math
import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import generate, generate_stream


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

# Number of measured runs per question
REPEATS = 5

# Runs before measurement to remove cold-start effects
WARMUP_RUNS = 2

# Measure time until first streamed token
MEASURE_TTFT = True

# Break latency into retrieval + generation
STAGE_LEVEL = True


# ============================================================
# RAG CONFIGURATION
# ============================================================

FETCH_K = 10
TOP_K = 5


# ============================================================
# SERVICE LEVEL OBJECTIVES
# ============================================================

# Full answer should be ready within 3 seconds at p95
SLO_P95_MS = 3000

# First token should appear within 1.2 seconds at p95
SLO_TTFT_P95_MS = 1200


# ============================================================
# PIPELINE ADAPTERS
# ============================================================

def run_end_to_end(pipeline, question):

    result = pipeline.invoke(question)

    return result["answer"]


def run_stages(pipeline, question):

    # -------------------------------
    # Retrieval + reranking
    # -------------------------------

    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(question)

    context = [
        doc.page_content
        for doc in docs
    ]

    t1 = time.perf_counter()

    # -------------------------------
    # Generation
    # -------------------------------

    answer = generate(
        question,
        context
    )

    t2 = time.perf_counter()

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
    }


def run_stages_streaming(pipeline, question):

    # -------------------------------
    # Retrieval + reranking
    # -------------------------------

    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(question)

    context = [
        doc.page_content
        for doc in docs
    ]

    t1 = time.perf_counter()

    # -------------------------------
    # Streaming generation
    # -------------------------------

    first_token_time = None

    pieces = []

    for piece in generate_stream(
        question,
        context
    ):

        if piece:

            if first_token_time is None:
                first_token_time = time.perf_counter()

            pieces.append(piece)

    t2 = time.perf_counter()

    answer = "".join(pieces)

    # Query → first token
    ttft_ms = (
        (first_token_time - t0) * 1000
        if first_token_time is not None
        else float("nan")
    )

    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, p):

    values = [
        value
        for value in values
        if not math.isnan(value)
    ]

    if not values:
        return float("nan")

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)

    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return values[int(k)]

    return (
        values[lower] * (upper - k)
        + values[upper] * (k - lower)
    )


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(pipeline):

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print(
        f"Warming up ({WARMUP_RUNS} runs, discarded)..."
    )

    for i in range(WARMUP_RUNS):

        question = QUESTIONS[
            i % len(QUESTIONS)
        ]

        run_end_to_end(
            pipeline,
            question
        )


    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    total_ms = []
    retrieval_ms = []
    generation_ms = []
    ttft_ms = []

    answer_lengths = []


    # --------------------------------------------------------
    # MEASURED RUNS
    # --------------------------------------------------------

    total_samples = (
        len(QUESTIONS) * REPEATS
    )

    print(
        f"Measuring {total_samples} runs..."
    )

    for question in QUESTIONS:

        for _ in range(REPEATS):

            start = time.perf_counter()


            if MEASURE_TTFT:

                answer, stage = (
                    run_stages_streaming(
                        pipeline,
                        question
                    )
                )

                retrieval_ms.append(
                    stage["retrieval"]
                )

                generation_ms.append(
                    stage["generation"]
                )

                ttft_ms.append(
                    stage["ttft"]
                )


            elif STAGE_LEVEL:

                answer, stage = (
                    run_stages(
                        pipeline,
                        question
                    )
                )

                retrieval_ms.append(
                    stage["retrieval"]
                )

                generation_ms.append(
                    stage["generation"]
                )


            else:

                answer = run_end_to_end(
                    pipeline,
                    question
                )


            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            total_ms.append(
                elapsed_ms
            )

            answer_lengths.append(
                len(answer or "")
            )


    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize(samples):

    clean = [
        value
        for value in samples
        if not math.isnan(value)
    ]

    if not clean:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(clean),

        "mean": (
            sum(clean) / len(clean)
        ),

        "p50": percentile(
            clean,
            50
        ),

        "p95": percentile(
            clean,
            95
        ),

        "p99": percentile(
            clean,
            99
        ),

        "min": min(clean),
        "max": max(clean),
    }


# ============================================================
# PRINT METRIC ROW
# ============================================================

def print_row(label, stats):

    print(
        f"{label:<12} | "
        f"n={stats['n']:<3} "
        f"mean={stats['mean']:7.1f} "
        f"p50={stats['p50']:7.1f} "
        f"p95={stats['p95']:7.1f} "
        f"p99={stats['p99']:7.1f} "
        f"min={stats['min']:7.1f} "
        f"max={stats['max']:7.1f}"
    )


# ============================================================
# SLO
# ============================================================

def slo_line(label, p95, budget):

    verdict = (
        "PASS"
        if p95 <= budget
        else "FAIL"
    )

    print(
        f"SLO: {label:<22} "
        f"p95 <= {budget:>5} ms  "
        f"-> p95 = {p95:7.0f} ms "
        f"[{verdict}]"
    )


# ============================================================
# REPORT
# ============================================================

def report(results):

    print("\n" + "=" * 78)
    print("RAG LATENCY EVALUATION")
    print("=" * 78)

    print(
        f"{'stage':<12} | "
        f"{'samples':<5} "
        f"{'mean':>11} "
        f"{'p50':>11} "
        f"{'p95':>11} "
        f"{'p99':>11} "
        f"{'min':>11} "
        f"{'max':>11}"
    )

    print("-" * 78)


    # --------------------------------------------------------
    # END-TO-END
    # --------------------------------------------------------

    total = summarize(
        results["total"]
    )

    print_row(
        "end-to-end",
        total
    )


    # --------------------------------------------------------
    # TTFT
    # --------------------------------------------------------

    if results["ttft"]:

        print_row(
            "ttft",
            summarize(
                results["ttft"]
            )
        )


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    if results["retrieval"]:

        print_row(
            "retrieval",
            summarize(
                results["retrieval"]
            )
        )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    if results["generation"]:

        print_row(
            "generation",
            summarize(
                results["generation"]
            )
        )


    # --------------------------------------------------------
    # ANSWER LENGTH
    # --------------------------------------------------------

    avg_length = (
        sum(results["answer_len"])
        / len(results["answer_len"])
    )

    print("-" * 78)

    print(
        f"Average answer length: "
        f"{avg_length:.0f} characters"
    )


    # --------------------------------------------------------
    # SLO RESULTS
    # --------------------------------------------------------

    print("=" * 78)

    slo_line(
        "full answer",
        total["p95"],
        SLO_P95_MS
    )


    if results["ttft"]:

        ttft = summarize(
            results["ttft"]
        )

        slo_line(
            "first token (perceived)",
            ttft["p95"],
            SLO_TTFT_P95_MS
        )


    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("RAG LATENCY BENCHMARK")
    print("========================================")

    print(f"Retriever : RerankingRetriever")
    print(f"fetch_k   : {FETCH_K}")
    print(f"top_k     : {TOP_K}")
    print(f"Generator : Cohere Command A")
    print(f"TTFT      : {MEASURE_TTFT}")
    print(f"Repeats   : {REPEATS}")
    print("========================================")


    pipeline = RagPipeline(
        fetch_k=FETCH_K,
        top_k=TOP_K,
    )

    results = benchmark(
        pipeline
    )

    report(
        results
    )


if __name__ == "__main__":
    main()