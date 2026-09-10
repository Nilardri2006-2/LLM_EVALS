"""
evals/eval_ops.py

Merged operational evaluation for the RAG application.

Evaluates:
    1. Latency
    2. Cost
    3. Reliability

Unlike quality/safety evaluations:
    - No golden dataset
    - No LLM judge
    - Uses software measurements + provider token usage

Pipeline:
    Query
      ↓
    RerankingRetriever
      ↓
    Cohere Command A
      ↓
    Operational measurements

Run:
    python -m evals.eval_ops
"""

# ============================================================
# 1. PROJECT ROOT
# ============================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 2. IMPORTS & ENV
# ============================================================

import math
import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import (
    generate,
    generate_stream,
    prompt,
    llm,
)

load_dotenv()


# ============================================================
# 3. SHARED CONFIG
# ============================================================

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]


# Your current RAG configuration
FETCH_K = 10
TOP_K = 5


# ============================================================
# 4. COST CONFIGURATION
# ============================================================

COST_REPEATS = 3

# IMPORTANT:
# Replace these with the CURRENT official Cohere prices
# for command-a-03-2025 before using the numbers for
# budgeting or your resume.

PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60

QUERIES_PER_DAY = 2000

USD_TO_INR = 88.0

COST_BUDGET_PER_QUERY_USD = 0.0015


# We stop BEFORE StrOutputParser()
# so usage_metadata remains available.
measured_chain = prompt | llm


# ============================================================
# 5. LATENCY CONFIGURATION
# ============================================================

LAT_REPEATS = 5

LAT_WARMUP_RUNS = 2

LAT_MEASURE_TTFT = True

LAT_STAGE_LEVEL = True

SLO_P95_MS = 3000

SLO_TTFT_P95_MS = 1200


# ============================================================
# 6. RELIABILITY CONFIGURATION
# ============================================================

REL_REPEATS = 5

MAX_RETRIES = 2

BACKOFF_BASE_S = 0.5


# ============================================================
# 7. SHARED HELPERS
# ============================================================

def percentile(values, p):

    values = [
        v for v in values
        if not math.isnan(v)
    ]

    if not values:
        return float("nan")

    values = sorted(values)

    k = (
        (len(values) - 1)
        * (p / 100.0)
    )

    lo = math.floor(k)
    hi = math.ceil(k)

    if lo == hi:
        return values[int(k)]

    return (
        values[lo] * (hi - k)
        + values[hi] * (k - lo)
    )


def col_avg(rows, key):

    return (
        sum(row[key] for row in rows)
        / len(rows)
        if rows
        else 0.0
    )


# ============================================================
# ============================================================
# LATENCY
# ============================================================
# ============================================================


def lat_end_to_end(
    pipeline,
    question,
):

    result = pipeline.invoke(
        question
    )

    return result["answer"]


def lat_stages(
    pipeline,
    question,
):

    # Retrieval + reranking
    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(
        question
    )

    context = [
        doc.page_content
        for doc in docs
    ]

    t1 = time.perf_counter()


    # Generation
    answer = generate(
        question,
        context
    )

    t2 = time.perf_counter()


    return answer, {
        "retrieval":
            (t1 - t0) * 1000,

        "generation":
            (t2 - t1) * 1000,
    }


def lat_stages_streaming(
    pipeline,
    question,
):

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    t0 = time.perf_counter()

    docs = pipeline.retriever.invoke(
        question
    )

    context = [
        doc.page_content
        for doc in docs
    ]

    t1 = time.perf_counter()


    # --------------------------------------------------------
    # Streaming generation
    # --------------------------------------------------------

    first_token_t = None

    pieces = []


    for piece in generate_stream(
        question,
        context
    ):

        if first_token_t is None:

            first_token_t = (
                time.perf_counter()
            )

        pieces.append(piece)


    t2 = time.perf_counter()


    answer = "".join(pieces)


    ttft_ms = (
        (first_token_t - t0) * 1000
        if first_token_t
        else float("nan")
    )


    return answer, {

        "retrieval":
            (t1 - t0) * 1000,

        "generation":
            (t2 - t1) * 1000,

        "ttft":
            ttft_ms,
    }


def lat_benchmark(pipeline):

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(
        f"[latency] warming up "
        f"({LAT_WARMUP_RUNS} runs)..."
    )

    for i in range(
        LAT_WARMUP_RUNS
    ):

        lat_end_to_end(
            pipeline,
            QUESTIONS[
                i % len(QUESTIONS)
            ],
        )


    total_ms = []

    retrieval_ms = []

    generation_ms = []

    ttft_ms = []

    answer_lengths = []


    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    print("[latency] measuring...")


    for question in QUESTIONS:

        for _ in range(
            LAT_REPEATS
        ):

            start = time.perf_counter()


            if LAT_MEASURE_TTFT:

                answer, stage = (
                    lat_stages_streaming(
                        pipeline,
                        question,
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


            elif LAT_STAGE_LEVEL:

                answer, stage = (
                    lat_stages(
                        pipeline,
                        question,
                    )
                )

                retrieval_ms.append(
                    stage["retrieval"]
                )

                generation_ms.append(
                    stage["generation"]
                )


            else:

                answer = lat_end_to_end(
                    pipeline,
                    question,
                )


            elapsed_ms = (
                time.perf_counter()
                - start
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


def lat_summarize(samples):

    clean = [
        s for s in samples
        if not math.isnan(s)
    ]

    return {

        "n": len(clean),

        "mean":
            sum(clean) / len(clean),

        "p50":
            percentile(clean, 50),

        "p95":
            percentile(clean, 95),

        "p99":
            percentile(clean, 99),

        "min":
            min(clean),

        "max":
            max(clean),
    }


def lat_print_row(
    label,
    summary,
):

    print(
        f"{label:<12} | "
        f"n={summary['n']:<3} "
        f"mean={summary['mean']:7.1f} "
        f"p50={summary['p50']:7.1f} "
        f"p95={summary['p95']:7.1f} "
        f"p99={summary['p99']:7.1f} "
        f"min={summary['min']:7.1f} "
        f"max={summary['max']:7.1f}"
    )


def lat_slo_line(
    label,
    p95,
    budget,
):

    verdict = (
        "PASS"
        if p95 <= budget
        else "FAIL"
    )

    print(
        f"SLO: {label:<22} "
        f"p95 <= {budget:>5} ms "
        f"-> p95 = {p95:7.0f} ms "
        f"[{verdict}]"
    )


def lat_report(results):

    print("\n" + "=" * 78)

    print(
        "LATENCY (milliseconds)"
    )

    print("=" * 78)


    total = lat_summarize(
        results["total"]
    )

    lat_print_row(
        "end-to-end",
        total,
    )


    if results["ttft"]:

        lat_print_row(
            "ttft",
            lat_summarize(
                results["ttft"]
            ),
        )


    if results["retrieval"]:

        lat_print_row(
            "retrieval",
            lat_summarize(
                results["retrieval"]
            ),
        )

        lat_print_row(
            "generation",
            lat_summarize(
                results["generation"]
            ),
        )


    avg_len = (
        sum(results["answer_len"])
        / len(results["answer_len"])
    )


    print("-" * 78)

    print(
        f"avg answer length: "
        f"{avg_len:.0f} chars"
    )


    print("=" * 78)

    lat_slo_line(
        "full answer",
        total["p95"],
        SLO_P95_MS,
    )


    if results["ttft"]:

        ttft = lat_summarize(
            results["ttft"]
        )

        lat_slo_line(
            "first token (perceived)",
            ttft["p95"],
            SLO_TTFT_P95_MS,
        )


    print("=" * 78)


def run_latency(
    pipeline,
    verbose=True,
):

    results = lat_benchmark(
        pipeline
    )


    if verbose:

        lat_report(
            results
        )


    total = lat_summarize(
        results["total"]
    )


    metrics = {

        "e2e_mean_ms":
            total["mean"],

        "e2e_p50_ms":
            total["p50"],

        "e2e_p95_ms":
            total["p95"],

        "e2e_p99_ms":
            total["p99"],

        "avg_answer_len":
            sum(results["answer_len"])
            / len(results["answer_len"]),

        "slo_e2e_pass":
            total["p95"]
            <= SLO_P95_MS,
    }


    if results["ttft"]:

        ttft = lat_summarize(
            results["ttft"]
        )

        metrics[
            "ttft_p95_ms"
        ] = ttft["p95"]

        metrics[
            "slo_ttft_pass"
        ] = (
            ttft["p95"]
            <= SLO_TTFT_P95_MS
        )


    if results["retrieval"]:

        metrics[
            "retrieval_p95_ms"
        ] = lat_summarize(
            results["retrieval"]
        )["p95"]

        metrics[
            "generation_p95_ms"
        ] = lat_summarize(
            results["generation"]
        )["p95"]


    return metrics


# ============================================================
# ============================================================
# COST
# ============================================================
# ============================================================


def cost_measure_tokens(
    pipeline,
    question,
):

    # Retrieval + reranking
    docs = pipeline.retriever.invoke(
        question
    )


    context_text = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # Generate using exact same prompt/model
    # but stop before StrOutputParser.
    message = measured_chain.invoke(
        {
            "question": question,
            "context": context_text,
        }
    )


    usage = (
        message.usage_metadata
        or {}
    )


    input_tokens = usage.get(
        "input_tokens",
        0,
    )


    output_tokens = usage.get(
        "output_tokens",
        0,
    )


    details = (
        usage.get(
            "input_token_details"
        )
        or {}
    )


    cached_tokens = (
        details.get(
            "cache_read",
            0,
        )
        or 0
    )


    return {

        "input":
            input_tokens,

        "output":
            output_tokens,

        "cached":
            cached_tokens,
    }


def cost_usd(
    input_tokens,
    output_tokens,
    cached_tokens,
):

    uncached_input = max(
        input_tokens
        - cached_tokens,
        0,
    )


    input_cost = (
        uncached_input
        / 1_000_000
        * PRICE_INPUT_PER_1M
    )


    cached_cost = (
        cached_tokens
        / 1_000_000
        * PRICE_CACHED_INPUT_PER_1M
    )


    output_cost = (
        output_tokens
        / 1_000_000
        * PRICE_OUTPUT_PER_1M
    )


    return {

        "input":
            input_cost,

        "cached":
            cached_cost,

        "output":
            output_cost,

        "total":
            input_cost
            + cached_cost
            + output_cost,
    }


def cost_benchmark(pipeline):

    rows = []


    print(
        "[cost] measuring token usage..."
    )


    for question in QUESTIONS:

        for _ in range(
            COST_REPEATS
        ):

            tokens = (
                cost_measure_tokens(
                    pipeline,
                    question,
                )
            )


            cost = cost_usd(
                tokens["input"],
                tokens["output"],
                tokens["cached"],
            )


            rows.append(
                {
                    **tokens,

                    **{
                        f"cost_{key}": value
                        for key, value
                        in cost.items()
                    },
                }
            )


    return rows


def cost_report(rows):

    n = len(rows)

    avg_input = col_avg(
        rows,
        "input",
    )

    avg_output = col_avg(
        rows,
        "output",
    )

    avg_cached = col_avg(
        rows,
        "cached",
    )

    avg_cost = col_avg(
        rows,
        "cost_total",
    )

    min_cost = min(
        row["cost_total"]
        for row in rows
    )

    max_cost = max(
        row["cost_total"]
        for row in rows
    )


    avg_output_cost = col_avg(
        rows,
        "cost_output",
    )


    output_share = (
        100
        * avg_output_cost
        / avg_cost
        if avg_cost
        else 0
    )


    print("\n" + "=" * 70)

    print(
        "COST — Cohere Command A"
    )

    print("=" * 70)

    print(
        f"samples             : {n}"
    )

    print(
        f"avg input tokens    : "
        f"{avg_input:.0f}"
    )

    print(
        f"avg cached tokens   : "
        f"{avg_cached:.0f}"
    )

    print(
        f"avg output tokens   : "
        f"{avg_output:.0f}"
    )

    print("-" * 70)

    print(
        f"avg cost / query    : "
        f"${avg_cost:.6f} "
        f"(Rs {avg_cost * USD_TO_INR:.4f})"
    )

    print(
        f"min / max           : "
        f"${min_cost:.6f} / "
        f"${max_cost:.6f}"
    )

    print(
        f"input / output cost : "
        f"{100 - output_share:.0f}% / "
        f"{output_share:.0f}%"
    )

    print("-" * 70)


    daily = (
        avg_cost
        * QUERIES_PER_DAY
    )

    monthly = (
        daily * 30
    )


    print(
        f"projection @ "
        f"{QUERIES_PER_DAY}/day:"
    )

    print(
        f"  per day            : "
        f"${daily:.2f} "
        f"(Rs {daily * USD_TO_INR:.2f})"
    )

    print(
        f"  per month          : "
        f"${monthly:.2f} "
        f"(Rs {monthly * USD_TO_INR:.2f})"
    )


    print("=" * 70)


    verdict = (
        "PASS"
        if avg_cost
        <= COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )


    print(
        f"BUDGET: <= "
        f"${COST_BUDGET_PER_QUERY_USD:.6f}"
        f" -> ${avg_cost:.6f} "
        f"[{verdict}]"
    )


    print("=" * 70)


def run_cost(
    pipeline,
    verbose=True,
):

    rows = cost_benchmark(
        pipeline
    )


    if verbose:

        cost_report(
            rows
        )


    avg_cost = col_avg(
        rows,
        "cost_total",
    )


    avg_output_cost = col_avg(
        rows,
        "cost_output",
    )


    output_share = (
        100
        * avg_output_cost
        / avg_cost
        if avg_cost
        else 0.0
    )


    return {

        "cost_per_query_usd":
            avg_cost,

        "cost_per_query_inr":
            avg_cost * USD_TO_INR,

        "avg_input_tokens":
            col_avg(rows, "input"),

        "avg_output_tokens":
            col_avg(rows, "output"),

        "avg_cached_tokens":
            col_avg(rows, "cached"),

        "output_cost_share_pct":
            output_share,

        "monthly_usd":
            avg_cost
            * QUERIES_PER_DAY
            * 30,

        "budget_pass":
            avg_cost
            <= COST_BUDGET_PER_QUERY_USD,
    }


# ============================================================
# ============================================================
# RELIABILITY
# ============================================================
# ============================================================


class Reliability:

    def __init__(self):

        self.calls = 0

        self.successes = 0

        self.failures = 0

        self.retries = 0

        self.attempts = 0


def call_with_retries(
    fn,
    reliability,
):

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

            if attempt < MAX_RETRIES:

                reliability.retries += 1


                wait_time = (
                    BACKOFF_BASE_S
                    * (2 ** attempt)
                )


                print(
                    f"[reliability] "
                    f"attempt {attempt + 1} "
                    f"failed; retrying in "
                    f"{wait_time:.1f}s"
                )


                time.sleep(
                    wait_time
                )


            else:

                reliability.failures += 1


                print(
                    f"[reliability] "
                    f"FAILED after "
                    f"{MAX_RETRIES} retries: "
                    f"{exc}"
                )


                return None


def rel_benchmark(pipeline):

    reliability = Reliability()


    total_requests = (
        len(QUESTIONS)
        * REL_REPEATS
    )


    print(
        f"[reliability] measuring "
        f"{total_requests} requests..."
    )


    for question in QUESTIONS:

        for _ in range(
            REL_REPEATS
        ):

            call_with_retries(

                lambda q=question:
                    pipeline.invoke(q),

                reliability,
            )


    return reliability


def rel_report(rel):

    calls = rel.calls or 1


    success_rate = (
        100
        * rel.successes
        / calls
    )


    error_rate = (
        100
        * rel.failures
        / calls
    )


    retry_rate = (
        100
        * rel.retries
        / calls
    )


    avg_attempts = (
        rel.attempts
        / calls
    )


    print("\n" + "=" * 60)

    print(
        "RELIABILITY"
    )

    print("=" * 60)

    print(
        f"total requests : "
        f"{rel.calls}"
    )

    print(
        f"total attempts : "
        f"{rel.attempts}"
    )

    print(
        f"successful     : "
        f"{rel.successes}"
    )

    print(
        f"failed         : "
        f"{rel.failures}"
    )

    print("-" * 60)

    print(
        f"success rate   : "
        f"{success_rate:.2f}%"
    )

    print(
        f"error rate     : "
        f"{error_rate:.2f}%"
    )

    print(
        f"retry rate     : "
        f"{retry_rate:.2f}%"
    )

    print(
        f"avg attempts   : "
        f"{avg_attempts:.2f}"
    )

    print("=" * 60)


def run_reliability(
    pipeline,
    verbose=True,
):

    rel = rel_benchmark(
        pipeline
    )


    if verbose:

        rel_report(
            rel
        )


    calls = rel.calls or 1


    return {

        "total_requests":
            rel.calls,

        "success_rate":
            100
            * rel.successes
            / calls,

        "error_rate":
            100
            * rel.failures
            / calls,

        "retry_rate":
            100
            * rel.retries
            / calls,

        "avg_attempts":
            rel.attempts
            / calls,
    }


# ============================================================
# 12. MERGED OPERATIONAL EVALUATION
# ============================================================

def run_ops(
    pipeline=None,
    verbose=True,
):

    pipeline = (
        pipeline
        or RagPipeline(
            fetch_k=FETCH_K,
            top_k=TOP_K,
        )
    )


    print("\n" + "=" * 70)

    print(
        "RAG OPERATIONAL EVALUATION"
    )

    print("=" * 70)

    print(
        "Retriever : RerankingRetriever"
    )

    print(
        f"fetch_k  : {FETCH_K}"
    )

    print(
        f"top_k    : {TOP_K}"
    )

    print(
        "Generator : Cohere Command A"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Run all three
    # --------------------------------------------------------

    latency = run_latency(
        pipeline,
        verbose=verbose,
    )


    cost = run_cost(
        pipeline,
        verbose=verbose,
    )


    reliability = run_reliability(
        pipeline,
        verbose=verbose,
    )


    # --------------------------------------------------------
    # Merge into one snapshot
    # --------------------------------------------------------

    snapshot = {}


    snapshot.update(
        {
            f"latency.{key}": value
            for key, value
            in latency.items()
        }
    )


    snapshot.update(
        {
            f"cost.{key}": value
            for key, value
            in cost.items()
        }
    )


    snapshot.update(
        {
            f"reliability.{key}": value
            for key, value
            in reliability.items()
        }
    )


    return snapshot


# ============================================================
# 13. MAIN
# ============================================================

def main():

    snapshot = run_ops(
        verbose=True
    )


    print("\n" + "=" * 70)

    print(
        "OPERATIONAL SNAPSHOT"
    )

    print("=" * 70)


    for key, value in snapshot.items():

        if isinstance(value, float):

            shown = f"{value:.4f}"

        else:

            shown = str(value)


        print(
            f"  {key:<32} "
            f"{shown}"
        )


    print("=" * 70)


if __name__ == "__main__":

    main()