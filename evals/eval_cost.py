"""
evals/eval_cost.py

Operational RAG evaluation: COST.

Measures:
    - Input tokens
    - Cached input tokens
    - Output tokens
    - Cost per query
    - Daily/monthly projected cost
    - Cost budget PASS/FAIL

No golden dataset.
No LLM judge.
Cost is derived from real token usage × provider pricing.

Run:
    python -m evals.eval_cost
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

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import prompt, llm


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

# Cost is relatively stable, so fewer repeats are sufficient.
REPEATS = 3


# ============================================================
# RAG CONFIGURATION
# ============================================================

FETCH_K = 10
TOP_K = 5


# ============================================================
# COHERE PRICING
# ============================================================

# IMPORTANT:
# Keep these values updated from Cohere's current pricing page.
#
# These are expressed as USD per 1 million tokens.

PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60


# ============================================================
# BUSINESS PROJECTION
# ============================================================

QUERIES_PER_DAY = 2000

USD_TO_INR = 95.0


# ============================================================
# COST BUDGET
# ============================================================

COST_BUDGET_PER_QUERY_USD = 0.0015


# ============================================================
# MEASURED GENERATION CHAIN
# ============================================================

# Your normal generator is:
#
#     prompt → llm → StrOutputParser
#
# StrOutputParser removes the AIMessage and leaves only a string.
#
# For cost measurement we stop at llm so that usage_metadata
# remains available.

measured_chain = prompt | llm


# ============================================================
# TOKEN MEASUREMENT
# ============================================================

def measure_tokens(pipeline, question):

    # --------------------------------------------------------
    # RETRIEVAL + RERANKING
    # --------------------------------------------------------

    docs = pipeline.retriever.invoke(question)

    context_text = "\n\n".join(
        doc.page_content
        for doc in docs
    )


    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    message = measured_chain.invoke(
        {
            "question": question,
            "context": context_text,
        }
    )


    # --------------------------------------------------------
    # TOKEN USAGE
    # --------------------------------------------------------

    usage = message.usage_metadata or {}

    input_tokens = usage.get(
        "input_tokens",
        0
    )

    output_tokens = usage.get(
        "output_tokens",
        0
    )


    # Cohere may expose cached-token information
    # through input_token_details.

    details = (
        usage.get("input_token_details")
        or {}
    )

    cached_tokens = (
        details.get("cache_read", 0)
        or 0
    )


    return {
        "input": input_tokens,
        "output": output_tokens,
        "cached": cached_tokens,
    }


# ============================================================
# COST CALCULATION
# ============================================================

def cost_usd(
    input_tokens,
    output_tokens,
    cached_tokens,
):

    # Input tokens not served from cache
    uncached_input = max(
        input_tokens - cached_tokens,
        0
    )


    # Normal input cost
    input_cost = (
        uncached_input
        / 1_000_000
        * PRICE_INPUT_PER_1M
    )


    # Cached input cost
    cached_cost = (
        cached_tokens
        / 1_000_000
        * PRICE_CACHED_INPUT_PER_1M
    )


    # Output cost
    output_cost = (
        output_tokens
        / 1_000_000
        * PRICE_OUTPUT_PER_1M
    )


    total = (
        input_cost
        + cached_cost
        + output_cost
    )


    return {
        "input": input_cost,
        "cached": cached_cost,
        "output": output_cost,
        "total": total,
    }


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(pipeline):

    rows = []

    print("Measuring token usage...")

    for question in QUESTIONS:

        for _ in range(REPEATS):

            tokens = measure_tokens(
                pipeline,
                question
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
                        for key, value in cost.items()
                    },
                }
            )

    return rows


# ============================================================
# AVERAGE
# ============================================================

def avg(rows, key):

    return (
        sum(row[key] for row in rows)
        / len(rows)
    )


# ============================================================
# REPORT
# ============================================================

def report(rows):

    n = len(rows)

    # --------------------------------------------------------
    # TOKEN USAGE
    # --------------------------------------------------------

    avg_input = avg(
        rows,
        "input"
    )

    avg_output = avg(
        rows,
        "output"
    )

    avg_cached = avg(
        rows,
        "cached"
    )


    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    avg_cost = avg(
        rows,
        "cost_total"
    )

    min_cost = min(
        row["cost_total"]
        for row in rows
    )

    max_cost = max(
        row["cost_total"]
        for row in rows
    )


    # --------------------------------------------------------
    # INPUT / OUTPUT COST SPLIT
    # --------------------------------------------------------

    avg_input_cost = (
        avg(rows, "cost_input")
        + avg(rows, "cost_cached")
    )

    avg_output_cost = avg(
        rows,
        "cost_output"
    )


    output_share = (
        100 * avg_output_cost / avg_cost
        if avg_cost
        else 0
    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 72)
    print("RAG COST EVALUATION")
    print("=" * 72)

    print(f"Generator model        : command-a-03-2025")
    print(f"Retriever              : RerankingRetriever")
    print(f"fetch_k                : {FETCH_K}")
    print(f"top_k                  : {TOP_K}")
    print(f"samples                : {n}")

    print("-" * 72)

    print(
        f"avg input tokens      : "
        f"{avg_input:8.0f}"
    )

    print(
        f"avg cached tokens     : "
        f"{avg_cached:8.0f}"
    )

    print(
        f"avg output tokens     : "
        f"{avg_output:8.0f}"
    )

    print("-" * 72)

    print(
        f"avg cost / query      : "
        f"${avg_cost:.6f} "
        f"(Rs {avg_cost * USD_TO_INR:.4f})"
    )

    print(
        f"min / max cost        : "
        f"${min_cost:.6f} / "
        f"${max_cost:.6f}"
    )

    print(
        f"input vs output cost  : "
        f"{100 - output_share:.0f}% / "
        f"{output_share:.0f}%"
    )

    print("-" * 72)


    # ========================================================
    # BUSINESS PROJECTION
    # ========================================================

    daily_cost = (
        avg_cost
        * QUERIES_PER_DAY
    )

    monthly_cost = (
        daily_cost
        * 30
    )


    print(
        f"Projection @ "
        f"{QUERIES_PER_DAY}/day:"
    )

    print(
        f"   per day            : "
        f"${daily_cost:8.2f} "
        f"(Rs {daily_cost * USD_TO_INR:8.2f})"
    )

    print(
        f"   per month          : "
        f"${monthly_cost:8.2f} "
        f"(Rs {monthly_cost * USD_TO_INR:8.2f})"
    )


    # ========================================================
    # BUDGET VERDICT
    # ========================================================

    print("=" * 72)

    verdict = (
        "PASS"
        if avg_cost <= COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )

    print(
        f"BUDGET: cost/query <= "
        f"${COST_BUDGET_PER_QUERY_USD:.6f}"
    )

    print(
        f"        actual        = "
        f"${avg_cost:.6f}"
        f"   [{verdict}]"
    )

    print("=" * 72)

    print(
        "Note: actual production cost may differ because "
        "provider-side prompt caching, pricing, and usage "
        "patterns can change."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("RAG COST BENCHMARK")
    print("========================================")

    pipeline = RagPipeline(
        fetch_k=FETCH_K,
        top_k=TOP_K,
    )

    rows = benchmark(
        pipeline
    )

    report(
        rows
    )


if __name__ == "__main__":
    main()