import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)

from src.reranker import RerankingRetriever
from src.cohere_judge import CohereJudge


load_dotenv()


# ============================================================
# CONFIG
# ============================================================

GOLDEN_PATH = "golden/generate_goldens_with_llm.json"

THRESHOLD = 0.7


# ============================================================
# COHERE AS THE LLM JUDGE
# ============================================================

JUDGE_MODEL = CohereJudge(
    model="command-a-03-2025"
)


# ============================================================
# 1. LOAD THE GOLDEN SET
# ============================================================

with open(GOLDEN_PATH, encoding="utf-8") as f:
    golden = json.load(f)

print(f"Loaded {len(golden)} golden test cases.")


# ============================================================
# 2. BUILD RERANKING RETRIEVER
# ============================================================

retriever = RerankingRetriever()


# ============================================================
# 3. RUN RERANKER ON EVERY QUESTION
# ============================================================

test_cases = []


for i, g in enumerate(golden):

    query = g["query"]

    print(
        f"\n[{i + 1}/{len(golden)}] "
        f"Retrieving: {query}"
    )

    # Run the reranking retriever
    retrieved = retriever.invoke(query)

    # Extract document text
    retrieval_context = [
        doc.page_content
        for doc in retrieved
    ]

    print(
        f"Retrieved {len(retrieval_context)} documents."
    )

    # Create DeepEval test case
    test_cases.append(
        LLMTestCase(
            input=query,

            expected_output=g["ideal_answer"],

            retrieval_context=retrieval_context,

            actual_output=(
                "(generator not evaluated in this run)"
            ),
        )
    )


# ============================================================
# 4. RETRIEVER METRICS
# ============================================================

metrics = [

    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),

    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),

]


# ============================================================
# 5. EVALUATE
# ============================================================

print("\n========================================")
print("Starting DeepEval")
print("Retriever : RerankingRetriever")
print("Judge     : Cohere Command A")
print("========================================\n")


evaluate(
    test_cases=test_cases,

    metrics=metrics,

    # Disable DeepEval cache because of
    # Windows shared-lock issue.
    cache_config=CacheConfig(
        write_cache=False,
        use_cache=False,
    ),

    hyperparameters={

        # Important: this is now reranked
        "retriever": "reranker",

        "embedding_model": "text-embedding-3-small",

        "chunk_size": 1000,

        "chunk_overlap": 150,

        "top_k": 5,

        "judge_model": "cohere/command-a-03-2025",

        "golden_set": GOLDEN_PATH,
    },
)


print("\n========================================")
print("Evaluation completed.")
print("========================================")