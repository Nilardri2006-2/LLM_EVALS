"""
evals/eval_generator.py
=======================

Component-level evaluation of the GENERATOR, in isolation.

Faithfulness:
Of the claims in the generated answer, how many are supported
by the context it was given?

Answer Relevancy:
How relevant is the generated answer to the student's question?

ISOLATION:
We feed the generator the GOLDEN context (known-good chunks),
NOT the retriever's output.

Therefore, a low generator score is primarily a generator issue,
rather than a retrieval issue.

Run:
    python -m evals.eval_generator
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

import os

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from src.generator import generate
from src.cohere_judge import CohereJudge


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

GOLDEN_PATH = "golden/faithfulness_dataset.json"

THRESHOLD = 0.7


# ============================================================
# COHERE AS THE EVALUATION JUDGE
# ============================================================

JUDGE_MODEL = CohereJudge(
    model="command-a-03-2025"
)


# ============================================================
# 1. LOAD GOLDEN SET
# ============================================================

import json

with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)

print(
    f"Loaded {len(goldens)} golden test cases."
)


# ============================================================
# 2. RUN GENERATOR USING GOLDEN CONTEXT
# ============================================================

test_cases = []


for i, g in enumerate(goldens):

    query = g["query"]

    # Known-good context.
    # We deliberately DON'T use the retriever here.
    context = g["ideal_context"]

    print(
        f"\n[{i + 1}/{len(goldens)}]"
    )

    print(
        f"Generating answer for: {query}"
    )

    # Run your Cohere-based generator
    answer = generate(
        query,
        context,
    )

    print(
        f"Generated answer: {answer[:150]}..."
    )

    # Build DeepEval test case
    test_cases.append(
        LLMTestCase(
            input=query,

            actual_output=answer,

            # Faithfulness will judge the answer
            # against this known-good context.
            retrieval_context=context,
        )
    )


# ============================================================
# 3. GENERATOR METRICS
# ============================================================

metrics = [

    # --------------------------------------------------------
    # FAITHFULNESS
    # --------------------------------------------------------
    #
    # Checks whether claims made by the generator
    # are actually supported by the provided context.
    #
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),

    # --------------------------------------------------------
    # ANSWER RELEVANCY
    # --------------------------------------------------------
    #
    # Checks whether the generated answer actually
    # answers the user's question.
    #
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        include_reason=True,
    ),
]


# ============================================================
# 4. RUN EVALUATION
# ============================================================

print("\n========================================")
print("Starting Generator Evaluation")
print("Generator : Cohere Command A")
print("Judge     : Cohere Command A")
print("========================================\n")


evaluate(
    test_cases=test_cases,

    metrics=metrics,

    # --------------------------------------------------------
    # Disable DeepEval cache.
    #
    # This avoids the Windows portalocker/shared-lock
    # problem you encountered earlier.
    # --------------------------------------------------------
    cache_config=CacheConfig(
        write_cache=False,
        use_cache=False,
    ),

    hyperparameters={

        "generator": "cohere-command-a-03-2025",

        "judge_model": "cohere-command-a-03-2025",

        "golden_set": GOLDEN_PATH,

        "temperature": 0,
    },
)


print("\n========================================")
print("Generator evaluation completed.")
print("========================================")