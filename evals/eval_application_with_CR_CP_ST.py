# evals/eval_application.py

"""
Application-level RAG evaluation.

Pipeline:
    Golden Query
        ↓
    RerankingRetriever
        ↓
    Cohere Generator
        ↓
    Actual Answer
        ↓
    Cohere Judge
        ↓
    Correctness
    Completeness
    Style

Run:
    python -m evals.eval_application
"""

import sys
from pathlib import Path

# Make project root importable when running as a module/script
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate import CacheConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline
from src.cohere_judge import CohereJudge


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = "golden/correctness_golden.json"

THRESHOLD = 0.7

JUDGE_MODEL = CohereJudge(
    model="command-a-03-2025"
)


# ============================================================
# LOAD GOLDENS
# ============================================================

def load_goldens():
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# APPLICATION EVALUATION
# ============================================================

def run(rag):

    # 1. LOAD QUERIES + IDEAL ANSWERS
    goldens = load_goldens()

    print(f"Loaded {len(goldens)} golden test cases.")

    # 2. RUN THE ACTUAL RAG PIPELINE
    test_cases = []

    for i, g in enumerate(goldens):

        query = g["question"]

        print(f"\n[{i + 1}/{len(goldens)}]")
        print(f"Query: {query}")

        # Live RAG:
        # query -> retrieve -> rerank -> generate
        result = rag.invoke(query)

        answer = result["answer"]

        # Your golden file should contain ideal_answer
        expected_answer = g.get("ideal_answer", "")

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_answer,
            )
        )

        print(f"Answer: {answer[:200]}...")

    # ========================================================
    # 3. APPLICATION-LEVEL METRICS
    # ========================================================

    # --------------------------------------------------------
    # 3a. CORRECTNESS
    # --------------------------------------------------------

    correctness = GEval(
        name="Correctness",

        evaluation_steps=[
            "Compare only the factual claims in the actual output "
            "against the expected output.",

            "A claim is wrong only if it contradicts the expected "
            "output or is factually false.",

            "Judge factual truth, not completeness or answer length.",

            "A factually accurate answer must score highly even if "
            "it is shorter than the expected output.",

            "Do NOT deduct for brevity, missing elaboration, or "
            "omitted points.",

            "Additional correct information must NEVER lower the score.",
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "All stated claims are factually correct and "
                    "consistent with the expected output. "
                    "No contradictions. Brevity is acceptable."
                ),
            ),

            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "Mostly correct but contains one or more minor "
                    "inaccuracies."
                ),
            ),

            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "Contains a clear factual error or contradicts "
                    "the expected output."
                ),
            ),
        ],

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )


    # --------------------------------------------------------
    # 3b. COMPLETENESS
    # --------------------------------------------------------

    completeness = GEval(
        name="Completeness",

        evaluation_steps=[
            "Identify the key points contained in the expected output.",

            "Check how many of those key points are addressed "
            "in the actual output.",

            "Penalize the actual output for each key point from "
            "the expected output that it omits or only partially covers.",

            "Judge coverage only.",

            "Do NOT lower the score because a covered point is "
            "stated incorrectly. Factual correctness is judged separately.",

            "Do NOT penalize additional correct information beyond "
            "the expected output.",
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Addresses essentially all key points in "
                    "the expected output."
                ),
            ),

            Rubric(
                score_range=(5, 8),
                expected_outcome=(
                    "Covers the main key points but misses "
                    "one or more secondary points."
                ),
            ),

            Rubric(
                score_range=(0, 4),
                expected_outcome=(
                    "Misses several important key points and "
                    "only partially covers the expected answer."
                ),
            ),
        ],

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )


    # --------------------------------------------------------
    # 3c. STYLE
    # --------------------------------------------------------

    style = GEval(
        name="Style",

        evaluation_steps=[
            "Judge only the teaching style and tone of the "
            "actual output.",

            "Do NOT judge factual correctness or completeness.",

            "Reward an intuitive explanatory tone using plain "
            "language before introducing technical terminology.",

            "Reward technical terms that are briefly explained "
            "when they are introduced.",

            "Reward direct conversational prose that sounds like "
            "a teacher explaining the concept naturally.",

            "An analogy or concrete example is a bonus for "
            "abstract concepts, but is not required.",

            "Penalize stiff, bureaucratic, robotic, unexplained, "
            "or overly jargon-heavy answers.",

            "Do NOT reward or penalize based on answer length.",
        ],

        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome=(
                    "Clearly conversational and intuitive teaching "
                    "style. Explains ideas naturally before formalizing them."
                ),
            ),

            Rubric(
                score_range=(7, 8),
                expected_outcome=(
                    "Clear, conversational, and well explained. "
                    "Fully acceptable even without an analogy."
                ),
            ),

            Rubric(
                score_range=(4, 6),
                expected_outcome=(
                    "Understandable but somewhat flat, formal, "
                    "or list-heavy."
                ),
            ),

            Rubric(
                score_range=(0, 3),
                expected_outcome=(
                    "Dry, stiff, robotic, jargon-heavy, or "
                    "poorly explained."
                ),
            ),
        ],

        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],

        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False,
    )


    # ========================================================
    # 4. RUN EVALUATION
    # ========================================================

    print("\n========================================")
    print("Starting Application Evaluation")
    print("========================================")

    print("Retriever : RerankingRetriever")
    print("Generator : Cohere Command A")
    print("Judge     : Cohere Command A")

    print("========================================\n")


    result = evaluate(
        test_cases=test_cases,

        metrics=[
            correctness,
            completeness,
            style,
        ],

        # Important for your Windows setup
        # to avoid DeepEval disk-cache issues.
        cache_config=CacheConfig(
            write_cache=False,
            use_cache=False,
        ),

        hyperparameters={
            "retriever": "reranked",

            "fetch_k": 10,
            "top_k": 5,

            "embedding_model": "text-embedding-3-small",

            "chunk_size": 1000,
            "chunk_overlap": 150,

            "reranker": (
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),

            "generator": (
                "cohere/command-a-03-2025"
            ),

            "judge_model": (
                "cohere/command-a-03-2025"
            ),

            "golden_set": GOLDEN_PATH,

            "temperature": 0,
        },
    )

    return result


# ============================================================
# STANDALONE RUNNER
# ============================================================

def run_local():

    rag = RagPipeline(
        fetch_k=10,
        top_k=5,
    )

    return run(rag)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_local()

    print("\n========================================")
    print("Application evaluation completed.")
    print("========================================")