import os
import glob
import json
import random

from dotenv import load_dotenv

import litellm
litellm.drop_params = True

from deepeval.synthesizer import Synthesizer
from deepeval.models import LiteLLMModel
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not COHERE_API_KEY:
    raise ValueError(
        "COHERE_API_KEY not found. "
        "Make sure it exists in your .env file."
    )


# ============================================================
# 2. DEFINE GENERATION MODEL
# ============================================================

model = LiteLLMModel(
    model="cohere/north-mini-code:free",
    api_key=COHERE_API_KEY,
)


# ============================================================
# 3. LOAD + CLEAN VTT FILES
# ============================================================

def load_chunks():

    texts = []

    for path in glob.glob("data/*.vtt"):

        with open(path, "r", encoding="utf-8") as f:

            lines = [
                line.strip()
                for line in f
                if (
                    line.strip()
                    and line.strip() != "WEBVTT"
                    and "-->" not in line
                )
            ]

        texts.append(" ".join(lines))

    # Combine all VTT files
    combined_text = "\n\n".join(texts)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text(combined_text)


# ============================================================
# 4. LOAD CHUNKS
# ============================================================

chunks = load_chunks()

if not chunks:
    raise ValueError("No chunks found. Check your data/*.vtt files.")

print(f"Loaded {len(chunks)} chunks.")


# ============================================================
# 5. SAMPLE CHUNKS
# ============================================================

sample = random.sample(
    chunks,
    min(15, len(chunks))
)

contexts = [[chunk] for chunk in sample]

print(f"Generating goldens from {len(contexts)} contexts...")


# ============================================================
# 6. GENERATE GOLDENS
# ============================================================

synthesizer = Synthesizer(
    model=model
)

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)


# ============================================================
# 7. CONVERT TO YOUR JSON SCHEMA
# ============================================================

rows = []

for i, golden in enumerate(goldens, 1):

    rows.append({
        "id": f"g{i:03d}",
        "query": golden.input,
        "ideal_answer": golden.expected_output,
        "source": "TODO-verify",
    })


# ============================================================
# 8. SAVE
# ============================================================

os.makedirs("goldens", exist_ok=True)

output_path = "goldens/retriever_deepeval_goldens.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        rows,
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"\nWrote {len(rows)} DRAFT goldens -> {output_path}")

print(
    "!! REVIEW EVERY ONE before using: "
    "check grounding, trim padding, fix leading questions."
)