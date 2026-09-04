import os
import json

import cohere
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM


class CohereJudge(DeepEvalBaseLLM):

    def __init__(self, model="command-a-03-2025"):
        self.model_name = model

        self.client = cohere.ClientV2(
            api_key=os.getenv("COHERE_API_KEY")
        )

    def load_model(self):
        return self.client

    def get_model_name(self):
        return self.model_name

    def generate(
        self,
        prompt: str,
        schema: BaseModel | None = None
    ):

        # --------------------------------------------------
        # Normal text generation
        # --------------------------------------------------

        if schema is None:

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response.message.content[0].text

        # --------------------------------------------------
        # Structured JSON generation for DeepEval
        # --------------------------------------------------

        json_schema = schema.model_json_schema()

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\n"
                        + "Return ONLY a valid JSON object "
                        "that follows the provided schema."
                    ),
                }
            ],
            response_format={
                "type": "json_object",
                "schema": json_schema,
            },
        )

        content = response.message.content[0].text

        # Convert Cohere JSON string → Python dict
        result = json.loads(content)

        # Convert dict → Pydantic object expected by DeepEval
        return schema(**result)

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None
    ):

        # Cohere's client can be used synchronously here.
        # DeepEval will still receive the expected schema object.

        return self.generate(prompt, schema)