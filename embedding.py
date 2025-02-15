import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_embedding(text: str) -> list[float]:
    """Generate an embedding for the given text using OpenAI."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
    )
    return response.data[0].embedding
