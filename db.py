from abc import ABC
from typing import Self

from pymilvus import MilvusClient


class VectorDB(ABC):
    def insert_text(self: Self, text: str, embedding: list[float]) -> None:
        raise NotImplementedError

    def retrieve_text(self: Self, query_embedding: list[float]) -> None:
        raise NotImplementedError


class MilvusDB(VectorDB):
    def __init__(self, db_path: str):
        self.client = MilvusClient(db_path)
        self.collection_name = "knowledge_base"

        self.client.create_collection(
            self.collection_name,
            dimension=1536,  # OpenAI's `text-embedding-ada-002` has 1536 dimensions
            auto_id=True,
            primary_field="id",
            vector_field="embedding",
            text_field="text",
        )

    def retrieve_text(self, query_embedding: list[float]) -> None:
        """Retrieve the most relevant text for a given embedding."""
        return None
