from abc import ABC, abstractmethod
from typing import List, Optional, Self, Tuple

from pymilvus import MilvusClient


class VectorDB(ABC):
    @abstractmethod
    def insert_text(
        self: Self,
        text: str,
        embedding: List[float],
    ) -> None:
        pass

    @abstractmethod
    def retrieve_text(
        self: Self,
        query_embedding: List[float],
    ) -> Optional[List[Tuple[str, float]]]:
        pass


class MilvusDB(VectorDB):
    def __init__(self: Self, db_path: str):
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

    def insert_text(self: Self, text: str, embedding: List[float]) -> None:
        """Insert a text-embedding pair into the database."""
        self.client.insert(
            self.collection_name,
            [
                {
                    "text": text,
                    "embedding": embedding,
                },
            ],
        )
        self.client.flush([self.collection_name])

    def retrieve_text(
        self: Self, query_embedding: List[float]
    ) -> Optional[List[Tuple[str, float]]]:
        """Retrieve the most relevant text for a given embedding."""

        self.client.load_collection(self.collection_name)

        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=1,
            output_fields=["text"],
        )

        return search_res[0] if search_res else None
