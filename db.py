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
    ) -> Optional[str]:
        pass


class MilvusDB(VectorDB):
    def __init__(self: Self, db_path: str):
        self.client = MilvusClient(db_path)
        self.collection_name = "knowledge_base"

        self.client.create_collection(
            self.collection_name,
            dimension=1536,
            auto_id=True,
            primary_field="id",
            vector_field="vector",
            text_field="text",
        )

    def insert_text(self: Self, text: str, embedding: List[float]) -> None:
        """Insert a text-embedding pair into the database."""

        self.client.load_collection(self.collection_name)

        self.client.insert(
            self.collection_name, [{"text": text, "vector": embedding}]
        )

        print(f"✅ Inserted: {text}")

    # NOTE Do a better job on what this function should do
    def retrieve_text(
        self: Self, query_embedding: List[float]
    ) -> Optional[str]:
        """Retrieve the most relevant text for a given embedding."""

        self.client.load_collection(self.collection_name)

        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=1,
            output_fields=["text"],
        )

        if not search_res or len(search_res[0]) == 0:
            return None

        return search_res[0][0]["entity"]["text"]
