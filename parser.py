import hashlib
import os
import uuid
from typing import Any, Dict, List

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


load_dotenv()


class LegalTextPipeline:
    """
    Pipeline for:
    1. Parsing German legal HTML documents
    2. Creating embeddings
    3. Storing vectors in Qdrant
    """

    def __init__(
        self,
        collection_name: str,
        gemini_api_key: str,
        path: str = "qdrant_local_db",
    ):
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing")

        self.collection_name = collection_name
        self.ai_client = genai.Client(api_key=gemini_api_key)
        self.client = QdrantClient(path=path)

        self.embedding_model = "gemini-embedding-001"
        self.vector_size = 3072

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not exist."""

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    @staticmethod
    def _generate_deterministic_uuid(content: str) -> str:
        """
        Generate stable IDs for vectors.
        This prevents duplicate entries when ingestion is repeated.
        """

        hash_value = hashlib.md5(
            content.encode("utf-8")
        ).hexdigest()

        return str(uuid.UUID(hash_value))

    def parse_legal_file(
        self,
        file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Extract legal paragraphs from German HTML documents.
        """

        with open(file_path, "r", encoding="iso-8859-1") as file:
            soup = BeautifulSoup(file, "html.parser")

        chunks = []
        current_heading = "N/A"

        elements = soup.find_all(["h2", "div"])

        for element in elements:

            if element.name == "h2":
                current_heading = element.get_text(strip=True)
                continue

            classes = element.get("class", [])

            if "jnnorm" not in classes:
                continue

            title_span = element.find(
                "span",
                class_="jnentitel"
            )

            num_span = element.find(
                "span",
                class_="jnenbez"
            )

            if not title_span:
                continue

            title = title_span.get_text(strip=True)

            paragraph_num = (
                num_span.get_text(strip=True)
                if num_span
                else "N/A"
            )

            content_parts = []

            for child in element.find_all(["div", "h4"]):

                if "Fußnote" in child.get_text():
                    break

                child_classes = child.get("class", [])

                if any(
                    c in child_classes
                    for c in ["jurAbsatz", "jurText"]
                ):
                    text = child.get_text(strip=True)

                    if text and "Nichtamtliches" not in text:
                        content_parts.append(text)

            body = " ".join(content_parts)

            if not body:
                continue

            chunks.append(
                {
                    "paragraph_num": paragraph_num,
                    "title": title,
                    "heading_area": current_heading,
                    "law_text": body,
                    "status": "active",
                }
            )

        return chunks

    def process_and_upsert(
        self,
        raw_chunks: List[Dict[str, Any]],
        batch_size: int = 50,
    ) -> None:
        """
        Generate embeddings and store documents in Qdrant.
        """

        for i in range(0, len(raw_chunks), batch_size):

            batch = raw_chunks[i:i + batch_size]

            texts_to_embed = [
                (
                    f"{chunk['heading_area']} "
                    f"{chunk['paragraph_num']} "
                    f"{chunk['title']} "
                    f"{chunk['law_text']}"
                )
                for chunk in batch
            ]

            response = self.ai_client.models.embed_content(
                model=self.embedding_model,
                contents=texts_to_embed,
            )

            points = [
                PointStruct(
                    id=self._generate_deterministic_uuid(
                        chunk["paragraph_num"] + chunk["title"]
                    ),
                    vector=response.embeddings[idx].values,
                    payload=chunk,
                )
                for idx, chunk in enumerate(batch)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )


if __name__ == "__main__":

    pipeline = LegalTextPipeline(
        collection_name="german_penal_code",
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
    )

    parsed_data = pipeline.parse_legal_file(
        "StGB - Strafgesetzbuch.html"
    )

    print(f"Items found: {len(parsed_data)}")

    for item in parsed_data:
        print(
            f"{item['paragraph_num']} - {item['title'][:30]}"
        )

    if parsed_data:
        pipeline.process_and_upsert(parsed_data)
        print("Done.")