"""
RAG Retriever
- Loads FAISS index
- Embeds query
- Retrieves top-k documents
- Supports metadata filtering
"""

import json

from sentence_transformers import SentenceTransformer
import faiss


class RAGRetriever:
    def __init__(
        self,
        embedding_model_name="all-MiniLM-L6-v2",
        index_path="data/faiss_index.bin",
        metadata_path="data/metadata.json"
    ):
        # Load embedding model safely
        try:
            self.model = SentenceTransformer(embedding_model_name)
        except Exception as e:
            raise RuntimeError(
                "❌ Failed to load embedding model. "
                "Check internet access or local model path.\n"
                f"Details: {e}"
            )

        # Load FAISS index
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load FAISS index from {index_path}\n{e}"
            )

        # Load metadata
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load metadata from {metadata_path}\n{e}"
            )

        if self.index.ntotal != len(self.documents):
            raise ValueError(
                "❌ FAISS index size does not match metadata size"
            )

    def search(
        self,
        query,
        top_k=5,
        language=None,
        crop=None,
        category=None,
        difficulty=None
    ):
        if not query:
            return []

        # Embed query
        try:
            query_vec = self.model.encode(
                [query],
                convert_to_numpy=True
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to embed query\n{e}"
            )

        # FAISS search
        distances, indices = self.index.search(query_vec, top_k * 5)

        results = []

        for idx in indices[0]:
            doc = self.documents[idx]
            meta = doc.get("metadata", {})

            # Apply filters
            if language and meta.get("language") != language:
                continue
            if crop and meta.get("crop") != crop:
                continue
            if category and meta.get("category") != category:
                continue
            if difficulty and meta.get("difficulty") != difficulty:
                continue

            results.append({
                "doc_id": doc.get("doc_id"),
                "text": doc.get("text"),
                "metadata": meta
            })

            if len(results) >= top_k:
                break

        return results


# ------------------------------
# CLI test
# ------------------------------
if __name__ == "__main__":
    retriever = RAGRetriever()

    query = "Why is early planting important for maize?"
    results = retriever.search(
        query=query,
        top_k=3,
        language="en",
        crop="maize"
    )

    for i, res in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(res["text"])
        print("Metadata:", res["metadata"])
