"""
RAG Index Builder
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


def build_index(
    embedding_model_name="all-MiniLM-L6-v2",
    meta_path="data/metadata.json",
    out_index="data/faiss_index.bin"
):

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    texts = [doc["text"] for doc in documents]
    print(f"📄 Loaded {len(texts)} documents")

    # Load model safely
    try:
        model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        print("❌ Failed to load embedding model")
        print(e)
        return

    # Encode
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, out_index)

    print("✅ FAISS index created")
    print(f"📦 Index size: {index.ntotal}")
    print(f"💾 Saved to: {out_index}")


if __name__ == "__main__":
    build_index()
