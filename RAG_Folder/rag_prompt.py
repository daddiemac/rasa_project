def build_rag_prompt(context_docs, query):
    context = "\n".join(f"- {doc['text']}" for doc in context_docs)

    return f"""
You are an agricultural assistant for Nigerian farmers.
Answer using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""
