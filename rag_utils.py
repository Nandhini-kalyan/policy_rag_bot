import os
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"


def load_policy_chunks(filepath: str):
    """Load policy file and split into chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({"id": idx, "text": chunk})
    return chunks


def get_embeddings(texts):
    """Get embeddings for a list of texts."""
    resp = openai.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors)


def build_index(chunks):
    """Build normalized embedding index."""
    texts = [c["text"] for c in chunks]
    embs = get_embeddings(texts)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    return {"chunks": chunks, "embeddings": embs}


def retrieve_similar(index, query, k: int = 3):
    """Retrieve top-k similar chunks for a query."""
    q_emb = get_embeddings([query])[0]
    q_emb = q_emb / np.linalg.norm(q_emb)

    scores = np.dot(index["embeddings"], q_emb)
    top_idx = np.argsort(scores)[-k:][::-1]

    results = []
    for idx in top_idx:
        results.append(
            {
                "text": index["chunks"][idx]["text"],
                "score": float(scores[idx]),
            }
        )
    return results
