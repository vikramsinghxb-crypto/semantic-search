import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# -----------------------------
# Load Models
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# Request Schema
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6

# -----------------------------
# Fake Documents (85)
# -----------------------------
documents = [
    {
        "id": i,
        "content": f"This is customer review number {i}. The battery life is decent but camera performance varies. Some users report heating issues.",
        "metadata": {"source": "reviews"}
    }
    for i in range(85)
]

# Precompute embeddings
doc_texts = [doc["content"] for doc in documents]
doc_embeddings = embedding_model.encode(doc_texts)

# -----------------------------
# Normalize Function (0–1)
# -----------------------------
def normalize(scores):
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 for _ in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

# -----------------------------
# Search Endpoint
# -----------------------------
@app.post("/search")
def search(request: SearchRequest):

    start_time = time.time()

    # -------- Stage 1: Vector Retrieval --------
    query_embedding = embedding_model.encode([request.query])

    similarities = cosine_similarity(
        query_embedding,
        doc_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:request.k]

    candidates = [
        {
            "doc": documents[idx],
            "initial_score": float(similarities[idx])
        }
        for idx in top_indices
    ]

    # -------- Stage 2: Re-ranking --------
    if request.rerank:

        pairs = [
            (request.query, c["doc"]["content"])
            for c in candidates
        ]

        rerank_scores = rerank_model.predict(pairs)

        normalized_scores = normalize(rerank_scores)

        for i in range(len(candidates)):
            candidates[i]["score"] = float(normalized_scores[i])

        candidates = sorted(
            candidates,
            key=lambda x: x["score"],
            reverse=True
        )

        final_results = candidates[:request.rerankK]

        reranked_flag = True

    else:
        final_results = candidates
        reranked_flag = False

    latency = int((time.time() - start_time) * 1000)

    return {
        "results": [
            {
                "id": r["doc"]["id"],
                "score": round(r["score"], 3),
                "content": r["doc"]["content"],
                "metadata": r["doc"]["metadata"]
            }
            for r in final_results
        ],
        "reranked": reranked_flag,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
