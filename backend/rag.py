from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from context import KNOWLEDGE_BASE

class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = KNOWLEDGE_BASE
        self._build_index()
        print(f"RAG index built with {len(self.chunks)} chunks.")

    def _build_index(self):
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        distances, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def add_chunks(self, new_chunks: list[str]):
        """Dynamically add new context chunks to the index."""
        self.chunks.extend(new_chunks)
        embeddings = self.model.encode(new_chunks, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(embeddings.astype(np.float32))
