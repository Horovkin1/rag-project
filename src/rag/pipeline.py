"""
RAG Pipeline
Поддерживает два режима:
  - LOCAL:  in-memory (для разработки, без внешних зависимостей)
  - QDRANT: облачный Qdrant (для продакшена)
Переключение через env: VECTOR_STORE=local|qdrant
"""

import os
import sys
import uuid
import time
import json
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─── Структуры данных ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    content: str
    metadata: dict
    embedding: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: dict
    score: float


# ─── Embeddings ───────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """
    Провайдер эмбеддингов.
    Приоритет: OpenAI → sentence-transformers → random (dev-заглушка)
    """

    def __init__(self):
        self._st_model = None
        self.provider = self._detect_provider()
        print(f"📐 Embedding provider: {self.provider}", file=sys.stderr)

        # Загружаем ST модель сразу при старте (не при первом запросе)
        if self.provider == "sentence_transformers":
            self._load_st_model()

    def _detect_provider(self) -> str:
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        try:
            import sentence_transformers  # noqa
            return "sentence_transformers"
        except ImportError:
            pass
        return "random"

    def _load_st_model(self):
        from sentence_transformers import SentenceTransformer
        print("⏳ Загрузка sentence-transformers модели (первый раз ~30 сек)...", file=sys.stderr)
        self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Модель загружена!", file=sys.stderr)

    async def embed(self, text: str) -> list[float]:
        if self.provider == "openai":
            return await self._embed_openai(text)
        elif self.provider == "sentence_transformers":
            # Запускаем синхронный ST в thread pool чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._embed_st_sync, text)
        else:
            return self._embed_random(text)

    async def _embed_openai(self, text: str) -> list[float]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def _embed_st_sync(self, text: str) -> list[float]:
        return self._st_model.encode(text, normalize_embeddings=True).tolist()

    def _embed_random(self, text: str) -> list[float]:
        """Dev-заглушка."""
        import hashlib
        import math
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [math.sin(seed * (i + 1)) * 0.5 for i in range(384)]


# ─── Vector Store ─────────────────────────────────────────────────────────────

class LocalVectorStore:
    """In-memory хранилище — для разработки."""

    def __init__(self):
        self._chunks: dict[str, Chunk] = {}

    async def upsert(self, chunk: Chunk) -> None:
        self._chunks[chunk.id] = chunk

    async def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        import math

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x**2 for x in a))
            nb = math.sqrt(sum(x**2 for x in b))
            return dot / (na * nb + 1e-8)

        scored = [
            SearchResult(
                chunk_id=c.id,
                content=c.content,
                metadata=c.metadata,
                score=cosine(embedding, c.embedding),
            )
            for c in self._chunks.values()
            if c.embedding
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)


class QdrantVectorStore:
    """Облачный Qdrant."""

    COLLECTION = "knowledge_base"

    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        url = os.environ["QDRANT_URL"]
        api_key = os.getenv("QDRANT_API_KEY")
        self.dim = int(os.getenv("EMBEDDING_DIM", 384))

        print(f"🔗 Подключение к Qdrant: {url}", file=sys.stderr)
        self.client = QdrantClient(url=url, api_key=api_key)

        if not self.client.collection_exists(self.COLLECTION):
            self.client.create_collection(
                self.COLLECTION,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            print(f"✅ Коллекция '{self.COLLECTION}' создана (dim={self.dim})", file=sys.stderr)
        else:
            print(f"✅ Коллекция '{self.COLLECTION}' уже существует", file=sys.stderr)

    async def upsert(self, chunk: Chunk) -> None:
        from qdrant_client.models import PointStruct
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.COLLECTION,
                points=[
                    PointStruct(
                        id=self._str_id(chunk.id),
                        vector=chunk.embedding,
                        payload={"content": chunk.content, **chunk.metadata},
                    )
                ],
            )
        )

    async def search(self, embedding: list[float], top_k: int) -> list[SearchResult]:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.client.query_points(
                collection_name=self.COLLECTION,
                query=embedding,
                limit=top_k,
            )
        )
        return [
            SearchResult(
                chunk_id=str(h.id),
                content=h.payload.get("content", ""),
                metadata={k: v for k, v in h.payload.items() if k != "content"},
                score=h.score,
            )
            for h in result.points
        ]

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.COLLECTION)
            return info.points_count or 0
        except Exception:
            return 0

    def _str_id(self, s: str) -> str:
        """Qdrant требует UUID или uint64 — конвертируем строку в UUID."""
        import hashlib
        h = hashlib.md5(s.encode()).hexdigest()
        return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ─── Text Splitter ────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        if i >= len(words):
            break
    return chunks or [text]


# ─── Metadata Store ───────────────────────────────────────────────────────────

class MetadataStore:
    """Персистентное хранилище метаданных документов (JSON-файл)."""

    def __init__(self):
        data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        self._path = data_dir / "docs.json"
        self._docs: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        self._path.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add(self, doc_id: str, chunks: int, metadata: dict) -> None:
        self._docs.append({
            "doc_id": doc_id,
            "chunks": chunks,
            "title": metadata.get("title", ""),
            "source": metadata.get("source", ""),
            "filename": metadata.get("filename", ""),
            "created_at": time.time(),
        })
        self._save()

    def list_docs(self) -> list[dict]:
        return list(self._docs)

    def count(self) -> int:
        return len(self._docs)

    def delete(self, doc_id: str) -> bool:
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["doc_id"] != doc_id]
        if len(self._docs) < before:
            self._save()
            return True
        return False


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:

    def __init__(self):
        self.embedder = EmbeddingProvider()
        store_type = os.getenv("VECTOR_STORE", "local")
        self.store = QdrantVectorStore() if store_type == "qdrant" else LocalVectorStore()
        self.meta = MetadataStore()
        print(f"🗄️  Vector store: {store_type}, docs: {self.meta.count()}", file=sys.stderr)

    async def ingest(self, content: str, metadata: dict | None = None) -> dict:
        metadata = metadata or {}
        chunks_text = split_text(content)
        doc_id = str(uuid.uuid4())[:8]

        for i, chunk_text in enumerate(chunks_text):
            embedding = await self.embedder.embed(chunk_text)
            chunk = Chunk(
                id=f"{doc_id}-{i}",
                content=chunk_text,
                metadata={**metadata, "doc_id": doc_id, "chunk_index": i},
                embedding=embedding,
            )
            await self.store.upsert(chunk)

        self.meta.add(doc_id, len(chunks_text), metadata)
        return {"doc_id": doc_id, "chunks": len(chunks_text), "status": "indexed"}

    async def delete(self, doc_id: str) -> dict:
        """Удалить документ из векторного хранилища и метаданных."""
        removed = self.meta.delete(doc_id)
        # Удаляем чанки из Qdrant если он используется
        if isinstance(self.store, QdrantVectorStore):
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.store.client.delete(
                    collection_name=self.store.COLLECTION,
                    points_selector=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    ),
                )
            )
        return {"doc_id": doc_id, "deleted": removed}

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = await self.embedder.embed(query)
        results = await self.store.search(embedding, top_k)
        return [
            {
                "content": r.content,
                "score": round(r.score, 4),
                "metadata": r.metadata,
            }
            for r in results
        ]

    async def get_stats(self) -> dict:
        return {
            "documents": self.meta.count(),
            "chunks": self.store.count(),
            "embedding_provider": self.embedder.provider,
            "vector_store": type(self.store).__name__,
        }

    def get_config(self) -> dict:
        return {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "top_k_default": 5,
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", 384)),
            "vector_store": os.getenv("VECTOR_STORE", "local"),
        }
