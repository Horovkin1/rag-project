"""Инструмент индексации документов."""

async def ingest_document(rag, content: str, metadata: dict | None = None) -> dict:
    return await rag.ingest(content, metadata or {})
