"""Инструмент поиска — обёртка для RAG pipeline."""
# Логика поиска реализована напрямую в server.py через rag.search()
# Этот файл — место для будущих расширений: фильтрация, re-ranking и т.д.

async def search_documents(rag, query: str, top_k: int = 5) -> list[dict]:
    return await rag.search(query, top_k=top_k)
