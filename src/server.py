"""
MCP Server с RAG-системой
Транспорт: stdio (для Claude Desktop) + HTTP/SSE (для облака)
"""

import asyncio
import json
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptMessage,
    GetPromptResult,
)

from tools.search import search_documents
from tools.ingest import ingest_document
from tools.calculator import calculate
from rag.pipeline import RAGPipeline

# ─── Инициализация ────────────────────────────────────────────────────────────

app = Server("rag-mcp-server")
rag = RAGPipeline()

# ─── Инструменты (Tools) ──────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_knowledge_base",
            description=(
                "Поиск по базе знаний с помощью семантического поиска (RAG). "
                "Используй, когда нужно найти информацию из загруженных документов."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "top_k": {"type": "integer", "description": "Количество результатов", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest_document",
            description="Загрузить и индексировать документ в базу знаний.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Текст документа"},
                    "metadata": {"type": "object", "description": "Метаданные: {title, source, tags}"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="calculate",
            description="Вычислить математическое выражение. Пример: '2 + 2 * 10'",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Математическое выражение"}
                },
                "required": ["expression"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "search_knowledge_base":
        results = await rag.search(arguments["query"], top_k=arguments.get("top_k", 5))
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]
    elif name == "ingest_document":
        result = await rag.ingest(arguments["content"], arguments.get("metadata", {}))
        return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
    elif name == "calculate":
        return [TextContent(type="text", text=calculate(arguments["expression"]))]
    return [TextContent(type="text", text=f"Неизвестный инструмент: {name}")]


# ─── Ресурсы ──────────────────────────────────────────────────────────────────

@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(uri="rag://stats", name="Статистика базы знаний", mimeType="application/json"),
        Resource(uri="rag://config", name="Конфигурация RAG", mimeType="application/json"),
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "rag://stats":
        return json.dumps(await rag.get_stats(), ensure_ascii=False, indent=2)
    elif uri == "rag://config":
        return json.dumps(rag.get_config(), ensure_ascii=False, indent=2)
    return json.dumps({"error": "Ресурс не найден"})


# ─── Промпты ──────────────────────────────────────────────────────────────────

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(name="rag_answer", description="Ответить на вопрос через базу знаний",
               arguments=[{"name": "question", "description": "Вопрос", "required": True}]),
        Prompt(name="summarize_topic", description="Саммари по теме из базы знаний",
               arguments=[{"name": "topic", "description": "Тема", "required": True}]),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    args = arguments or {}
    if name == "rag_answer":
        q = args.get("question", "")
        return GetPromptResult(description="RAG-ответ", messages=[
            PromptMessage(role="user", content=TextContent(type="text",
                text=f"Используй search_knowledge_base для поиска, затем ответь на вопрос:\n\n{q}"))
        ])
    elif name == "summarize_topic":
        t = args.get("topic", "")
        return GetPromptResult(description="Саммари", messages=[
            PromptMessage(role="user", content=TextContent(type="text",
                text=f"Найди информацию о '{t}' через search_knowledge_base и составь структурированное саммари."))
        ])
    return GetPromptResult(description="", messages=[])


# ─── REST API для веб-UI ──────────────────────────────────────────────────────
# Добавляем к HTTP-серверу дополнительные эндпоинты

def add_rest_routes(routes, rag_instance):
    """Добавить REST-роуты для веб-UI."""
    from pathlib import Path
    from starlette.requests import Request
    from starlette.responses import JSONResponse, FileResponse
    from starlette.routing import Route, Mount
    from starlette.staticfiles import StaticFiles

    async def index(request: Request):
        ui_path = Path(__file__).parent.parent / "ui" / "index.html"
        return FileResponse(str(ui_path))

    async def tool_call(request: Request):
        tool_name = request.path_params["tool_name"]
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            if tool_name == "search_knowledge_base":
                result = await rag_instance.search(body.get("query", ""), top_k=body.get("top_k", 5))
            elif tool_name == "ingest_document":
                result = await rag_instance.ingest(body.get("content", ""), body.get("metadata", {}))
            elif tool_name == "calculate":
                from tools.calculator import calculate as _calc
                result = _calc(body.get("expression", ""))
            else:
                return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)
            return JSONResponse(result if isinstance(result, (dict, list)) else {"result": result})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def upload_file(request: Request):
        """Принять файл (PDF/DOCX/XLSX/TXT/...) и проиндексировать."""
        try:
            form = await request.form()
            file = form.get("file")
            if file is None:
                return JSONResponse({"error": "Файл не передан"}, status_code=400)

            filename = file.filename or "document"
            title = form.get("title") or filename
            source = form.get("source") or "upload"

            content_bytes = await file.read()
            if not content_bytes:
                return JSONResponse({"error": "Файл пуст"}, status_code=400)

            from tools.file_parser import parse_file
            text = parse_file(filename, content_bytes)

            if not text.strip():
                return JSONResponse({"error": "Не удалось извлечь текст из файла"}, status_code=400)

            result = await rag_instance.ingest(
                text,
                {"title": title, "source": source, "filename": filename},
            )
            result["chars"] = len(text)
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def chat_endpoint(request: Request):
        try:
            import httpx as _httpx
            body = await request.json()
            query = body.get("query", "")
            context = body.get("context", "")

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return JSONResponse({"error": "GROQ_API_KEY not set"}, status_code=503)

            system = (
                "Ты — умный помощник с доступом к базе знаний. "
                "Отвечай только на основе предоставленного контекста. "
                "Если контекст не содержит ответа — честно скажи об этом. "
                "Отвечай на том же языке, что и вопрос. Будь лаконичен и точен."
            )
            user_msg = (
                f"Контекст из базы знаний:\n\n{context}\n\n---\n\nВопрос: {query}"
                if context else query
            )

            payload = {
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 1024,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
            }

            async with _httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                data = r.json()
                if "error" in data:
                    return JSONResponse({"error": data["error"]["message"]}, status_code=500)
                answer = data["choices"][0]["message"]["content"]
            return JSONResponse({"answer": answer})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    async def stats_endpoint(request: Request):
        return JSONResponse(await rag_instance.get_stats())

    async def config_endpoint(request: Request):
        return JSONResponse(rag_instance.get_config())

    async def documents_list(request: Request):
        docs = rag_instance.meta.list_docs()
        return JSONResponse(docs)

    async def document_delete(request: Request):
        doc_id = request.path_params["doc_id"]
        result = await rag_instance.delete(doc_id)
        return JSONResponse(result)

    ui_dir = Path(__file__).parent.parent / "ui"

    routes.extend([
        Route("/",                        endpoint=index,           methods=["GET"]),
        Route("/tool/{tool_name}",        endpoint=tool_call,       methods=["POST"]),
        Route("/upload",                  endpoint=upload_file,     methods=["POST"]),
        Route("/chat",                    endpoint=chat_endpoint,   methods=["POST"]),
        Route("/stats",                   endpoint=stats_endpoint,  methods=["GET"]),
        Route("/config",                  endpoint=config_endpoint, methods=["GET"]),
        Route("/documents",               endpoint=documents_list,  methods=["GET"]),
        Route("/document/{doc_id}",       endpoint=document_delete, methods=["DELETE"]),
        Mount("/ui",                      app=StaticFiles(directory=str(ui_dir), html=True)),
    ])


# ─── Запуск ───────────────────────────────────────────────────────────────────

async def run_stdio():
    """stdio транспорт — для Claude Desktop."""
    from mcp.server.stdio import stdio_server
    print("▶ Запуск в режиме stdio (Claude Desktop)", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def run_http():
    """HTTP/SSE транспорт — для облака и браузера."""
    import webbrowser
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    import uvicorn

    sse_transport = SseServerTransport("/messages")

    async def handle_sse(request: Request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(streams[0], streams[1], app.create_initialization_options())

    async def health(request: Request):
        return JSONResponse({"status": "ok", "tools": 3})

    routes = [
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=sse_transport.handle_post_message),
        Route("/health", endpoint=health),
    ]
    add_rest_routes(routes, rag)

    starlette_app = Starlette(
        routes=routes,
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        ],
    )

    port = int(os.getenv("PORT", 8000))
    url = f"http://localhost:{port}"
    print(f"🚀 MCP сервер запущен на порту {port}", file=sys.stderr)
    print(f"   UI:     {url}", file=sys.stderr)
    print(f"   SSE:    {url}/sse", file=sys.stderr)
    print(f"   Health: {url}/health", file=sys.stderr)

    async def _open_browser():
        await asyncio.sleep(1.5)
        webbrowser.open(url)

    asyncio.ensure_future(_open_browser())

    config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Режим определяется переменной окружения:
    # MCP_TRANSPORT=stdio  — для Claude Desktop (по умолчанию если не задан PORT)
    # MCP_TRANSPORT=http   — для облака/браузера
    transport = os.getenv("MCP_TRANSPORT", "http")
    if transport == "stdio":
        asyncio.run(run_stdio())
    else:
        asyncio.run(run_http())
