"""FastAPI application entry point."""
from fastapi import FastAPI

app = FastAPI(
    title="Document Search Platform API",
    description="Agentic RAG Document Search Platform",
    version="1.0.0",
)

# TODO: Include routers
# from api.routes import search, ingest, health

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "document-search-platform"}
