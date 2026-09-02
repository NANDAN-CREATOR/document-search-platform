import logging, uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from agents.rag_pipeline import AgenticRAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter()
_pipeline: Optional[AgenticRAGPipeline] = None

def get_pipeline() -> AgenticRAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AgenticRAGPipeline()
    return _pipeline

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SourceReference(BaseModel):
    filename: str
    score: float

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceReference]
    chunks_retrieved: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "document-search"
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str
    model: str
    choices: List[ChatChoice]

# --- Search endpoint ---
@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = get_pipeline().run(request.query)
        return SearchResponse(
            query=result["query"],
            answer=result["answer"],
            sources=[SourceReference(**s) for s in result["sources"]],
            chunks_retrieved=result["chunks_retrieved"],
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- OpenWebUI compatible models endpoint ---
@router.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "document-search",
                "object": "model",
                "created": 1700000000,
                "owned_by": "document-search-platform",
            }
        ]
    }

# --- OpenWebUI compatible chat completions endpoint ---
@router.post("/chat/completions", response_model=ChatResponse)
async def openwebui_chat(request: ChatRequest):
    last_user_msg = next((m for m in reversed(request.messages) if m.role == "user"), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found")
    try:
        result = get_pipeline().run(last_user_msg.content)
        sources_text = "\n".join([f"- {s['filename']} (score: {s['score']:.3f})" for s in result["sources"]])
        full_answer = f"{result['answer']}\n\n**Sources:**\n{sources_text}" if result["sources"] else result["answer"]
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            model=request.model or "document-search",
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=full_answer), finish_reason="stop")],
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
