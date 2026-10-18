import hmac
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents 
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit

router = APIRouter(prefix="/api")
settings = get_settings()

logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),

        }

    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while answering. Please try again, or contact IT support.",
        ) from exc



MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


def check_admin_key(provided: str) -> None:
    if settings.app_env != "development" and settings.admin_api_key.startswith("change-me"):
        raise HTTPException(status_code=503, detail="Admin key is not configured on the server")
    if not hmac.compare_digest(provided.encode("utf-8"), settings.admin_api_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    check_admin_key(x_admin_key)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum is {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name
    dest.write_bytes(data)

    try:
        docs = load_file(dest)
        chunks = chunk_documents(docs)
    except Exception:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Could not read this file. Make sure it is a valid, text-based document.")

    if not chunks:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No readable text found in this file")

    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}