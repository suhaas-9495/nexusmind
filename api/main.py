"""
NexusMind v2 — FastAPI Backend
Production-grade REST API replacing Streamlit.
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import upload, query, chat
from auth.jwt_handler import get_current_user
from observability.logger import setup_logging, get_request_logger
from backend.config import settings

setup_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🧠 NexusMind v2 starting up…")
    yield
    logger.info("NexusMind shutting down.")


app = FastAPI(
    title="NexusMind API",
    description="Production-grade Agentic RAG — hybrid retrieval, evaluation, JWT auth",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        get_request_logger().info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed_ms}ms)"
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = str(elapsed_ms)
        return response
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        get_request_logger().error(
            f"[{request_id}] {request.method} {request.url.path} "
            f"→ ERROR ({elapsed_ms}ms): {exc}"
        )
        raise


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/upload-doc", tags=["Documents"])
app.include_router(query.router,  prefix="/query",      tags=["Query"])
app.include_router(chat.router,   prefix="/chat",       tags=["Chat"])

# Auth routes (login/register — public)
from auth import routes as auth_routes
app.include_router(auth_routes.router, prefix="/auth", tags=["Auth"])

# Evaluation routes
from evaluation import routes as eval_routes
app.include_router(eval_routes.router, prefix="/evaluate", tags=["Evaluation"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "2.0.0"}
