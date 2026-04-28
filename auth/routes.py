"""
NexusMind v2 — Auth Routes
/auth/register and /auth/login endpoints.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, EmailStr

from auth.jwt_handler import hash_password, verify_password, create_access_token

router = APIRouter()
logger = logging.getLogger(__name__)

# Simple JSON file store — replace with a real DB (PostgreSQL etc.) in prod
USER_DB_PATH = Path("logs/users.json")


def _load_users() -> dict:
    if USER_DB_PATH.exists():
        return json.loads(USER_DB_PATH.read_text())
    return {}


def _save_users(users: dict) -> None:
    USER_DB_PATH.parent.mkdir(exist_ok=True)
    USER_DB_PATH.write_text(json.dumps(users, indent=2))


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(req: RegisterRequest):
    users = _load_users()
    if req.username in users:
        raise HTTPException(status_code=400, detail="Username already taken.")

    users[req.username] = {
        "user_id":  req.username,
        "password": hash_password(req.password),
        "docs": [],
    }
    _save_users(users)
    token = create_access_token(req.username)
    logger.info(f"New user registered: {req.username}")
    return TokenResponse(access_token=token, user_id=req.username)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    users = _load_users()
    user = users.get(req.username)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password."
        )
    token = create_access_token(req.username)
    logger.info(f"User logged in: {req.username}")
    return TokenResponse(access_token=token, user_id=req.username)
