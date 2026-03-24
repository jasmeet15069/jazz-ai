# server.py
"""
JAZZ Uncensored AI — v5.1 (Agent Jobs Edition)
===============================================
✅ All v5.0 features (SQLite, ChromaDB, RAG, Memory, Vision, Streaming)
✅ Agent Job System — scheduled autonomous browser agents
✅ Naukri.com auto-apply agent (Playwright headless browser)
✅ LinkedIn Easy Apply agent
✅ Dynamic APScheduler — jobs survive restarts, loaded from DB
✅ Job execution logs with full output per run
✅ Encrypted credential storage (Fernet)
✅ CRUD API: /agent-jobs (create, list, get, update, delete, run, toggle, logs)
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite
import fitz  # PyMuPDF
import openpyxl
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from cryptography.fernet import Fernet
from docx import Document as DocxDocument
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from openai import OpenAI
from passlib.context import CryptContext
from pydantic import BaseModel, Field

import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.FileHandler("jazz.log"), logging.StreamHandler()],
)
logger = logging.getLogger("jazz")

# ─── Config ───────────────────────────────────────────────────────────────────
def _env(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    if not val:
        logger.warning(f"[CONFIG] {key} not set")
    return val

SECRET_KEY: str = _env("JWT_SECRET_KEY")
if not SECRET_KEY:
    import secrets as _sec
    SECRET_KEY = _sec.token_urlsafe(64)
    logger.warning("[CONFIG] JWT_SECRET_KEY not set — ephemeral key")

# Fernet encryption for stored credentials
_FERNET_KEY_RAW = os.getenv("FERNET_KEY", "")
if not _FERNET_KEY_RAW:
    _FERNET_KEY_RAW = Fernet.generate_key().decode()
    logger.warning(f"[CONFIG] FERNET_KEY not set — add to .env: FERNET_KEY={_FERNET_KEY_RAW}")
_fernet = Fernet(_FERNET_KEY_RAW.encode() if isinstance(_FERNET_KEY_RAW, str) else _FERNET_KEY_RAW)

def encrypt_creds(data: dict) -> str:
    return _fernet.encrypt(json.dumps(data).encode()).decode()

def decrypt_creds(token: str) -> dict:
    return json.loads(_fernet.decrypt(token.encode()).decode())

ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

ADMIN_EMAIL:    str = os.getenv("ADMIN_EMAIL", "jasmeet.15069@gmail.com")
ADMIN_PASSWORD: str = _env("ADMIN_PASSWORD")
GROQ_API_KEY:   str = _env("GROQ_API_KEY")
HF_TOKEN:       str = _env("HF_TOKEN")
DB_PATH             = os.getenv("DB_PATH", "./jazz.db")

MYSQL_ENABLED = bool(os.getenv("MYSQL_HOST"))
MYSQL_CONFIG: Dict[str, Any] = {
    "host": os.getenv("MYSQL_HOST", ""), "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", ""), "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", ""), "ssl_disabled": False,
    "ssl_verify_cert": False, "connection_timeout": 10, "autocommit": True,
}

ALLOWED_ORIGINS: List[str] = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

CHUNK_SIZE = 500; CHUNK_OVERLAP = 50; TOP_K_RETRIEVAL = 4
WORKSPACE_DIR = Path("workspace"); WORKSPACE_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".html",".css",".js",".py",".txt",".md",".json",".xml",".yaml",".yml",
    ".csv",".sql",".sh",".bat",".conf",".ini",".log",".svg",".ts",".jsx",".tsx",
}
MAX_FILE_SIZE = 10*1024*1024; MAX_IMAGE_SIZE = 20*1024*1024
ALLOWED_IMAGE_TYPES = {"image/jpeg","image/png","image/gif","image/webp"}

SUBSCRIPTION_LIMITS: Dict[str, Dict[str, Any]] = {
    "free":       {"messages_per_day":50,  "documents":5,  "max_file_mb":10,  "mysql_access":False,"image_analysis":True,"agent_jobs":1},
    "pro":        {"messages_per_day":500, "documents":50, "max_file_mb":50,  "mysql_access":True, "image_analysis":True,"agent_jobs":10},
    "enterprise": {"messages_per_day":-1,  "documents":-1, "max_file_mb":100, "mysql_access":True, "image_analysis":True,"agent_jobs":-1},
}

MAX_MEMORIES = 100; _SERVER_START = datetime.utcnow()
_executor = ThreadPoolExecutor(max_workers=8)

# ─── Security ─────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security    = HTTPBearer(auto_error=False)

# ─── SQLite helpers ───────────────────────────────────────────────────────────
def db() -> aiosqlite.Connection:
    return aiosqlite.connect(DB_PATH)

async def db_execute(sql: str, params: tuple = ()) -> None:
    async with db() as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute(sql, params); await conn.commit()

async def db_fetchone(sql: str, params: tuple = ()) -> Optional[Dict]:
    async with db() as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            row = await cur.fetchone(); return dict(row) if row else None

async def db_fetchall(sql: str, params: tuple = ()) -> List[Dict]:
    async with db() as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall(); return [dict(r) for r in rows]

async def db_insert(sql: str, params: tuple = ()) -> int:
    async with db() as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            last_id = cur.lastrowid
        await conn.commit(); return last_id

async def db_count(sql: str, params: tuple = ()) -> int:
    row = await db_fetchone(sql, params)
    return list(row.values())[0] if row else 0

# ─── Schema ───────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
    full_name TEXT DEFAULT '', role TEXT DEFAULT 'client', subscription TEXT DEFAULT 'free',
    memory_enabled INTEGER DEFAULT 1, avatar_url TEXT DEFAULT '',
    preferred_model TEXT DEFAULT 'uncensored',
    created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chat_history (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, message TEXT NOT NULL, response TEXT NOT NULL,
    query_type TEXT DEFAULT 'chat', agent_used TEXT DEFAULT 'general_agent',
    model_used TEXT DEFAULT 'uncensored', response_time_ms INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}', created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, filename TEXT NOT NULL,
    original_name TEXT NOT NULL, file_type TEXT DEFAULT '', file_size INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0, chroma_collection TEXT DEFAULT 'user_documents',
    uploaded_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS user_memories (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
    category TEXT DEFAULT 'general', source TEXT DEFAULT 'manual',
    created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS mysql_query_logs (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, natural_language TEXT NOT NULL,
    generated_sql TEXT NOT NULL, execution_success INTEGER DEFAULT 1,
    row_count INTEGER DEFAULT 0, execution_time_ms INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Agent Jobs ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_jobs (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    name             TEXT NOT NULL,
    description      TEXT DEFAULT '',
    job_type         TEXT NOT NULL,        -- naukri_apply | linkedin_apply | custom
    cron_schedule    TEXT NOT NULL,        -- "0 11 * * *"  (min hr day mon dow)
    time_window_start TEXT DEFAULT '11:00',-- HH:MM  only run inside this window
    time_window_end   TEXT DEFAULT '12:00',-- HH:MM
    credentials_enc  TEXT DEFAULT '',      -- Fernet-encrypted JSON
    parameters       TEXT DEFAULT '{}',   -- JSON search params
    status           TEXT DEFAULT 'idle', -- idle | running | completed | failed
    enabled          INTEGER DEFAULT 1,
    last_run_at      TEXT DEFAULT NULL,
    last_run_status  TEXT DEFAULT NULL,
    next_run_at      TEXT DEFAULT NULL,
    total_runs       INTEGER DEFAULT 0,
    total_applied    INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')), updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS agent_job_logs (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    status       TEXT DEFAULT 'running',  -- running | completed | failed
    output       TEXT DEFAULT '',         -- full timestamped log
    applied_jobs TEXT DEFAULT '[]',       -- JSON array of applied positions
    error        TEXT DEFAULT NULL,
    started_at   TEXT DEFAULT (datetime('now')),
    completed_at TEXT DEFAULT NULL,
    duration_sec INTEGER DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES agent_jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id,created_at);
CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_user  ON user_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_ajob_user ON agent_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_ajlog_job ON agent_job_logs(job_id,started_at);
"""

async def init_db() -> None:
    async with db() as conn:
        await conn.executescript(SCHEMA_SQL); await conn.commit()
    existing = await db_fetchone("SELECT id FROM users WHERE email=?", (ADMIN_EMAIL,))
    if not existing and ADMIN_PASSWORD:
        uid = str(uuid.uuid4()); phash = pwd_context.hash(ADMIN_PASSWORD)
        await db_execute(
            "INSERT INTO users (id,email,password_hash,full_name,role,subscription) VALUES (?,?,?,?,?,?)",
            (uid, ADMIN_EMAIL, phash, "Admin", "admin", "enterprise"),
        )
        logger.info(f"[SQLITE] Admin user created: {ADMIN_EMAIL}")
    logger.info(f"[SQLITE] Initialized ✅  path={DB_PATH}")

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
chroma_client: Optional[chromadb.PersistentClient] = None
documents_collection = None; sqlite_schema_collection = None; _embedding_fn = None
try:
    _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    documents_collection     = chroma_client.get_or_create_collection(name="user_documents",  embedding_function=_embedding_fn)
    sqlite_schema_collection = chroma_client.get_or_create_collection(name="sqlite_schema",   embedding_function=_embedding_fn)
    logger.info("[CHROMADB] Initialized ✅")
except Exception as exc:
    logger.error(f"[CHROMADB] Init failed: {exc} ❌")

# ─── MySQL (optional) ─────────────────────────────────────────────────────────
_mysql_pool = None
def _init_mysql_pool():
    global _mysql_pool
    if not MYSQL_ENABLED: return
    try:
        from mysql.connector import pooling as mp
        _mysql_pool = mp.MySQLConnectionPool(pool_name="jazz_pool",pool_size=5,pool_reset_session=True,**MYSQL_CONFIG)
        logger.info("[MYSQL] Pool created ✅")
    except Exception as exc:
        logger.error(f"[MYSQL] Pool failed: {exc} ❌")

def get_mysql_connection():
    global _mysql_pool
    if _mysql_pool is None: _init_mysql_pool()
    if _mysql_pool is None: return None
    try: return _mysql_pool.get_connection()
    except Exception:
        _init_mysql_pool()
        try: return _mysql_pool.get_connection() if _mysql_pool else None
        except Exception: return None

# ─── Rate Limiter ─────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self): self._window: Dict[str,List[float]] = defaultdict(list)
    def _cleanup(self,uid,window_s=86400):
        now=time.time(); self._window[uid]=[t for t in self._window[uid] if now-t<window_s]
    def check(self,uid,tier):
        limit=SUBSCRIPTION_LIMITS.get(tier,SUBSCRIPTION_LIMITS["free"])["messages_per_day"]
        if limit==-1: return True
        self._cleanup(uid)
        if len(self._window[uid])>=limit: return False
        self._window[uid].append(time.time()); return True
    def usage_today(self,uid): self._cleanup(uid); return len(self._window[uid])
rate_limiter = RateLimiter()

# ─── Pydantic Models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:str=Field(...,min_length=1,max_length=8000)
    model_type:str=Field("uncensored",pattern="^(censored|uncensored)$")
    use_rag:bool=True; image_url:Optional[str]=None

class ChatResponse(BaseModel):
    response:str; response_time:float; agent_used:str
    tokens_used:Optional[int]=None; memory_updated:bool=False; new_memories:List[str]=[]

class UserSignup(BaseModel):
    email:str=Field(...,pattern=r"^[^@]+@[^@]+\.[^@]+$")
    password:str=Field(...,min_length=8); full_name:Optional[str]=None

class UserLogin(BaseModel):
    email:str; password:str

class TokenResponse(BaseModel):
    access_token:str; token_type:str="bearer"; user:Dict[str,Any]

class ProfileUpdate(BaseModel):
    full_name:Optional[str]=None; avatar_url:Optional[str]=None
    preferred_model:Optional[str]=Field(None,pattern="^(censored|uncensored)$")
    memory_enabled:Optional[bool]=None

class DocumentResponse(BaseModel):
    id:str; filename:str; original_name:str; file_type:str
    file_size:int; chunk_count:int; uploaded_at:str

class MySQLQueryRequest(BaseModel):
    question:str=Field(...,min_length=1,max_length=1000)

class MySQLRawQueryRequest(BaseModel):
    sql:str=Field(...,min_length=1)

class MySQLQueryResponse(BaseModel):
    success:bool; question:Optional[str]=None; generated_sql:Optional[str]=None
    results:List[Dict[str,Any]]=[]; row_count:int=0; execution_time_ms:int=0; error:Optional[str]=None

class SubscriptionUpdate(BaseModel):
    user_id:str; tier:str=Field(...,pattern="^(free|pro|enterprise)$")

class FileOperationRequest(BaseModel):
    filename:str; content:Optional[str]=None
    operation:str=Field(...,pattern="^(create|read|update|delete|list)$")

class MemoryCreate(BaseModel):
    content:str=Field(...,min_length=1,max_length=500); category:Optional[str]="general"

class MemoryUpdate(BaseModel):
    content:str=Field(...,min_length=1,max_length=500)

class MemorySettingsUpdate(BaseModel):
    enabled:bool

# Agent Job models
class AgentJobCreate(BaseModel):
    name:               str  = Field(..., min_length=1, max_length=100)
    description:        Optional[str] = ""
    job_type:           str  = Field(..., pattern="^(naukri_apply|linkedin_apply|custom)$")
    cron_schedule:      str  = Field(..., description="5-field cron e.g. '0 11 * * *'")
    time_window_start:  Optional[str] = "11:00"   # HH:MM
    time_window_end:    Optional[str] = "12:00"   # HH:MM
    credentials:        Dict[str, Any] = Field(..., description="Login creds (stored encrypted)")
    parameters:         Optional[Dict[str, Any]] = {}

class AgentJobUpdate(BaseModel):
    name:              Optional[str]             = None
    description:       Optional[str]             = None
    cron_schedule:     Optional[str]             = None
    time_window_start: Optional[str]             = None
    time_window_end:   Optional[str]             = None
    credentials:       Optional[Dict[str, Any]]  = None
    parameters:        Optional[Dict[str, Any]]  = None
    enabled:           Optional[bool]            = None

# ─── Auth Utilities ───────────────────────────────────────────────────────────
def create_access_token(data:Dict, expires_delta:Optional[timedelta]=None)->str:
    to_encode=data.copy()
    to_encode["exp"]=datetime.utcnow()+(expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)

def verify_token(token:str)->Optional[Dict]:
    try: return jwt.decode(token,SECRET_KEY,algorithms=[ALGORITHM])
    except JWTError: return None

async def get_current_user(credentials:Optional[HTTPAuthorizationCredentials]=Depends(security))->Dict:
    if not credentials: raise HTTPException(401,"Authorization header missing",headers={"WWW-Authenticate":"Bearer"})
    payload=verify_token(credentials.credentials)
    if not payload: raise HTTPException(401,"Invalid or expired token",headers={"WWW-Authenticate":"Bearer"})
    return payload

async def require_admin(credentials:Optional[HTTPAuthorizationCredentials]=Depends(security))->Dict:
    if not credentials: raise HTTPException(401,"Authorization header missing")
    payload=verify_token(credentials.credentials)
    if not payload: raise HTTPException(401,"Invalid or expired token")
    if payload.get("role")!="admin": raise HTTPException(403,"Admin access required")
    return payload

async def get_user_subscription(user_id:str)->str:
    row=await db_fetchone("SELECT subscription FROM users WHERE id=?",(user_id,))
    return (row or {}).get("subscription","free")

async def is_memory_enabled(user_id:str)->bool:
    row=await db_fetchone("SELECT memory_enabled FROM users WHERE id=?",(user_id,))
    return bool((row or {}).get("memory_enabled",1))

# ─── Memory System ────────────────────────────────────────────────────────────
async def get_user_memories(user_id:str)->List[Dict]:
    return await db_fetchall("SELECT * FROM user_memories WHERE user_id=? ORDER BY created_at ASC",(user_id,))

def format_memories_for_prompt(memories:List[Dict])->str:
    if not memories: return ""
    lines=["[User Memory:]"]
    for m in memories: lines.append(f"• ({m.get('category','general')}) {m['content']}")
    return "\n".join(lines)

def _sync_extract_memories(user_id:str,user_message:str,ai_response:str)->List[str]:
    import sqlite3 as _s3
    def _cnt():
        with _s3.connect(DB_PATH) as c: r=c.execute("SELECT COUNT(*) FROM user_memories WHERE user_id=?",(user_id,)).fetchone(); return r[0] if r else 0
    def _existing():
        with _s3.connect(DB_PATH) as c: return [r[0].lower() for r in c.execute("SELECT content FROM user_memories WHERE user_id=?",(user_id,)).fetchall()]
    def _ins(mid,content,cat):
        with _s3.connect(DB_PATH) as c: c.execute("INSERT INTO user_memories (id,user_id,content,category,source) VALUES (?,?,?,?,?)",(mid,user_id,content,cat,"auto")); c.commit()
    def _enabled():
        with _s3.connect(DB_PATH) as c: r=c.execute("SELECT memory_enabled FROM users WHERE id=?",(user_id,)).fetchone(); return bool(r[0]) if r else True
    try:
        if not _enabled() or _cnt()>=MAX_MEMORIES: return []
        client,model=get_model_client("censored")
        prompt=f"""Extract 0-3 memorable facts about the USER.
User: {user_message}
AI: {ai_response[:500]}
Return ONLY JSON: [{{"content":"...","category":"preference|profession|goal|personal|technical|general"}}]
If nothing memorable: []"""
        comp=client.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],max_tokens=300,temperature=0.1)
        raw=re.sub(r"```json|```","",comp.choices[0].message.content.strip()).strip()
        extracted=json.loads(raw)
        if not isinstance(extracted,list): return []
        existing=_existing(); new=[]
        for item in extracted[:3]:
            if not isinstance(item,dict) or not item.get("content"): continue
            content=item["content"].strip()[:500]; cat=item.get("category","general")
            if any(content.lower() in e or e in content.lower() for e in existing): continue
            _ins(str(uuid.uuid4()),content,cat); new.append(content)
        return new
    except Exception as exc:
        logger.error(f"[MEMORY] extraction error: {exc}"); return []

async def extract_and_save_memories(user_id,user_message,ai_response):
    loop=asyncio.get_event_loop()
    return await loop.run_in_executor(_executor,_sync_extract_memories,user_id,user_message,ai_response)

# ─── Document Processing ──────────────────────────────────────────────────────
def chunk_text(text,size=CHUNK_SIZE,overlap=CHUNK_OVERLAP):
    chunks,start=[],0
    while start<len(text):
        chunk=text[start:start+size]
        if chunk.strip(): chunks.append(chunk)
        start+=size-overlap
    return chunks

def extract_text_from_pdf(path):
    try:
        with fitz.open(path) as pdf: return "\n".join(p.get_text() for p in pdf)
    except Exception as exc: logger.error(f"[PDF] {exc}"); return ""

def extract_text_from_docx(path):
    try: doc=DocxDocument(path); return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc: logger.error(f"[DOCX] {exc}"); return ""

def extract_text_from_excel(path):
    try:
        wb=openpyxl.load_workbook(path,data_only=True); parts=[]
        for sheet in wb.worksheets:
            parts.append(f"\nSheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                r=" | ".join(str(c) for c in row if c is not None)
                if r.strip(): parts.append(r)
        return "\n".join(parts)
    except Exception as exc: logger.error(f"[EXCEL] {exc}"); return ""

def process_document(path,mime):
    if "pdf" in mime: return extract_text_from_pdf(path)
    if "word" in mime or "docx" in mime: return extract_text_from_docx(path)
    if "excel" in mime or "sheet" in mime: return extract_text_from_excel(path)
    if "csv" in mime:
        try:
            with open(path,"r",encoding="utf-8") as f: return "\n".join(" | ".join(r) for r in csv.reader(f))
        except Exception: return ""
    try: return Path(path).read_text(encoding="utf-8")
    except Exception: return ""

# ─── RAG ──────────────────────────────────────────────────────────────────────
def get_relevant_context(user_id,query,top_k=TOP_K_RETRIEVAL):
    if documents_collection is None: return ""
    try:
        results=documents_collection.query(query_texts=[query],n_results=top_k,where={"user_id":user_id})
        if not results or not results["documents"] or not results["documents"][0]: return ""
        parts=[]
        for i,doc in enumerate(results["documents"][0]):
            meta=results["metadatas"][0][i] if results.get("metadatas") else {}
            parts.append(f"[Source: {meta.get('filename','Unknown')}]\n{doc}")
        return "\n\n---\n\n".join(parts)
    except Exception as exc: logger.error(f"[RAG] {exc}"); return ""

# ─── Schema Indexing ──────────────────────────────────────────────────────────
_last_reindex_at: Optional[str] = None
_last_mysql_reindex_at: Optional[str] = None

def index_sqlite_schema():
    global _last_reindex_at
    if sqlite_schema_collection is None: return False
    import sqlite3 as _s3
    try:
        with _s3.connect(DB_PATH) as conn:
            tables=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
        existing=sqlite_schema_collection.get()
        if existing["ids"]: sqlite_schema_collection.delete(ids=existing["ids"])
        chunks,ids,metas=[],[],[]
        with _s3.connect(DB_PATH) as conn:
            for tbl in tables:
                try:
                    cols=conn.execute(f"PRAGMA table_info(`{tbl}`)").fetchall()
                    col_info=", ".join(f"{c[1]}({c[2]})"+(",PK" if c[5] else "") for c in cols)
                    samples=conn.execute(f"SELECT * FROM `{tbl}` LIMIT 3").fetchall()
                    text=f"Table: {tbl} | Columns: {col_info}"
                    if samples: text+=f" | Samples: {json.dumps([list(r) for r in samples],default=str)}"
                    chunks.append(text); ids.append(f"tbl_{tbl}"); metas.append({"table_name":tbl,"type":"schema"})
                except Exception as exc: logger.warning(f"[SQLITE-SCHEMA] Skipping {tbl}: {exc}")
        if chunks: sqlite_schema_collection.add(documents=chunks,ids=ids,metadatas=metas)
        _last_reindex_at=datetime.utcnow().isoformat()
        logger.info(f"[SQLITE] Schema indexed — {len(chunks)} tables ✅"); return True
    except Exception as exc: logger.error(f"[SQLITE-SCHEMA] Index error: {exc}"); return False

def get_schema_context(query,top_k=3):
    if sqlite_schema_collection is None: return ""
    try:
        results=sqlite_schema_collection.query(query_texts=[query],n_results=top_k)
        if results and results["documents"] and results["documents"][0]: return "\n\n".join(results["documents"][0])
        return ""
    except Exception as exc: logger.error(f"[SQLITE-SCHEMA] context: {exc}"); return ""

def index_mysql_schema():
    global _last_mysql_reindex_at
    if not MYSQL_ENABLED or sqlite_schema_collection is None: return False
    conn=get_mysql_connection()
    if not conn: return False
    try:
        cur=conn.cursor(dictionary=True); cur.execute("SHOW TABLES")
        tables=[list(r.values())[0] for r in cur.fetchall()]
        chunks,ids,metas=[],[],[]
        for tbl in tables:
            try:
                cur.execute(f"DESCRIBE `{tbl}`"); cols=cur.fetchall()
                cur.execute(f"SELECT * FROM `{tbl}` LIMIT 3"); samples=cur.fetchall()
                col_info=", ".join(f"{c['Field']}({c['Type']})"+(",PK" if c["Key"]=="PRI" else "") for c in cols)
                text=f"[MySQL] Table: {tbl} | Columns: {col_info}"
                if samples: text+=f" | Samples: {json.dumps(samples,default=str)}"
                chunks.append(text); ids.append(f"mysql_tbl_{tbl}"); metas.append({"table_name":tbl,"type":"mysql_schema"})
            except Exception as exc: logger.warning(f"[MYSQL] Skipping {tbl}: {exc}")
        if chunks: sqlite_schema_collection.add(documents=chunks,ids=ids,metadatas=metas)
        cur.close(); conn.close(); _last_mysql_reindex_at=datetime.utcnow().isoformat()
        logger.info(f"[MYSQL] Schema indexed — {len(chunks)} tables ✅"); return True
    except Exception as exc: logger.error(f"[MYSQL] Schema index error: {exc}"); return False

DB_KEYWORDS={"table","column","row","select","count","show","describe","database","schema","query","fetch","list","find","get","records","how many","what is in","show me","tell me about","get me","find all","data from"}
def is_db_intent(message): low=message.lower(); return any(kw in low for kw in DB_KEYWORDS)

# ─── AI Clients ───────────────────────────────────────────────────────────────
def get_model_client(model_type):
    if model_type=="censored":
        return OpenAI(base_url="https://api.groq.com/openai/v1",api_key=GROQ_API_KEY),"llama-3.3-70b-versatile"
    return OpenAI(base_url="https://router.huggingface.co/v1",api_key=HF_TOKEN),"dphn/Dolphin-Mistral-24B-Venice-Edition:featherless-ai"

def get_vision_client():
    return OpenAI(base_url="https://router.huggingface.co/v1",api_key=HF_TOKEN),"Qwen/Qwen2.5-VL-7B-Instruct:fastest"

# ─── Image Analysis ───────────────────────────────────────────────────────────
async def analyze_image_pipeline(image_data,user_message,model_type,user_id,memories):
    t0=time.time(); vc,vm=get_vision_client()
    vision_prompt=f"""Analyze this image and return JSON:
{{"description":"...","objects":[],"text_in_image":null,"scene_type":"photo/diagram/screenshot/art/chart/other","technical_details":"...","user_question_relevance":"re: {user_message}"}}"""
    vision_json={}
    try:
        vcomp=vc.chat.completions.create(model=vm,messages=[{"role":"user","content":[{"type":"text","text":vision_prompt},{"type":"image_url","image_url":{"url":image_data}}]}],max_tokens=600,temperature=0.1)
        raw=re.sub(r"```json|```","",vcomp.choices[0].message.content.strip()).strip()
        vision_json=json.loads(raw)
    except Exception as exc:
        logger.error(f"[VISION] {exc}"); vision_json={"description":"Image provided","scene_type":"unknown","user_question_relevance":user_message}
    ai_client,ai_model=get_model_client(model_type)
    memory_block=format_memories_for_prompt(memories)
    cot=f"""{memory_block}\n## Image Analysis:\n{json.dumps(vision_json,indent=2)}\n## User:\n{user_message}\nProvide a thorough response."""
    comp=ai_client.chat.completions.create(model=ai_model,messages=[{"role":"user","content":cot}],max_tokens=1024,temperature=0.7)
    final=comp.choices[0].message.content or ""
    logger.info(f"[VISION] Pipeline complete in {time.time()-t0:.2f}s")
    return final,json.dumps(vision_json)

# ─── Agents ───────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self,name,system_prompt,keywords): self.name=name; self.system_prompt=system_prompt; self.keywords=keywords
    def build_prompt(self,message,context="",memory_block=""):
        parts=[self.system_prompt]
        if memory_block: parts.append(f"\n{memory_block}\n")
        if context: parts.append(f"\nRelevant context:\n{context}\n")
        parts.append(f"\nUser: {message}")
        if context: parts.append("\nAnswer using the context where relevant.")
        return "\n".join(parts)

AGENTS=[
    Agent("coding_agent","You are a senior software engineer. Write clean, efficient, well-documented code. Always include error handling.",
          ["code","python","script","program","function","class","debug","algorithm","api","bug","error","javascript","typescript","sql","refactor","implement","build","develop"]),
    Agent("research_agent","You are an expert researcher. Provide comprehensive, structured explanations.",
          ["research","explain","analyze","investigate","compare","review","report","study","summarize","difference","how does","why does","what is","overview","deep dive"]),
    Agent("vision_agent","You are an expert visual analyst. Analyze images with precision.",
          ["image","photo","picture","screenshot","diagram","chart","see","look","show","visual","analyze image","what is this"]),
    Agent("file_agent","You are a file system expert. Handle file operations safely.",
          ["file","read","write","save","open","create","delete","folder","directory","path","document","upload","download","rename"]),
    Agent("data_agent","You are a data analyst. Interpret data and generate insights.",
          ["data","chart","graph","plot","statistics","average","sum","trend","insight","analysis","metrics","kpi","dashboard"]),
    Agent("general_agent","You are JAZZ, a highly capable AI assistant. Be helpful, clear, and concise.",
          ["hello","hi","help","chat","talk","general"]),
]
DEFAULT_AGENT=AGENTS[-1]

def route_message(message,has_image=False):
    if has_image: return next(a for a in AGENTS if a.name=="vision_agent")
    low=message.lower(); best,best_score=DEFAULT_AGENT,0
    for agent in AGENTS[:-1]:
        score=sum(1 for kw in agent.keywords if kw in low)
        if score>best_score: best,best_score=agent,score
    return best

# ─── NL → SQL ─────────────────────────────────────────────────────────────────
async def run_nl_to_sql(question,user_id):
    t0=time.time(); schema_ctx=get_schema_context(question)
    client,model=get_model_client("censored")
    prompt=f"MySQL schema:\n{schema_ctx}\n\nConvert to safe SELECT only:\n{question}\nReturn ONLY SQL."
    comp=client.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],max_tokens=250,temperature=0.05)
    sql=re.sub(r"```sql|```","",comp.choices[0].message.content.strip()).strip()
    forbidden=["INSERT","UPDATE","DELETE","DROP","ALTER","CREATE","TRUNCATE"]
    if any(kw in sql.upper() for kw in forbidden) or not sql.upper().startswith("SELECT"):
        return MySQLQueryResponse(success=False,question=question,error="Only SELECT queries are allowed")
    if "LIMIT" not in sql.upper(): sql+=" LIMIT 200"
    conn=get_mysql_connection()
    if not conn: return MySQLQueryResponse(success=False,question=question,error="MySQL unavailable")
    try:
        cur=conn.cursor(dictionary=True); cur.execute(sql); rows=cur.fetchall(); cur.close(); conn.close()
        elapsed=int((time.time()-t0)*1000)
        await db_insert("INSERT INTO mysql_query_logs (id,user_id,natural_language,generated_sql,execution_success,row_count,execution_time_ms) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()),user_id,question,sql,1,len(rows),elapsed))
        return MySQLQueryResponse(success=True,question=question,generated_sql=sql,results=rows,row_count=len(rows),execution_time_ms=elapsed)
    except Exception as exc:
        try: conn.close()
        except Exception: pass
        return MySQLQueryResponse(success=False,question=question,generated_sql=sql,error=str(exc))

# ─── File Manager ─────────────────────────────────────────────────────────────
class FileManager:
    def __init__(self,base): self.base=base.resolve(); self.base.mkdir(parents=True,exist_ok=True)
    def _safe_path(self,filename):
        name=os.path.basename(filename)
        if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS: raise ValueError(f"Extension not allowed")
        resolved=(self.base/name).resolve()
        if not str(resolved).startswith(str(self.base)): raise ValueError("Path traversal denied")
        return resolved
    def create(self,fn,content):
        p=self._safe_path(fn)
        if p.exists(): return {"success":False,"message":"File already exists"}
        p.write_text(content,encoding="utf-8"); return {"success":True,"filename":fn}
    def read(self,fn):
        p=self._safe_path(fn)
        if not p.exists(): return {"success":False,"message":"File not found"}
        return {"success":True,"filename":fn,"content":p.read_text(encoding="utf-8")}
    def update(self,fn,content):
        p=self._safe_path(fn)
        if not p.exists(): return {"success":False,"message":"File not found"}
        p.write_text(content,encoding="utf-8"); return {"success":True}
    def delete(self,fn):
        p=self._safe_path(fn)
        if not p.exists(): return {"success":False,"message":"File not found"}
        p.unlink(); return {"success":True}
    def list_files(self): return sorted([{"name":f.name,"size":f.stat().st_size,"modified":datetime.fromtimestamp(f.stat().st_mtime).isoformat()} for f in self.base.iterdir() if f.is_file()],key=lambda x:x["modified"],reverse=True)

file_mgr=FileManager(WORKSPACE_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# ██████╗  AGENT JOB ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class JobLogger:
    def __init__(self): self.lines:List[str]=[]; self.applied:List[Dict]=[]
    def log(self,msg):
        ts=datetime.now().strftime("%H:%M:%S"); entry=f"[{ts}] {msg}"
        self.lines.append(entry); logger.info(f"[JOB] {msg}")
    def add_applied(self,job): self.applied.append(job)
    def output(self): return "\n".join(self.lines)

def _is_within_time_window(start_str:str,end_str:str)->bool:
    try:
        now=datetime.now(); fmt="%H:%M"
        start=datetime.strptime(start_str,fmt).replace(year=now.year,month=now.month,day=now.day)
        end  =datetime.strptime(end_str,  fmt).replace(year=now.year,month=now.month,day=now.day)
        return start<=now<=end
    except Exception: return True

# ─── Naukri.com Agent ─────────────────────────────────────────────────────────
def _run_naukri_agent(creds:Dict,params:Dict,jlog:JobLogger)->List[Dict]:
    """
    Playwright headless browser agent for Naukri.com.

    creds:  { "email": "...", "password": "..." }
    params: {
        "roles":            ["Data Analyst", "Power BI Developer"],
        "location":         "Gurugram",
        "experience_min":   2,
        "experience_max":   5,
        "max_apply":        10,
        "easy_apply_only":  true,
        "keywords_exclude": ["intern", "contract"]
    }
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        jlog.log("❌ Playwright not installed. Run: pip install playwright && playwright install chromium")
        return []

    email      = creds.get("email","")
    password   = creds.get("password","")
    roles      = params.get("roles",["Data Analyst"])
    location   = params.get("location","")
    exp_min    = params.get("experience_min",0)
    exp_max    = params.get("experience_max",10)
    max_apply  = min(params.get("max_apply",10),50)
    easy_only  = params.get("easy_apply_only",True)
    kw_exclude = [k.lower() for k in params.get("keywords_exclude",[])]
    applied:List[Dict]=[]

    jlog.log(f"🚀 Naukri agent | roles={roles} location={location} exp={exp_min}-{exp_max} max={max_apply}")

    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox","--disable-dev-shm-usage"])
        ctx=browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width":1280,"height":800}
        )
        page=ctx.new_page(); page.set_default_timeout(30000)

        # ── LOGIN ──
        try:
            jlog.log("🔐 Logging in to Naukri...")
            page.goto("https://www.naukri.com/nlogin/login",wait_until="domcontentloaded")
            page.wait_for_timeout(2000)
            # Email field
            email_sel="input[placeholder*='Email'], input[type='email'], #usernameField"
            page.wait_for_selector(email_sel,timeout=10000)
            page.fill(email_sel,email)
            page.wait_for_timeout(400)
            # Password field
            pass_sel="input[type='password'], #passwordField"
            page.fill(pass_sel,password)
            page.wait_for_timeout(400)
            # Submit
            page.click("button[type='submit']")
            page.wait_for_timeout(4000)
            if "login" in page.url.lower():
                jlog.log("❌ Login failed — check Naukri credentials")
                browser.close(); return []
            jlog.log("✅ Logged in to Naukri")
        except Exception as exc:
            jlog.log(f"❌ Login error: {exc}"); browser.close(); return []

        total_applied=0
        for role in roles:
            if total_applied>=max_apply: break
            jlog.log(f"\n🔍 Searching: '{role}' | location: '{location}'")

            kw_enc  = role.replace(" ","%20")
            loc_enc = location.replace(" ","%20") if location else ""
            if loc_enc:
                search_url=f"https://www.naukri.com/jobs?k={kw_enc}&l={loc_enc}&experience={exp_min}&to={exp_max}"
            else:
                search_url=f"https://www.naukri.com/jobs?k={kw_enc}&experience={exp_min}&to={exp_max}"

            try:
                page.goto(search_url,wait_until="domcontentloaded"); page.wait_for_timeout(3000)
            except Exception as exc:
                jlog.log(f"⚠ Search page error: {exc}"); continue

            # Job cards — try multiple selectors Naukri uses
            job_cards=page.query_selector_all(
                "article.jobTuple, div.srp-jobtuple-wrapper, div[class*='jobTuple'], "
                "li[class*='jobTupleHeader'], div[class*='cust-job-tuple']"
            )
            jlog.log(f"   Found {len(job_cards)} cards")

            for card in job_cards:
                if total_applied>=max_apply: break
                try:
                    title_el  =card.query_selector("a.title, a[class*='title'], a[class*='jobTitle']")
                    company_el=card.query_selector("a.subTitle, a[class*='company'], span[class*='company'], div[class*='company']")
                    title  =title_el.inner_text().strip()   if title_el   else "Unknown"
                    company=company_el.inner_text().strip() if company_el else "Unknown"
                    job_url=title_el.get_attribute("href")  if title_el   else ""

                    if any(kw in title.lower() for kw in kw_exclude):
                        jlog.log(f"   ⏭ Skipped (excluded kw): {title}"); continue
                    if not job_url: continue

                    jlog.log(f"   📋 {title} @ {company}")
                    job_page=ctx.new_page()
                    try:
                        job_page.goto(job_url,wait_until="domcontentloaded"); job_page.wait_for_timeout(2000)
                        apply_btn=job_page.query_selector(
                            "button:has-text('Apply'), button:has-text('Easy Apply'), "
                            "button:has-text('Apply Now'), a:has-text('Apply')"
                        )
                        if not apply_btn:
                            jlog.log(f"      ⏭ No apply button"); job_page.close(); continue
                        btn_text=apply_btn.inner_text().lower()
                        if easy_only and "easy" not in btn_text and "apply" not in btn_text:
                            jlog.log(f"      ⏭ Not easy apply"); job_page.close(); continue

                        apply_btn.click(); job_page.wait_for_timeout(2500)
                        # Handle modal steps
                        for _ in range(4):
                            step_btn=job_page.query_selector("button:has-text('Apply'),button:has-text('Submit'),button:has-text('Continue'),button:has-text('Next')")
                            if not step_btn: break
                            try: step_btn.click(); job_page.wait_for_timeout(1500)
                            except Exception: break

                        applied.append({"title":title,"company":company,"url":job_url,"applied_at":datetime.utcnow().isoformat(),"status":"applied"})
                        jlog.add_applied(applied[-1]); total_applied+=1
                        jlog.log(f"      ✅ Applied! ({total_applied}/{max_apply})")
                        job_page.wait_for_timeout(800)
                    except Exception as exc:
                        jlog.log(f"      ⚠ Error: {exc}")
                    finally:
                        try: job_page.close()
                        except Exception: pass
                except Exception as exc:
                    jlog.log(f"   ⚠ Card error: {exc}")

        browser.close()
        jlog.log(f"\n🏁 Naukri agent done. Applied to {total_applied} jobs.")
    return applied

# ─── LinkedIn Agent ───────────────────────────────────────────────────────────
def _run_linkedin_agent(creds:Dict,params:Dict,jlog:JobLogger)->List[Dict]:
    """
    LinkedIn Easy Apply agent via Playwright.
    creds:  { "email":"...", "password":"..." }
    params: { "roles":["Data Analyst"], "location":"India", "max_apply":5 }
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        jlog.log("❌ Playwright not installed."); return []

    email=creds.get("email",""); password=creds.get("password","")
    roles=params.get("roles",["Data Analyst"])
    location=params.get("location","India")
    max_apply=min(params.get("max_apply",5),25)
    applied:List[Dict]=[]

    jlog.log(f"🚀 LinkedIn agent | roles={roles} location={location}")

    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,args=["--no-sandbox"])
        ctx=browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36")
        page=ctx.new_page(); page.set_default_timeout(30000)

        try:
            jlog.log("🔐 Logging in to LinkedIn...")
            page.goto("https://www.linkedin.com/login",wait_until="domcontentloaded")
            page.fill("#username",email); page.fill("#password",password)
            page.click("button[type='submit']"); page.wait_for_timeout(4000)
            if "checkpoint" in page.url or "login" in page.url:
                jlog.log("❌ LinkedIn login failed / verification required"); browser.close(); return []
            jlog.log("✅ Logged in to LinkedIn")
        except Exception as exc:
            jlog.log(f"❌ Login error: {exc}"); browser.close(); return []

        total_applied=0
        for role in roles:
            if total_applied>=max_apply: break
            jlog.log(f"\n🔍 Searching LinkedIn: '{role}'")
            try:
                kw_enc=role.replace(" ","%20"); loc_enc=location.replace(" ","%20")
                # f_LF=f_AL is LinkedIn's Easy Apply filter
                url=f"https://www.linkedin.com/jobs/search/?keywords={kw_enc}&location={loc_enc}&f_LF=f_AL"
                page.goto(url,wait_until="domcontentloaded"); page.wait_for_timeout(3000)
                cards=page.query_selector_all("li.jobs-search-results__list-item")
                jlog.log(f"   Found {len(cards)} listings")
                for card in cards:
                    if total_applied>=max_apply: break
                    try:
                        card.click(); page.wait_for_timeout(2000)
                        title_el  =page.query_selector("h1.job-details-jobs-unified-top-card__job-title")
                        company_el=page.query_selector("div.job-details-jobs-unified-top-card__company-name")
                        title  =title_el.inner_text().strip()   if title_el   else "Unknown"
                        company=company_el.inner_text().strip() if company_el else "Unknown"
                        easy_btn=page.query_selector("button:has-text('Easy Apply')")
                        if not easy_btn: jlog.log(f"   ⏭ No Easy Apply: {title}"); continue
                        easy_btn.click(); page.wait_for_timeout(2000)
                        for _ in range(5):
                            next_btn=page.query_selector("button:has-text('Next'),button:has-text('Review'),button:has-text('Submit application')")
                            if not next_btn: break
                            next_btn.click(); page.wait_for_timeout(1500)
                        applied.append({"title":title,"company":company,"applied_at":datetime.utcnow().isoformat(),"status":"applied"})
                        jlog.add_applied(applied[-1]); total_applied+=1
                        jlog.log(f"   ✅ Applied: {title} @ {company} ({total_applied}/{max_apply})")
                        close=page.query_selector("button[aria-label='Dismiss']")
                        if close: close.click()
                        page.wait_for_timeout(1000)
                    except Exception as exc: jlog.log(f"   ⚠ Error: {exc}")
            except Exception as exc: jlog.log(f"⚠ Search error: {exc}")

        browser.close(); jlog.log(f"\n🏁 LinkedIn agent done. Applied to {total_applied} jobs.")
    return applied

# ─── Job Executor ─────────────────────────────────────────────────────────────
def _execute_job_sync(job_id:str)->None:
    import sqlite3 as _s3
    def _db(sql,p=()):
        with _s3.connect(DB_PATH) as c:
            c.row_factory=_s3.Row; return c.execute(sql,p).fetchone()
    def _dbw(sql,p=()):
        with _s3.connect(DB_PATH) as c: c.execute(sql,p); c.commit()

    row=_db("SELECT * FROM agent_jobs WHERE id=?",(job_id,))
    if not row: logger.error(f"[JOB] Job {job_id} not found"); return
    job=dict(row)

    tw_start=job.get("time_window_start") or "00:00"
    tw_end  =job.get("time_window_end")   or "23:59"
    if not _is_within_time_window(tw_start,tw_end):
        logger.info(f"[JOB] '{job['name']}' outside window {tw_start}–{tw_end}, skipping tick"); return

    now_iso=datetime.utcnow().isoformat(); log_id=str(uuid.uuid4())
    _dbw("UPDATE agent_jobs SET status='running',last_run_at=?,updated_at=? WHERE id=?",(now_iso,now_iso,job_id))
    _dbw("INSERT INTO agent_job_logs (id,job_id,user_id,status,started_at) VALUES (?,?,?,?,?)",(log_id,job_id,job["user_id"],"running",now_iso))

    jlog=JobLogger(); jlog.log(f"⚡ Job '{job['name']}' starting (type={job['job_type']})")
    t0=time.time(); applied:List[Dict]=[]; error=None

    try:
        creds =decrypt_creds(job["credentials_enc"]) if job.get("credentials_enc") else {}
        params=json.loads(job.get("parameters") or "{}")
        if job["job_type"]=="naukri_apply":
            applied=_run_naukri_agent(creds,params,jlog)
        elif job["job_type"]=="linkedin_apply":
            applied=_run_linkedin_agent(creds,params,jlog)
        else:
            jlog.log(f"⚠ Unknown job type: {job['job_type']}")
        final_status="completed"
        jlog.log(f"\n✅ Job complete. Applied to {len(applied)} positions.")
    except Exception as exc:
        error=str(exc); final_status="failed"
        jlog.log(f"\n❌ Job failed: {exc}"); logger.error(f"[JOB] '{job['name']}' failed: {exc}")

    elapsed=int(time.time()-t0); done_iso=datetime.utcnow().isoformat()
    _dbw("UPDATE agent_job_logs SET status=?,output=?,applied_jobs=?,error=?,completed_at=?,duration_sec=? WHERE id=?",
        (final_status,jlog.output(),json.dumps(applied,default=str),error,done_iso,elapsed,log_id))
    _dbw("UPDATE agent_jobs SET status='idle',last_run_at=?,last_run_status=?,total_runs=total_runs+1,total_applied=total_applied+?,updated_at=? WHERE id=?",
        (done_iso,final_status,len(applied),done_iso,job_id))
    logger.info(f"[JOB] '{job['name']}' {final_status} in {elapsed}s — applied: {len(applied)}")

async def execute_job(job_id:str)->None:
    loop=asyncio.get_event_loop()
    await loop.run_in_executor(_executor,_execute_job_sync,job_id)

# ─── Scheduler ────────────────────────────────────────────────────────────────
scheduler=BackgroundScheduler(daemon=True)

def _schedule_job(job:Dict)->None:
    sid=f"agent_{job['id']}"
    try:
        parts=job["cron_schedule"].split()
        if len(parts)!=5: raise ValueError(f"Invalid cron: {job['cron_schedule']}")
        minute,hour,day,month,dow=parts
        trigger=CronTrigger(minute=minute,hour=hour,day=day,month=month,day_of_week=dow)
        def _wrapper(jid=job["id"]):
            import asyncio as _aio
            loop=_aio.new_event_loop()
            try: loop.run_until_complete(execute_job(jid))
            finally: loop.close()
        if scheduler.get_job(sid): scheduler.remove_job(sid)
        scheduler.add_job(_wrapper,trigger=trigger,id=sid,replace_existing=True)
        logger.info(f"[SCHEDULER] Registered '{job['name']}' ({job['cron_schedule']})")
    except Exception as exc:
        logger.error(f"[SCHEDULER] Failed to register {job['id']}: {exc}")

def _unschedule_job(job_id:str)->None:
    sid=f"agent_{job_id}"
    if scheduler.get_job(sid): scheduler.remove_job(sid); logger.info(f"[SCHEDULER] Removed {job_id}")

def _load_all_jobs_into_scheduler()->None:
    import sqlite3 as _s3
    try:
        with _s3.connect(DB_PATH) as conn:
            conn.row_factory=_s3.Row
            rows=conn.execute("SELECT * FROM agent_jobs WHERE enabled=1").fetchall()
        for row in rows: _schedule_job(dict(row))
        logger.info(f"[SCHEDULER] Loaded {len(rows)} agent jobs ✅")
    except Exception as exc:
        logger.error(f"[SCHEDULER] Failed to load jobs: {exc}")

# ─── App ──────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app:FastAPI):
    logger.info("="*60)
    logger.info("[STARTUP] JAZZ AI v5.1 starting (Agent Jobs Edition)...")
    logger.info(f"  CHROMADB: {'✅' if chroma_client else '❌'}")
    logger.info(f"  GROQ:     {'✅' if GROQ_API_KEY else '❌'}")
    logger.info(f"  HF_TOKEN: {'✅' if HF_TOKEN else '❌'}")
    logger.info(f"  DB:       {DB_PATH}")
    logger.info(f"  ADMIN:    {ADMIN_EMAIL}")
    logger.info("="*60)
    await init_db(); index_sqlite_schema()
    if MYSQL_ENABLED: _init_mysql_pool(); index_mysql_schema()
    scheduler.add_job(index_sqlite_schema,trigger=IntervalTrigger(hours=1),id="index_sqlite_schema",replace_existing=True)
    if MYSQL_ENABLED:
        scheduler.add_job(index_mysql_schema,trigger=IntervalTrigger(minutes=30),id="mysql_reindex",replace_existing=True)
    scheduler.start()
    _load_all_jobs_into_scheduler()
    logger.info("[STARTUP] All services ready ✅")
    yield
    logger.info("[SHUTDOWN] Stopping...")
    scheduler.shutdown(wait=False); _executor.shutdown(wait=False)

app=FastAPI(title="JAZZ AI",version="5.1.0",description="Production AI platform with Agent Jobs",lifespan=lifespan,docs_url="/api/docs",redoc_url=None)
app.add_middleware(CORSMiddleware,allow_origins=ALLOWED_ORIGINS,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

@app.middleware("http")
async def add_request_id(request:Request,call_next):
    rid=str(uuid.uuid4())[:8]; request.state.request_id=rid
    response=await call_next(request)
    response.headers["X-Request-ID"]=rid; response.headers["X-Powered-By"]="JAZZ-AI-v5.1"
    return response

_static_dir=Path("static")
if _static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/static",StaticFiles(directory="static"),name="static")
    @app.get("/")
    async def root(): return FileResponse("static/index.html")
else:
    @app.get("/")
    async def root(): return {"status":"JAZZ AI API","version":"5.1.0","docs":"/api/docs"}

# ═══════════════ AUTH ═══════════════════════════════════════════════════════
@app.post("/auth/signup",response_model=TokenResponse)
async def signup(req:UserSignup):
    existing=await db_fetchone("SELECT id FROM users WHERE email=?",(req.email,))
    if existing: raise HTTPException(400,"Email already registered")
    uid=str(uuid.uuid4()); phash=pwd_context.hash(req.password)
    role="admin" if req.email==ADMIN_EMAIL else "client"
    sub ="enterprise" if req.email==ADMIN_EMAIL else "free"
    await db_execute("INSERT INTO users (id,email,password_hash,full_name,role,subscription) VALUES (?,?,?,?,?,?)",(uid,req.email,phash,req.full_name or "",role,sub))
    token=create_access_token({"sub":uid,"email":req.email,"role":role})
    return TokenResponse(access_token=token,user={"id":uid,"email":req.email,"role":role,"subscription":sub,"full_name":req.full_name or ""})

@app.post("/auth/login",response_model=TokenResponse)
async def login(req:UserLogin):
    if req.email==ADMIN_EMAIL and ADMIN_PASSWORD and req.password==ADMIN_PASSWORD:
        user=await db_fetchone("SELECT * FROM users WHERE email=?",(ADMIN_EMAIL,))
        if not user:
            uid=str(uuid.uuid4()); phash=pwd_context.hash(ADMIN_PASSWORD)
            await db_execute("INSERT INTO users (id,email,password_hash,full_name,role,subscription) VALUES (?,?,?,?,?,?)",(uid,ADMIN_EMAIL,phash,"Admin","admin","enterprise"))
            user={"id":uid,"email":ADMIN_EMAIL,"role":"admin","subscription":"enterprise","full_name":"Admin","memory_enabled":1}
        token=create_access_token({"sub":user["id"],"email":req.email,"role":"admin"})
        return TokenResponse(access_token=token,user={"id":user["id"],"email":req.email,"role":"admin","subscription":"enterprise","full_name":"Admin","memory_enabled":True})
    user=await db_fetchone("SELECT * FROM users WHERE email=?",(req.email,))
    if not user: raise HTTPException(401,"Invalid credentials")
    try: ok=pwd_context.verify(req.password,user["password_hash"])
    except Exception: ok=False
    if not ok: raise HTTPException(401,"Invalid credentials")
    token=create_access_token({"sub":user["id"],"email":user["email"],"role":user["role"]})
    return TokenResponse(access_token=token,user={"id":user["id"],"email":user["email"],"role":user["role"],"subscription":user["subscription"],"full_name":user.get("full_name",""),"memory_enabled":bool(user.get("memory_enabled",1))})

@app.post("/auth/admin-login",response_model=TokenResponse)
async def admin_login(req:UserLogin):
    if req.email!=ADMIN_EMAIL: raise HTTPException(403,"Admin access only")
    return await login(req)

@app.get("/auth/me")
async def me(user:Dict=Depends(get_current_user)):
    row=await db_fetchone("SELECT id,email,full_name,role,subscription,memory_enabled,avatar_url,preferred_model,created_at FROM users WHERE id=?",(user["sub"],))
    if not row: raise HTTPException(404,"User not found")
    row["memory_enabled"]=bool(row.get("memory_enabled",1)); return row

@app.post("/auth/logout")
async def logout(user:Dict=Depends(get_current_user)): return {"message":"Logged out"}

# ═══════════════ PROFILE ════════════════════════════════════════════════════
@app.get("/profile")
async def get_profile(user:Dict=Depends(get_current_user)):
    row=await db_fetchone("SELECT id,email,full_name,role,subscription,memory_enabled,avatar_url,preferred_model,created_at FROM users WHERE id=?",(user["sub"],))
    if not row: raise HTTPException(404,"Profile not found")
    row["memory_enabled"]=bool(row.get("memory_enabled",1)); return row

@app.put("/profile")
async def update_profile(req:ProfileUpdate,user:Dict=Depends(get_current_user)):
    data=req.model_dump(exclude_none=True)
    if not data: raise HTTPException(400,"Nothing to update")
    sets=", ".join(f"{k}=?" for k in data)
    await db_execute(f"UPDATE users SET {sets}, updated_at=? WHERE id=?",tuple(data.values())+(datetime.utcnow().isoformat(),user["sub"]))
    return {"success":True}

@app.get("/user/stats")
async def user_stats(user:Dict=Depends(get_current_user)):
    uid=user["sub"]; sub_tier=await get_user_subscription(uid)
    limits=SUBSCRIPTION_LIMITS.get(sub_tier,SUBSCRIPTION_LIMITS["free"])
    return {
        "subscription":sub_tier,"messages_used_today":rate_limiter.usage_today(uid),
        "messages_limit":limits["messages_per_day"],
        "documents":await db_count("SELECT COUNT(*) FROM documents WHERE user_id=?",(uid,)),
        "document_limit":limits["documents"],"mysql_access":limits["mysql_access"],
        "image_analysis":limits["image_analysis"],
        "memory_count":await db_count("SELECT COUNT(*) FROM user_memories WHERE user_id=?",(uid,)),
        "agent_jobs":await db_count("SELECT COUNT(*) FROM agent_jobs WHERE user_id=?",(uid,)),
        "agent_jobs_limit":limits.get("agent_jobs",1),
    }

# ═══════════════ MEMORY ═════════════════════════════════════════════════════
@app.get("/memory")
async def list_memories(user:Dict=Depends(get_current_user)):
    uid=user["sub"]; memories=await get_user_memories(uid); mem_enabled=await is_memory_enabled(uid)
    return {"memories":memories,"count":len(memories),"memory_enabled":mem_enabled,"max_memories":MAX_MEMORIES}

@app.post("/memory")
async def create_memory(req:MemoryCreate,user:Dict=Depends(get_current_user)):
    uid=user["sub"]; count=await db_count("SELECT COUNT(*) FROM user_memories WHERE user_id=?",(uid,))
    if count>=MAX_MEMORIES: raise HTTPException(400,f"Memory limit reached ({MAX_MEMORIES})")
    mid=str(uuid.uuid4())
    await db_execute("INSERT INTO user_memories (id,user_id,content,category,source) VALUES (?,?,?,?,?)",(mid,uid,req.content.strip(),req.category or "general","manual"))
    return {"success":True,"memory":await db_fetchone("SELECT * FROM user_memories WHERE id=?",(mid,))}

@app.put("/memory/{memory_id}")
async def update_memory(memory_id:str,req:MemoryUpdate,user:Dict=Depends(get_current_user)):
    row=await db_fetchone("SELECT id FROM user_memories WHERE id=? AND user_id=?",(memory_id,user["sub"]))
    if not row: raise HTTPException(404,"Memory not found")
    await db_execute("UPDATE user_memories SET content=?,updated_at=? WHERE id=? AND user_id=?",(req.content.strip(),datetime.utcnow().isoformat(),memory_id,user["sub"]))
    return {"success":True,"memory":await db_fetchone("SELECT * FROM user_memories WHERE id=?",(memory_id,))}

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id:str,user:Dict=Depends(get_current_user)):
    await db_execute("DELETE FROM user_memories WHERE id=? AND user_id=?",(memory_id,user["sub"])); return {"success":True}

@app.delete("/memory")
async def clear_all_memories(user:Dict=Depends(get_current_user)):
    await db_execute("DELETE FROM user_memories WHERE user_id=?",(user["sub"],)); return {"success":True,"message":"All memories cleared"}

@app.put("/memory/settings/toggle")
async def toggle_memory(req:MemorySettingsUpdate,user:Dict=Depends(get_current_user)):
    await db_execute("UPDATE users SET memory_enabled=?,updated_at=? WHERE id=?",(1 if req.enabled else 0,datetime.utcnow().isoformat(),user["sub"]))
    return {"success":True,"memory_enabled":req.enabled}

# ═══════════════ IMAGE ══════════════════════════════════════════════════════
@app.post("/images/upload")
async def upload_image(file:UploadFile=File(...),user:Dict=Depends(get_current_user)):
    if file.content_type not in ALLOWED_IMAGE_TYPES: raise HTTPException(400,f"Unsupported image type: {file.content_type}")
    content=await file.read()
    if len(content)>MAX_IMAGE_SIZE: raise HTTPException(413,"Image too large")
    b64=base64.b64encode(content).decode("utf-8")
    return {"success":True,"data_url":f"data:{file.content_type};base64,{b64}","filename":file.filename,"size":len(content),"mime_type":file.content_type}

# ═══════════════ CHAT (non-streaming) ════════════════════════════════════════
@app.post("/chat",response_model=ChatResponse)
async def chat(req:ChatRequest,user:Dict=Depends(get_current_user)):
    t0=time.time(); uid=user["sub"]; tier=await get_user_subscription(uid)
    if not rate_limiter.check(uid,tier): raise HTTPException(429,"Daily limit reached. Upgrade your plan.")
    mem_on=await is_memory_enabled(uid); memories=await get_user_memories(uid) if mem_on else []
    memory_block=format_memories_for_prompt(memories); has_image=bool(req.image_url)
    if has_image:
        try:
            text,_=await analyze_image_pipeline(req.image_url,req.message,req.model_type,uid,memories)
            new_memories=await extract_and_save_memories(uid,req.message,text)
            await db_insert("INSERT INTO chat_history (id,user_id,message,response,query_type,agent_used,model_used,response_time_ms) VALUES (?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()),uid,req.message,text,"image","vision_agent",req.model_type,int((time.time()-t0)*1000)))
            return ChatResponse(response=text,response_time=time.time()-t0,agent_used="vision_agent",memory_updated=bool(new_memories),new_memories=new_memories)
        except Exception as exc: raise HTTPException(500,f"Image analysis failed: {exc}")
    if is_db_intent(req.message) and MYSQL_ENABLED and SUBSCRIPTION_LIMITS.get(tier,{}).get("mysql_access"):
        result=await run_nl_to_sql(req.message,uid)
        if result.success:
            text=f"**SQL:**\n```sql\n{result.generated_sql}\n```\n\n**Results** ({result.row_count} rows):\n"+json.dumps(result.results[:20],indent=2,default=str)
            return ChatResponse(response=text,response_time=time.time()-t0,agent_used="data_agent")
    agent=route_message(req.message,has_image); context=get_relevant_context(uid,req.message) if req.use_rag else ""
    prompt=agent.build_prompt(req.message,context,memory_block); ai,model=get_model_client(req.model_type)
    completion=ai.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],max_tokens=1024,temperature=0.7)
    text=completion.choices[0].message.content or ""; new_memories=await extract_and_save_memories(uid,req.message,text)
    await db_insert("INSERT INTO chat_history (id,user_id,message,response,query_type,agent_used,model_used,response_time_ms) VALUES (?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()),uid,req.message,text,"rag" if context else "chat",agent.name,req.model_type,int((time.time()-t0)*1000)))
    return ChatResponse(response=text,response_time=time.time()-t0,agent_used=agent.name,memory_updated=bool(new_memories),new_memories=new_memories)

# ═══════════════ CHAT (streaming) ════════════════════════════════════════════
@app.post("/chat-stream")
async def chat_stream(req:ChatRequest,user:Dict=Depends(get_current_user)):
    t0=time.time(); uid=user["sub"]; tier=await get_user_subscription(uid)
    if not rate_limiter.check(uid,tier): raise HTTPException(429,"Daily limit reached.")
    mem_on=await is_memory_enabled(uid); memories=await get_user_memories(uid) if mem_on else []
    memory_block=format_memories_for_prompt(memories); has_image=bool(req.image_url)
    if has_image:
        async def _image_stream():
            yield f"data: {json.dumps({'type':'agent','agent':'vision_agent'})}\n\n"
            yield f"data: {json.dumps({'type':'status','message':'🔍 Analyzing image...'})}\n\n"
            try:
                text,_=await analyze_image_pipeline(req.image_url,req.message,req.model_type,uid,memories)
                words=text.split()
                for i in range(0,len(words),8):
                    chunk=" ".join(words[i:i+8])+(" " if i+8<len(words) else "")
                    yield f"data: {json.dumps({'type':'chunk','content':chunk})}\n\n"; await asyncio.sleep(0.01)
                new_memories=await extract_and_save_memories(uid,req.message,text)
                await db_insert("INSERT INTO chat_history (id,user_id,message,response,query_type,agent_used,model_used,response_time_ms) VALUES (?,?,?,?,?,?,?,?)",
                    (str(uuid.uuid4()),uid,req.message,text,"image","vision_agent",req.model_type,int((time.time()-t0)*1000)))
                yield f"data: {json.dumps({'type':'done','response_time':time.time()-t0,'agent':'vision_agent','full_response':text,'memory_updated':bool(new_memories),'new_memories':new_memories})}\n\n"
            except Exception as exc: yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"
        return StreamingResponse(_image_stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})
    agent=route_message(req.message,has_image); context=get_relevant_context(uid,req.message) if req.use_rag else ""
    prompt=agent.build_prompt(req.message,context,memory_block); ai,model=get_model_client(req.model_type)
    try: stream=ai.chat.completions.create(model=model,messages=[{"role":"user","content":prompt}],max_tokens=1024,temperature=0.7,stream=True)
    except Exception as exc: raise HTTPException(503,f"AI service error: {exc}")
    async def _generate():
        full=""
        try:
            yield f"data: {json.dumps({'type':'agent','agent':agent.name})}\n\n"
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content=chunk.choices[0].delta.content; full+=content
                    yield f"data: {json.dumps({'type':'chunk','content':content})}\n\n"
            elapsed=time.time()-t0; new_memories=await extract_and_save_memories(uid,req.message,full)
            await db_insert("INSERT INTO chat_history (id,user_id,message,response,query_type,agent_used,model_used,response_time_ms) VALUES (?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()),uid,req.message,full,"rag" if context else "chat",agent.name,req.model_type,int(elapsed*1000)))
            yield f"data: {json.dumps({'type':'done','response_time':elapsed,'agent':agent.name,'full_response':full,'memory_updated':bool(new_memories),'new_memories':new_memories})}\n\n"
        except Exception as exc: yield f"data: {json.dumps({'type':'error','message':str(exc)})}\n\n"
    return StreamingResponse(_generate(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.get("/chat/history")
async def chat_history(page:int=1,per_page:int=20,user:Dict=Depends(get_current_user)):
    uid=user["sub"]; offset=(page-1)*per_page
    rows=await db_fetchall("SELECT * FROM chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",(uid,per_page,offset))
    return {"history":rows,"page":page,"per_page":per_page}

# ═══════════════ DOCUMENTS ═══════════════════════════════════════════════════
@app.post("/documents/upload")
async def upload_document(file:UploadFile=File(...),user:Dict=Depends(get_current_user)):
    if not documents_collection: raise HTTPException(503,"Vector DB unavailable")
    uid=user["sub"]; tier=await get_user_subscription(uid)
    limits=SUBSCRIPTION_LIMITS.get(tier,SUBSCRIPTION_LIMITS["free"])
    if limits["documents"]!=-1:
        count=await db_count("SELECT COUNT(*) FROM documents WHERE user_id=?",(uid,))
        if count>=limits["documents"]: raise HTTPException(403,f"Document limit reached for {tier} plan")
    content=await file.read()
    if len(content)>limits["max_file_mb"]*1024*1024: raise HTTPException(413,"File too large")
    tmp=tempfile.mkdtemp()
    try:
        fpath=os.path.join(tmp,file.filename or "upload")
        with open(fpath,"wb") as f: f.write(content)
        mime=file.content_type or "application/octet-stream"; text=process_document(fpath,mime)
        if not text.strip(): raise HTTPException(400,"Could not extract text from document")
        chunks=chunk_text(text); doc_id=str(uuid.uuid4())
        documents_collection.add(documents=chunks,ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
            metadatas=[{"user_id":uid,"filename":file.filename,"doc_id":doc_id} for _ in chunks])
        now=datetime.utcnow().isoformat()
        await db_execute("INSERT INTO documents (id,user_id,filename,original_name,file_type,file_size,chunk_count,chroma_collection,uploaded_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id,uid,file.filename or "",file.filename or "",mime,len(content),len(chunks),"user_documents",now))
        return DocumentResponse(id=doc_id,filename=file.filename or "",original_name=file.filename or "",file_type=mime,file_size=len(content),chunk_count=len(chunks),uploaded_at=now)
    finally: shutil.rmtree(tmp,ignore_errors=True)

@app.get("/documents")
async def list_documents(user:Dict=Depends(get_current_user)):
    return {"documents":await db_fetchall("SELECT * FROM documents WHERE user_id=? ORDER BY uploaded_at DESC",(user["sub"],))}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id:str,user:Dict=Depends(get_current_user)):
    uid=user["sub"]
    if documents_collection:
        try:
            existing=documents_collection.get(where={"doc_id":doc_id,"user_id":uid})
            if existing["ids"]: documents_collection.delete(ids=existing["ids"])
        except Exception as exc: logger.warning(f"[CHROMA] Delete: {exc}")
    await db_execute("DELETE FROM documents WHERE id=? AND user_id=?",(doc_id,uid)); return {"success":True}

# ═══════════════ AGENT JOBS  ⚡ ═══════════════════════════════════════════════
@app.post("/agent-jobs")
async def create_agent_job(req:AgentJobCreate,user:Dict=Depends(get_current_user)):
    uid=user["sub"]; tier=await get_user_subscription(uid)
    limits=SUBSCRIPTION_LIMITS.get(tier,SUBSCRIPTION_LIMITS["free"])
    job_limit=limits.get("agent_jobs",1)
    if job_limit!=-1:
        count=await db_count("SELECT COUNT(*) FROM agent_jobs WHERE user_id=?",(uid,))
        if count>=job_limit: raise HTTPException(403,f"Agent job limit ({job_limit}) reached for {tier} plan. Upgrade to Pro/Enterprise.")
    if len(req.cron_schedule.strip().split())!=5:
        raise HTTPException(400,"cron_schedule must be 5 fields: minute hour day month weekday  e.g. '0 11 * * *'")
    job_id=str(uuid.uuid4()); now_iso=datetime.utcnow().isoformat()
    creds_enc=encrypt_creds(req.credentials); params_js=json.dumps(req.parameters or {})
    await db_execute(
        "INSERT INTO agent_jobs (id,user_id,name,description,job_type,cron_schedule,time_window_start,time_window_end,credentials_enc,parameters,status,enabled,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (job_id,uid,req.name,req.description or "",req.job_type,req.cron_schedule,
         req.time_window_start or "11:00",req.time_window_end or "12:00",
         creds_enc,params_js,"idle",1,now_iso,now_iso),
    )
    job=await db_fetchone("SELECT * FROM agent_jobs WHERE id=?",(job_id,))
    job.pop("credentials_enc",None)
    _schedule_job({**job,"credentials_enc":creds_enc})
    return {"success":True,"job":job}

@app.get("/agent-jobs")
async def list_agent_jobs(user:Dict=Depends(get_current_user)):
    rows=await db_fetchall(
        "SELECT id,user_id,name,description,job_type,cron_schedule,time_window_start,time_window_end,parameters,status,enabled,last_run_at,last_run_status,next_run_at,total_runs,total_applied,created_at,updated_at FROM agent_jobs WHERE user_id=? ORDER BY created_at DESC",
        (user["sub"],))
    return {"jobs":rows}

@app.get("/agent-jobs/{job_id}")
async def get_agent_job(job_id:str,user:Dict=Depends(get_current_user)):
    row=await db_fetchone(
        "SELECT id,user_id,name,description,job_type,cron_schedule,time_window_start,time_window_end,parameters,status,enabled,last_run_at,last_run_status,next_run_at,total_runs,total_applied,created_at,updated_at FROM agent_jobs WHERE id=? AND user_id=?",
        (job_id,user["sub"]))
    if not row: raise HTTPException(404,"Job not found"); return row

@app.put("/agent-jobs/{job_id}")
async def update_agent_job(job_id:str,req:AgentJobUpdate,user:Dict=Depends(get_current_user)):
    existing=await db_fetchone("SELECT * FROM agent_jobs WHERE id=? AND user_id=?",(job_id,user["sub"]))
    if not existing: raise HTTPException(404,"Job not found")
    updates:Dict[str,Any]={}
    if req.name              is not None: updates["name"]             =req.name
    if req.description       is not None: updates["description"]      =req.description
    if req.cron_schedule     is not None:
        if len(req.cron_schedule.split())!=5: raise HTTPException(400,"cron_schedule must be 5 fields")
        updates["cron_schedule"]=req.cron_schedule
    if req.time_window_start is not None: updates["time_window_start"]=req.time_window_start
    if req.time_window_end   is not None: updates["time_window_end"]  =req.time_window_end
    if req.parameters        is not None: updates["parameters"]       =json.dumps(req.parameters)
    if req.enabled           is not None: updates["enabled"]          =1 if req.enabled else 0
    if req.credentials       is not None: updates["credentials_enc"]  =encrypt_creds(req.credentials)
    if not updates: raise HTTPException(400,"Nothing to update")
    updates["updated_at"]=datetime.utcnow().isoformat()
    sets=", ".join(f"{k}=?" for k in updates)
    await db_execute(f"UPDATE agent_jobs SET {sets} WHERE id=?",tuple(updates.values())+(job_id,))
    updated=await db_fetchone("SELECT * FROM agent_jobs WHERE id=?",(job_id,))
    if updated.get("enabled"): _schedule_job(updated)
    else: _unschedule_job(job_id)
    updated.pop("credentials_enc",None)
    return {"success":True,"job":updated}

@app.delete("/agent-jobs/{job_id}")
async def delete_agent_job(job_id:str,user:Dict=Depends(get_current_user)):
    existing=await db_fetchone("SELECT id FROM agent_jobs WHERE id=? AND user_id=?",(job_id,user["sub"]))
    if not existing: raise HTTPException(404,"Job not found")
    _unschedule_job(job_id); await db_execute("DELETE FROM agent_jobs WHERE id=?",(job_id,))
    return {"success":True}

@app.post("/agent-jobs/{job_id}/run")
async def run_agent_job_now(job_id:str,user:Dict=Depends(get_current_user)):
    """Trigger a job immediately, ignoring time window."""
    existing=await db_fetchone("SELECT id FROM agent_jobs WHERE id=? AND user_id=?",(job_id,user["sub"]))
    if not existing: raise HTTPException(404,"Job not found")
    asyncio.create_task(execute_job(job_id))
    return {"success":True,"message":"Job triggered — check /agent-jobs/{id}/logs for progress"}

@app.post("/agent-jobs/{job_id}/toggle")
async def toggle_agent_job(job_id:str,user:Dict=Depends(get_current_user)):
    row=await db_fetchone("SELECT * FROM agent_jobs WHERE id=? AND user_id=?",(job_id,user["sub"]))
    if not row: raise HTTPException(404,"Job not found")
    new_enabled=0 if row["enabled"] else 1
    await db_execute("UPDATE agent_jobs SET enabled=?,updated_at=? WHERE id=?",(new_enabled,datetime.utcnow().isoformat(),job_id))
    if new_enabled: _schedule_job({**row,"enabled":new_enabled})
    else: _unschedule_job(job_id)
    return {"success":True,"enabled":bool(new_enabled)}

@app.get("/agent-jobs/{job_id}/logs")
async def get_job_logs(job_id:str,limit:int=20,user:Dict=Depends(get_current_user)):
    existing=await db_fetchone("SELECT id FROM agent_jobs WHERE id=? AND user_id=?",(job_id,user["sub"]))
    if not existing: raise HTTPException(404,"Job not found")
    rows=await db_fetchall("SELECT id,job_id,status,output,applied_jobs,error,started_at,completed_at,duration_sec FROM agent_job_logs WHERE job_id=? ORDER BY started_at DESC LIMIT ?",(job_id,limit))
    for r in rows:
        try: r["applied_jobs"]=json.loads(r.get("applied_jobs") or "[]")
        except Exception: r["applied_jobs"]=[]
    return {"logs":rows}

@app.get("/agent-jobs/{job_id}/logs/{log_id}")
async def get_job_log_detail(job_id:str,log_id:str,user:Dict=Depends(get_current_user)):
    existing=await db_fetchone("SELECT id FROM agent_jobs WHERE id=? AND user_id=?",(job_id,user["sub"]))
    if not existing: raise HTTPException(404,"Job not found")
    row=await db_fetchone("SELECT * FROM agent_job_logs WHERE id=? AND job_id=?",(log_id,job_id))
    if not row: raise HTTPException(404,"Log not found")
    try: row["applied_jobs"]=json.loads(row.get("applied_jobs") or "[]")
    except Exception: row["applied_jobs"]=[]
    return row

@app.get("/admin/agent-jobs")
async def admin_all_jobs(user:Dict=Depends(require_admin)):
    rows=await db_fetchall("SELECT j.id,j.name,j.job_type,j.cron_schedule,j.status,j.enabled,j.last_run_at,j.last_run_status,j.total_runs,j.total_applied,u.email FROM agent_jobs j LEFT JOIN users u ON j.user_id=u.id ORDER BY j.created_at DESC")
    return {"jobs":rows}

# ═══════════════ FILE WORKSPACE ══════════════════════════════════════════════
@app.post("/files")
async def file_operation(req:FileOperationRequest):
    ops={"create":lambda:file_mgr.create(req.filename,req.content or ""),"read":lambda:file_mgr.read(req.filename),
         "update":lambda:file_mgr.update(req.filename,req.content or ""),"delete":lambda:file_mgr.delete(req.filename),
         "list":lambda:{"success":True,"files":file_mgr.list_files()}}
    fn=ops.get(req.operation)
    if not fn: raise HTTPException(400,f"Unknown operation: {req.operation}")
    return fn()

# ═══════════════ ADMIN ═══════════════════════════════════════════════════════
@app.get("/admin/stats")
async def admin_stats(user:Dict=Depends(require_admin)):
    today=datetime.utcnow().strftime("%Y-%m-%d")
    return {
        "total_users":     await db_count("SELECT COUNT(*) FROM users"),
        "total_documents": await db_count("SELECT COUNT(*) FROM documents"),
        "chats_today":     await db_count("SELECT COUNT(*) FROM chat_history WHERE created_at >= ?",(today,)),
        "total_memories":  await db_count("SELECT COUNT(*) FROM user_memories"),
        "total_agent_jobs":await db_count("SELECT COUNT(*) FROM agent_jobs"),
        "agent_runs_today":await db_count("SELECT COUNT(*) FROM agent_job_logs WHERE started_at >= ?",(today,)),
        "mysql_enabled":MYSQL_ENABLED,"last_schema_reindex":_last_reindex_at,
        "system_status":"healthy","version":"5.1.0","uptime_since":_SERVER_START.isoformat(),
    }

@app.get("/admin/users")
async def admin_users(user:Dict=Depends(require_admin)):
    rows=await db_fetchall("SELECT id,email,full_name,role,subscription,memory_enabled,created_at FROM users ORDER BY created_at DESC")
    for r in rows: r["memory_enabled"]=bool(r.get("memory_enabled",1))
    return {"users":rows}

@app.put("/admin/subscription")
async def admin_update_subscription(req:SubscriptionUpdate,user:Dict=Depends(require_admin)):
    await db_execute("UPDATE users SET subscription=?,updated_at=? WHERE id=?",(req.tier,datetime.utcnow().isoformat(),req.user_id))
    return {"success":True,"user_id":req.user_id,"subscription":req.tier}

@app.put("/admin/user-role")
async def admin_update_role(user_id:str,role:str,user:Dict=Depends(require_admin)):
    if role not in ("admin","client"): raise HTTPException(400,"Role must be 'admin' or 'client'")
    await db_execute("UPDATE users SET role=?,updated_at=? WHERE id=?",(role,datetime.utcnow().isoformat(),user_id))
    return {"success":True}

@app.delete("/admin/user/{user_id}")
async def admin_delete_user(user_id:str,user:Dict=Depends(require_admin)):
    await db_execute("DELETE FROM users WHERE id=?",(user_id,)); return {"success":True}

@app.get("/admin/documents")
async def admin_documents(user:Dict=Depends(require_admin)):
    return {"documents":await db_fetchall("SELECT d.*,u.email FROM documents d LEFT JOIN users u ON d.user_id=u.id ORDER BY d.uploaded_at DESC")}

@app.get("/admin/chat-logs")
async def admin_chat_logs(limit:int=100,user:Dict=Depends(require_admin)):
    return {"logs":await db_fetchall("SELECT h.*,u.email FROM chat_history h LEFT JOIN users u ON h.user_id=u.id ORDER BY h.created_at DESC LIMIT ?",(limit,))}

@app.get("/admin/memories")
async def admin_memories(user:Dict=Depends(require_admin)):
    return {"memories":await db_fetchall("SELECT m.*,u.email FROM user_memories m LEFT JOIN users u ON m.user_id=u.id ORDER BY m.created_at DESC LIMIT 200")}

@app.get("/logs")
async def get_logs(user:Dict=Depends(require_admin)):
    try: return {"logs":Path("jazz.log").read_text().splitlines()[-200:]}
    except FileNotFoundError: return {"logs":[]}

# ═══════════════ HEALTH ══════════════════════════════════════════════════════
@app.get("/health")
async def health():
    db_ok=False
    try: await db_fetchone("SELECT 1"); db_ok=True
    except Exception: pass
    return {
        "status":"healthy" if db_ok else "degraded","timestamp":datetime.utcnow().isoformat(),
        "uptime_seconds":int((datetime.utcnow()-_SERVER_START).total_seconds()),
        "services":{"sqlite":db_ok,"chromadb":chroma_client is not None,"mysql":_mysql_pool is not None},
        "agents":[a.name for a in AGENTS],"version":"5.1.0",
        "features":["memory","vision","rag","streaming","agent_jobs"],
    }

@app.get("/model-info")
async def model_info():
    return {"models":[
        {"id":"censored",  "name":"Groq Llama 3.3 70B",        "speed":"ultra-fast","filtered":True},
        {"id":"uncensored","name":"Dolphin Mistral 24B Venice","speed":"variable",  "filtered":False},
    ],"vision_model":"Qwen2.5-VL-7B-Instruct"}

# ═══════════════ ENTRY POINT ═════════════════════════════════════════════════
if __name__=="__main__":
    import uvicorn
    uvicorn.run("server:app",host="0.0.0.0",port=int(os.getenv("PORT","8000")),
                reload=False,workers=1,log_level="info",access_log=True,timeout_keep_alive=75)
