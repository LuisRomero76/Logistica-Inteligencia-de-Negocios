"""
main.py
───────
Punto de entrada de LogiBrain API.

Responsabilidades:
  • Inicializa la aplicación FastAPI con metadatos OpenAPI.
  • Crea las tablas del Data Warehouse en PostgreSQL al arrancar
    (idempotente: CREATE TABLE IF NOT EXISTS vía SQLAlchemy).
  • Registra los dos APIRouter:
      - /predict  → Motor predictivo ML  (routes_ml.py)
      - /chat     → Agente Text-to-SQL   (routes_chat.py)
  • Configura CORS permisivo para desarrollo (restringir en producción).
  • Expone /health para health-checks del orquestador (Docker, K8s, etc.)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.config import engine

# Importamos los modelos para que SQLAlchemy los registre en Base.metadata
# antes de llamar a create_all(). Sin esta importación las tablas no se crean.
import database.models as _models  # noqa: F401

from database.config import Base
from api.routes_ml import router as ml_router
from api.routes_chat import router as chat_router


# ─────────────────────────────────────────────────────────────────
#  Lifespan: setup y teardown de la aplicación
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup → crea las tablas si no existen.
    Shutdown → no se requiere limpieza adicional (SQLAlchemy cierra el pool).
    """
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas del Data Warehouse verificadas / creadas correctamente.")
    yield
    print("👋 LogiBrain API apagándose.")


# ─────────────────────────────────────────────────────────────────
#  Aplicación FastAPI
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LogiBrain API",
    description=(
        "Backend de analítica logística corporativa.\n\n"
        "Integra **Data Engineering** (Star Schema en PostgreSQL), "
        "**Machine Learning** (RandomForest para predicción de retrasos) "
        "e **IA Generativa** (Agente Text-to-SQL con LangChain).\n\n"
        "Diseñado para ser consumido por un frontend Vue/React."
    ),
    version="1.0.0",
    contact={
        "name": "Equipo LogiBrain",
        "url": "https://github.com/tu-usuario/logibrain",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────
# En producción reemplaza allow_origins=["*"] por el dominio real del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
app.include_router(ml_router)      # POST /predict
app.include_router(chat_router)    # POST /chat


# ─────────────────────────────────────────────────────────────────
#  Endpoints base
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Sistema"], summary="Raíz de la API")
async def root() -> dict:
    return {
        "sistema": "LogiBrain API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "prediccion_ml": "POST /predict",
            "agente_ia":     "POST /chat",
        },
    }


@app.get("/health", tags=["Sistema"], summary="Health-check")
async def health_check() -> dict:
    """
    Verifica que la API y la conexión a la base de datos estén operativas.
    Útil para monitoreo en Docker / Kubernetes.
    """
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "database": db_status,
    }
