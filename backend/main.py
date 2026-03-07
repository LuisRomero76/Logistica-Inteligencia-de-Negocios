from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database.config import engine

import database.models as _models

from database.config import Base
from api.routes_ml import router as ml_router
from api.routes_chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas del Data Warehouse verificadas / creadas correctamente.")
    yield
    print("👋 LogiBrain API apagándose.")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ml_router)
app.include_router(chat_router)


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
