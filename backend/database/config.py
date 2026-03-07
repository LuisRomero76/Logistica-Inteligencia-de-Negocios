"""
database/config.py
──────────────────
Configuración central de SQLAlchemy 2.0:
  • Engine (pool de conexiones hacia PostgreSQL)
  • SessionLocal (fábrica de sesiones para operaciones ORM)
  • Base (clase padre de todos los modelos declarativos)
  • get_db() (dependency injection para FastAPI)
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

# Carga .env antes de leer variables de entorno
load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://logibrain:logibrain123@localhost:5432/logibrain_db",
)

# pool_pre_ping=True detecta conexiones caídas y las renueva automáticamente
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,  # Cambia a True para ver el SQL generado en consola (depuración)
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """Clase base que hereda todos los modelos ORM del proyecto."""
    pass


# ── Dependency Injection para FastAPI ───────────────────────────
def get_db():
    """
    Generador de sesiones de BD para inyectar como dependencia en los
    endpoints de FastAPI.  Garantiza el cierre de la sesión al terminar
    la request, incluso si se lanza una excepción.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
