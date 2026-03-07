"""
api/routes_ml.py
────────────────
Endpoint REST para inferencia del modelo predictivo.

POST /predict
  → Recibe las características de una entrega
  → Devuelve minutos_retraso predichos + nivel de riesgo

Diseño:
  • El predictor se inicializa de forma lazy (solo en la primera request)
    para no bloquear el arranque de la API si el .pkl aún no existe.
  • La validación de entrada la realiza Pydantic automáticamente
    (rangos, tipos, patrón para condición de clima).
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ml.predictor import LogiBrainPredictor

router = APIRouter(tags=["Machine Learning — Predicción"])

# ── Singleton lazy del predictor ────────────────────────────────
_predictor: LogiBrainPredictor | None = None


def _get_predictor() -> LogiBrainPredictor:
    global _predictor
    if _predictor is None:
        _predictor = LogiBrainPredictor()
    return _predictor


# ─────────────────────────────────────────────────────────────────
#  Schemas Pydantic
# ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Características necesarias para predecir el retraso de una entrega."""

    distancia_km: float = Field(
        ..., gt=0, le=5_000,
        description="Distancia de la ruta en kilómetros.",
        examples=[450.0],
    )
    experiencia_anios: int = Field(
        ..., ge=0, le=50,
        description="Años de experiencia del conductor.",
        examples=[5],
    )
    calificacion: float = Field(
        ..., ge=1.0, le=5.0,
        description="Calificación del conductor (1–5).",
        examples=[3.8],
    )
    capacidad_kg: float = Field(
        ..., gt=0, le=50_000,
        description="Capacidad de carga del vehículo en kg.",
        examples=[8000.0],
    )
    antiguedad_vehiculo: int = Field(
        ..., ge=0, le=40,
        description="Antigüedad del vehículo en años (año_actual - año_fabricacion).",
        examples=[7],
    )
    temp_promedio: float = Field(
        ..., ge=-30, le=50,
        description="Temperatura promedio durante la entrega en °C.",
        examples=[12.5],
    )
    condicion_clima: Literal["soleado", "lluvioso", "nevado", "tormentoso"] = Field(
        ...,
        description="Condición meteorológica durante la entrega.",
        examples=["lluvioso"],
    )

    model_config = {"json_schema_extra": {
        "example": {
            "distancia_km": 450.0,
            "experiencia_anios": 3,
            "calificacion": 3.2,
            "capacidad_kg": 8000.0,
            "antiguedad_vehiculo": 8,
            "temp_promedio": 12.5,
            "condicion_clima": "lluvioso",
        }
    }}


class PredictResponse(BaseModel):
    minutos_retraso_predicho: float = Field(
        description="Minutos de retraso estimados por el modelo."
    )
    nivel_riesgo: Literal["BAJO", "MEDIO", "ALTO"] = Field(
        description="Clasificación cualitativa: BAJO (<15 min) | MEDIO (15–45) | ALTO (>45)."
    )
    mensaje: str = Field(description="Mensaje interpretativo para el usuario final.")


# ─────────────────────────────────────────────────────────────────
#  Endpoint
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predice el retraso de una entrega",
    response_description="Retraso estimado en minutos con nivel de riesgo.",
)
async def predict_delay(request: PredictRequest) -> PredictResponse:
    """
    Utiliza el modelo RandomForest serializado para estimar cuántos minutos
    de retraso tendrá una entrega dado su contexto (clima, conductor, ruta,
    vehículo).
    """
    try:
        predictor = _get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo predictivo aún no está disponible. "
                f"Detalle: {exc}"
            ),
        )

    try:
        minutos = predictor.predecir_retraso(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante la inferencia: {exc}",
        )

    # ── Clasificación de riesgo ────────────────────────────────────
    if minutos < 15:
        nivel = "BAJO"
        emoji = "🟢"
    elif minutos <= 45:
        nivel = "MEDIO"
        emoji = "🟡"
    else:
        nivel = "ALTO"
        emoji = "🔴"

    mensaje = (
        f"{emoji} Se estiman {minutos:.1f} min de retraso. "
        f"Nivel de riesgo: {nivel}."
    )

    return PredictResponse(
        minutos_retraso_predicho=minutos,
        nivel_riesgo=nivel,
        mensaje=mensaje,
    )
