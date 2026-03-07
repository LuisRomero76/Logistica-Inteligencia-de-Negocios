from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ai_agent.sql_assistant import LogiBrainSQLAssistant

router = APIRouter(tags=["AI Asistente — Text-to-SQL"])

_assistant: LogiBrainSQLAssistant | None = None


def _get_assistant() -> LogiBrainSQLAssistant:
    global _assistant
    if _assistant is None:
        _assistant = LogiBrainSQLAssistant()
    return _assistant


class ChatRequest(BaseModel):
    pregunta: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Pregunta analítica en lenguaje natural.",
        examples=["¿Cuál es el conductor con mayor retraso promedio?"],
    )

    model_config = {"json_schema_extra": {
        "example": {
            "pregunta": "¿Cuántas entregas hubo con clima tormentoso en 2023?",
        }
    }}


class ChatResponse(BaseModel):
    respuesta: str = Field(description="Respuesta en lenguaje natural generada por el agente.")
    pregunta_original: str = Field(description="La pregunta tal como fue recibida.")
    sql_generado: str | None = Field(
        default=None,
        description="Consulta SQL que ejecutó el agente (cuando está disponible).",
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Consulta analítica en lenguaje natural",
    response_description="Respuesta del agente Text-to-SQL.",
)
async def chat_with_data(request: ChatRequest) -> ChatResponse:
    try:
        assistant = _get_assistant()
    except Exception as exc:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"No se pudo inicializar el agente IA.\n"
                f"Error: {type(exc).__name__}: {exc}\n\n"
                f"Traceback:\n{tb}"
            ),
        )

    result = assistant.consultar(request.pregunta)

    if result.get("error"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["respuesta"],
        )

    return ChatResponse(
        respuesta=result["respuesta"],
        pregunta_original=request.pregunta,
        sql_generado=result.get("sql_generado"),
    )
