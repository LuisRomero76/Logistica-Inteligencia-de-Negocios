"""
api/routes_chat.py
──────────────────
Endpoint REST para el agente conversacional Text-to-SQL.

POST /chat
  → Recibe una pregunta en lenguaje natural (español)
  → El agente LangChain traduce la pregunta a SQL, la ejecuta contra
     el Data Warehouse y devuelve una respuesta en lenguaje natural.

Diseño:
  • El agente se inicializa de forma lazy para no bloquear el arranque
    si la clave de API aún no está configurada.
  • Cuando la GROQ_API_KEY no está disponible, se devuelve HTTP 503
    con un mensaje claro en lugar de un crash silencioso.
"""

from __future__ import annotations

import traceback

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ai_agent.sql_assistant import LogiBrainSQLAssistant

router = APIRouter(tags=["AI Asistente — Text-to-SQL"])

# ── Singleton lazy del asistente ────────────────────────────────
_assistant: LogiBrainSQLAssistant | None = None


def _get_assistant() -> LogiBrainSQLAssistant:
    global _assistant
    if _assistant is None:
        _assistant = LogiBrainSQLAssistant()
    return _assistant


# ─────────────────────────────────────────────────────────────────
#  Schemas Pydantic
# ─────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────
#  Endpoint
# ─────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Consulta analítica en lenguaje natural",
    response_description="Respuesta del agente Text-to-SQL.",
)
async def chat_with_data(request: ChatRequest) -> ChatResponse:
    """
    Envía una pregunta en español al agente LogiBrain.
    El agente utiliza LangChain para:
      1. Interpretar la intención de la pregunta.
      2. Generar la consulta SQL adecuada contra el Data Warehouse.
      3. Ejecutar la consulta y formular una respuesta en lenguaje natural.

    > **Nota**: Requiere que `GROQ_API_KEY` esté definida en el archivo `.env`.
    > Obtén tu clave gratuita en https://console.groq.com
    """
    try:
        assistant = _get_assistant()
    except Exception as exc:
        # Captura el traceback completo para facilitar el diagnóstico
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
