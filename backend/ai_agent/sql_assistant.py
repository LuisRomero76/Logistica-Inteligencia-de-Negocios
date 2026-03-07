"""
ai_agent/sql_assistant.py
─────────────────────────
Agente Text-to-SQL basado en LangChain + Groq (tier free).

Arquitectura del agente:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Usuario: pregunta en lenguaje natural                          │
  │       ↓                                                         │
  │  Groq — llama-3.3-70b-versatile (free tier, muy rápido)        │
  │       ↓  [Razonamiento + selección de herramienta]              │
  │  SQLDatabaseToolkit                                             │
  │    • sql_db_list_tables   → lista las tablas disponibles        │
  │    • sql_db_schema        → obtiene la definición de tabla(s)   │
  │    • sql_db_query         → ejecuta la consulta SQL             │
  │    • sql_db_query_checker → valida SQL antes de ejecutarlo      │
  │       ↓                                                         │
  │  PostgreSQL (logibrain_db)                                      │
  │       ↓                                                         │
  │  Respuesta en lenguaje natural                                  │
  └─────────────────────────────────────────────────────────────────┘

Clave de API gratuita:
  Obtén tu GROQ_API_KEY en: https://console.groq.com
  (No requiere tarjeta de crédito — 14,400 req/día en el free tier)

Cambiar a otro proveedor:
  OpenAI  → from langchain_openai import ChatOpenAI
             ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
  Gemini  → from langchain_google_genai import ChatGoogleGenerativeAI
             ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

Seguridad:
  • El SQLDatabase solo expone las tablas del esquema de estrella.
  • El agente opera con permisos SELECT: la URL de BD debe apuntar a un
    usuario de PostgreSQL con permisos de solo lectura en producción.
  • max_iterations=6 evita bucles de razonamiento infinitos.
"""

from __future__ import annotations

import os

import certifi
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

load_dotenv()

# ─────────────────────────────────────────────────────────────────
#  Tablas que el agente puede consultar
#  (principio de menor privilegio: no exponemos tablas del sistema)
# ─────────────────────────────────────────────────────────────────
ALLOWED_TABLES = [
    "dim_vehiculos",
    "dim_conductores",
    "dim_rutas",
    "dim_clima",
    "fact_entregas",
]

# ─────────────────────────────────────────────────────────────────
#  Prompt del sistema para contextualizar al agente
# ─────────────────────────────────────────────────────────────────
SYSTEM_PREFIX = """Eres LogiBrain, un asistente analítico experto en logística y distribución.
Tienes acceso a un Data Warehouse PostgreSQL con el siguiente modelo en estrella:

  • dim_vehiculos   → flota (marca, anio, capacidad_kg)
  • dim_conductores → conductores (nombre, experiencia_anios, calificacion)
  • dim_rutas       → rutas (origen, destino, distancia_km)
  • dim_clima       → clima (condicion: soleado|lluvioso|nevado|tormentoso, temp_promedio)
  • fact_entregas   → historial de entregas (id_vehiculo, id_conductor, id_ruta, id_clima,
                       fecha_entrega, minutos_retraso)

INSTRUCCIONES:
1. Responde siempre en español.
2. Usa JOINs apropiados cuando necesites datos de múltiples tablas.
3. Sé conciso pero informativo. Si el resultado es largo, resume los hallazgos clave.
4. Si la pregunta es ambigua, aclara qué entendiste antes de dar la respuesta.
5. No ejecutes operaciones DML (INSERT, UPDATE, DELETE, DROP). Solo SELECT.
"""


class LogiBrainSQLAssistant:
    """
    Agente conversacional para consultas analíticas sobre el Data Warehouse.

    Uso:
        assistant = LogiBrainSQLAssistant()
        result = assistant.consultar("¿Cuál es el conductor con mayor retraso promedio?")
        print(result["respuesta"])
    """

    def __init__(self) -> None:
        self._agent = None
        self._initialize()

    def _initialize(self) -> None:
        database_url = os.getenv("DATABASE_URL")
        api_key = os.getenv("GROQ_API_KEY")

        if not database_url:
            raise EnvironmentError(
                "DATABASE_URL no está configurada en las variables de entorno."
            )
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY no está configurada. "
                "Obtén tu clave gratuita en https://console.groq.com "
                "y agrégala en el archivo .env para activar el agente /chat."
            )

        # ── Normaliza el scheme para psycopg2 ─────────────────────
        db_url = database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

        # ── Conecta solo las tablas permitidas ─────────────────────
        db = SQLDatabase.from_uri(
            db_url,
            include_tables=ALLOWED_TABLES,
            sample_rows_in_table_info=2,
        )

        # ── Fix SSL para entornos Anaconda/Windows ─────────────────
        # Conda define SSL_CERT_FILE apuntando a un archivo que puede no
        # existir en el entorno virtual. httpx (cliente async de Groq) lee
        # esa variable al inicializar el transporte, causando FileNotFoundError.
        # Sobreescribimos con el path real de certifi antes de instanciar ChatGroq.
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

        # ── Modelo de lenguaje: Groq llama-3.3-70b (free tier) ────
        # llama-3.3-70b-versatile → mejor modelo de Groq para Text-to-SQL
        # Groq usa LPU (Language Processing Unit) → inferencia extremadamente rápida
        # Para cambiar a otro proveedor, reemplaza estas 3 líneas:
        #   OpenAI → from langchain_openai import ChatOpenAI
        #            ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
        #   Gemini → from langchain_google_genai import ChatGoogleGenerativeAI
        #            ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,          # 0 → respuestas deterministas y precisas
            groq_api_key=api_key,
        )

        # ── Toolkit estándar de SQL ────────────────────────────────
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # ── Agente con tools de SQL ────────────────────────────────
        # agent_type="tool-calling" es compatible con llama3 en Groq
        self._agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type="tool-calling",
            prefix=SYSTEM_PREFIX,
            verbose=False,           # Cambia a True para ver el razonamiento paso a paso
            max_iterations=6,
            handle_parsing_errors=True,
        )

    def consultar(self, pregunta: str) -> dict:
        """
        Ejecuta una consulta en lenguaje natural contra el Data Warehouse.

        Returns
        -------
        dict con claves:
            • respuesta   (str)  → respuesta en lenguaje natural
            • sql_generado (str | None) → SQL ejecutado si está disponible
            • error       (bool) → True si ocurrió un error
        """
        try:
            resultado = self._agent.invoke({"input": pregunta})
            respuesta = resultado.get("output", "No se pudo generar una respuesta.")
            return {
                "respuesta": respuesta,
                "sql_generado": None,  # El agente de herramientas no expone el SQL directamente
                "error": False,
            }
        except Exception as exc:
            return {
                "respuesta": (
                    f"Lo siento, ocurrió un error al procesar tu consulta: {exc}. "
                    "Por favor, reformula la pregunta o inténtalo de nuevo."
                ),
                "sql_generado": None,
                "error": True,
            }
