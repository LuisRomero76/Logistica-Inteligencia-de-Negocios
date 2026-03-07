from __future__ import annotations

import os

import certifi
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

load_dotenv()

ALLOWED_TABLES = [
    "dim_vehiculos",
    "dim_conductores",
    "dim_rutas",
    "dim_clima",
    "fact_entregas",
]

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

        db_url = database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

        db = SQLDatabase.from_uri(
            db_url,
            include_tables=ALLOWED_TABLES,
            sample_rows_in_table_info=2,
        )

        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=api_key,
        )

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        self._agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            agent_type="tool-calling",
            prefix=SYSTEM_PREFIX,
            verbose=False,
            max_iterations=6,
            handle_parsing_errors=True,
        )

    def consultar(self, pregunta: str) -> dict:
        try:
            resultado = self._agent.invoke({"input": pregunta})
            respuesta = resultado.get("output", "No se pudo generar una respuesta.")
            return {
                "respuesta": respuesta,
                "sql_generado": None,
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
