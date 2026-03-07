# LogiBrain — Predicción de Retrasos y Asistente IA Logístico

Sistema de analítica logística corporativa que combina **Data Engineering**, **Machine Learning** e **Inteligencia Artificial Generativa** para predecir retrasos en entregas y responder consultas en lenguaje natural sobre el Data Warehouse.

---

## Tabla de contenidos

1. [¿De qué trata el proyecto?](#1-de-qué-trata-el-proyecto)
2. [Arquitectura](#2-arquitectura)
3. [Stack tecnológico](#3-stack-tecnológico)
4. [Estructura del repositorio](#4-estructura-del-repositorio)
5. [Requisitos previos](#5-requisitos-previos)
6. [Variables de entorno](#6-variables-de-entorno)
7. [Levantar el proyecto](#7-levantar-el-proyecto)
8. [API — Endpoints](#8-api--endpoints)
9. [Modelo de datos (Star Schema)](#9-modelo-de-datos-star-schema)
10. [Modelo de Machine Learning](#10-modelo-de-machine-learning)
11. [Agente IA (Text-to-SQL)](#11-agente-ia-text-to-sql)

---

## 1. ¿De qué trata el proyecto?

**LogiBrain** es una plataforma de analítica logística que integra tres capas:

| Capa | Descripción |
|---|---|
| **Data Engineering** | Star Schema en PostgreSQL con 10 000 registros sintéticos generados con Faker que simulan operaciones logísticas reales (vehículos, conductores, rutas, clima y entregas). |
| **Machine Learning** | Modelo RandomForestRegressor entrenado sobre el Data Warehouse para predecir los minutos de retraso de una entrega en función de distancia, experiencia del conductor, condición climática, etc. |
| **IA Generativa** | Agente LangChain (Text-to-SQL) alimentado por el modelo `llama-3.3-70b-versatile` de Groq que traduce preguntas en lenguaje natural a SQL y devuelve respuestas sobre los datos logísticos. |

El frontend Vue 3 expone un **dashboard administrativo** con dos módulos: el predictor ML (formulario → resultado) y el chat con el asistente IA.

---

## 2. Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
│          Vue 3 + Vite + Tailwind CSS v4                     │
│      PredictorML.vue          ChatIA.vue                    │
└───────────────────┬─────────────────────┬───────────────────┘
                    │  POST /predict       │  POST /chat
                    ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND                         │
│   routes_ml.py ──► predictor.py (logibrain_model.pkl)       │
│   routes_chat.py ──► sql_assistant.py (LangChain + Groq)    │
└───────────────────────────────┬─────────────────────────────┘
                                │ SQLAlchemy
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              POSTGRESQL 15 (Docker, puerto 5434)            │
│   dim_vehiculos · dim_conductores · dim_rutas               │
│   dim_clima · fact_entregas                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Stack tecnológico

### Backend
| Tecnología | Versión | Rol |
|---|---|---|
| Python | 3.10+ | Lenguaje principal |
| FastAPI | 0.111.0 | Framework de API REST |
| Uvicorn | 0.29.0 | Servidor ASGI |
| SQLAlchemy | 2.0.30 | ORM / acceso a BD |
| PostgreSQL | 15-alpine | Base de datos (Docker) |
| Scikit-Learn | 1.4.2 | Modelo ML (RandomForest) |
| LangChain | 0.2.5 | Orquestación del agente IA |
| langchain-groq | 0.1.9 | Integración con Groq LLM |
| Faker | 24.11.0 | Generación de datos sintéticos |
| certifi | ≥2024.2.2 | Fix SSL para Anaconda/Windows |

### Frontend
| Tecnología | Versión | Rol |
|---|---|---|
| Vue 3 | 3.x | Framework reactivo |
| Vite | 7.x | Bundler / dev server |
| Tailwind CSS | v4 | Estilos |
| Axios | — | Cliente HTTP |
| lucide-vue-next | — | Iconos |
| pnpm | — | Gestor de paquetes |

---

## 4. Estructura del repositorio

```
LogiBrain/
├── backend/
│   ├── main.py                  # App FastAPI, lifespan, CORS, health-check
│   ├── requirements.txt
│   ├── docker-compose.yml       # PostgreSQL 15 en Docker
│   ├── .env                     # Variables secretas (no versionar)
│   ├── database/
│   │   ├── config.py            # Engine, SessionLocal, Base, get_db()
│   │   └── models.py            # ORM Star Schema (5 tablas)
│   ├── ml/
│   │   ├── train_model.py       # Entrenamiento y serialización del modelo
│   │   ├── predictor.py         # Clase LogiBrainPredictor (inferencia)
│   │   └── logibrain_model.pkl  # Modelo serializado (se genera al entrenar)
│   ├── ai_agent/
│   │   └── sql_assistant.py     # Agente LangChain Text-to-SQL con Groq
│   ├── api/
│   │   ├── routes_ml.py         # POST /predict
│   │   └── routes_chat.py       # POST /chat
│   └── scripts/
│       ├── generate_data.py     # ETL: genera e inserta 10 000 registros
│       └── inspect_model.py     # Utilidad: inspecciona el .pkl
└── front-chat/
    ├── index.html
    ├── vite.config.js
    ├── .env                     # VITE_API_BASE_URL
    └── src/
        ├── main.js
        ├── App.vue              # Layout principal, header, cards de info
        ├── assets/main.css      # @import "tailwindcss"
        ├── components/
        │   ├── PredictorML.vue  # Formulario de predicción
        │   └── ChatIA.vue       # Chat con el asistente IA
        └── services/
            └── api.js           # Cliente Axios centralizado
```

---

## 5. Requisitos previos

Asegúrate de tener instalado:

- **Docker Desktop** — para levantar PostgreSQL
- **Python 3.10+** con `pip` o entorno Conda (recomendado: Anaconda)
- **Node.js 18+** y **pnpm** (`npm install -g pnpm`)
- **Cuenta en Groq** — obtén tu API key gratuita en [console.groq.com](https://console.groq.com)

---

## 6. Variables de entorno

### Backend — `backend/.env`

Crea el archivo `backend/.env` con el siguiente contenido:

```env
# Base de datos
POSTGRES_USER=admin
POSTGRES_PASSWORD=123456
POSTGRES_DB=logibrain_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5434

# Groq (LLM para el agente IA)
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

> Obtén tu `GROQ_API_KEY` en [console.groq.com](https://console.groq.com). El tier gratuito incluye 14 400 requests/día con el modelo `llama-3.3-70b-versatile`.

### Frontend — `front-chat/.env`

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## 7. Levantar el proyecto

### Paso 1 — Iniciar PostgreSQL

```bash
cd backend
docker compose up -d
```

Verifica que el contenedor esté corriendo:

```bash
docker ps
# → logibrain_postgres   postgres:15-alpine   0.0.0.0:5434->5432/tcp
```

---

### Paso 2 — Instalar dependencias Python

Con **Anaconda** (recomendado):

```bash
conda create -n logibrain python=3.10 -y
conda activate logibrain
cd backend
pip install -r requirements.txt
```

Con **venv** estándar:

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

---

### Paso 3 — Generar datos e insertar en el Data Warehouse

Este script crea las tablas (si no existen) e inserta 10 000 registros de entregas sintéticas con sus dimensiones.

```bash
# Desde backend/
python scripts/generate_data.py
```

> La primera vez tarda ~15-30 segundos. Al finalizar verás un resumen con el total de registros insertados por tabla.

---

### Paso 4 — Entrenar el modelo ML

```bash
# Desde backend/
python ml/train_model.py
```

Esto consulta el Data Warehouse, entrena un `RandomForestRegressor` y guarda el modelo en `backend/ml/logibrain_model.pkl`. Al finalizar se imprimen métricas de evaluación (RMSE, MAE, R²).

---

### Paso 5 — Levantar el backend FastAPI

```bash
# Desde backend/
uvicorn main:app --reload
```

La API queda disponible en:
- **http://localhost:8000** — raíz
- **http://localhost:8000/docs** — documentación interactiva (Swagger UI)
- **http://localhost:8000/redoc** — documentación alternativa
- **http://localhost:8000/health** — estado de la API y conexión a BD

---

### Paso 6 — Levantar el frontend

En una **nueva terminal**:

```bash
cd front-chat
pnpm install     # solo la primera vez
pnpm dev
```

El dashboard queda disponible en **http://localhost:5173**.

---

### Resumen de comandos (orden correcto)

```bash
# 1. Base de datos
cd backend && docker compose up -d

# 2. Datos y modelo (solo la primera vez)
python scripts/generate_data.py
python ml/train_model.py

# 3. Backend
uvicorn main:app --reload

# 4. Frontend (nueva terminal)
cd front-chat && pnpm dev
```

---

## 8. API — Endpoints

### `POST /predict` — Predicción de retraso

Predice los minutos de retraso estimados para una entrega.

**Request body:**
```json
{
  "distancia_km": 450.0,
  "experiencia_anios": 5,
  "calificacion": 3.8,
  "capacidad_kg": 8000.0,
  "antiguedad_vehiculo": 7,
  "temp_promedio": 12.5,
  "condicion_clima": "lluvioso"
}
```

**Response:**
```json
{
  "minutos_retraso_predicho": 38.4,
  "nivel_riesgo": "MEDIO",
  "mensaje": "Retraso moderado. Monitorear la entrega."
}
```

| Nivel de riesgo | Criterio |
|---|---|
| `BAJO` | < 15 minutos |
| `MEDIO` | 15 – 45 minutos |
| `ALTO` | > 45 minutos |

---

### `POST /chat` — Asistente IA logístico

Recibe una pregunta en lenguaje natural y devuelve una respuesta generada a partir del Data Warehouse.

**Request body:**
```json
{
  "mensaje": "¿Cuál es el conductor con mayor promedio de retraso en rutas lluviosas?"
}
```

**Response:**
```json
{
  "respuesta": "El conductor con mayor promedio de retraso en condiciones lluviosas es...",
  "sql_generado": "SELECT c.nombre, AVG(f.minutos_retraso) ...",
  "error": null
}
```

---

### `GET /health` — Estado del sistema

```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

---

## 9. Modelo de datos (Star Schema)

```
                    ┌─────────────────┐
                    │  dim_vehiculos  │
                    │  id_vehiculo PK │
                    │  marca          │
                    │  anio           │
                    │  capacidad_kg   │
                    └────────┬────────┘
                             │
┌────────────────┐  ┌────────▼────────────┐  ┌───────────────────┐
│ dim_conductores│  │   fact_entregas     │  │    dim_rutas      │
│ id_conductor PK├──┤ id_entrega PK       ├──┤ id_ruta PK        │
│ nombre         │  │ id_vehiculo FK      │  │ origen            │
│ experiencia    │  │ id_conductor FK     │  │ destino           │
│ calificacion   │  │ id_ruta FK          │  │ distancia_km      │
└────────────────┘  │ id_clima FK         │  └───────────────────┘
                    │ fecha_entrega       │
                    │ minutos_retraso     │  ┌───────────────────┐
                    └────────────────────┘  │    dim_clima      │
                                            │ id_clima PK       │
                                            │ condicion         │
                                            │ temp_promedio     │
                                            └───────────────────┘
```

---

## 10. Modelo de Machine Learning

| Parámetro | Valor |
|---|---|
| Algoritmo | `RandomForestRegressor` |
| Estimadores | 150 |
| Profundidad máxima | 12 |
| Min samples leaf | 5 |
| Variable objetivo | `minutos_retraso` (regresión) |

**Features usadas:**

| Feature | Tipo |
|---|---|
| `distancia_km` | Numérica |
| `experiencia_anios` | Numérica |
| `calificacion` | Numérica |
| `capacidad_kg` | Numérica |
| `antiguedad_vehiculo` | Numérica |
| `temp_promedio` | Numérica |
| `condicion_clima` | Categórica (OrdinalEncoder) |

El pipeline de Scikit-Learn aplica `StandardScaler` a las features numéricas y `OrdinalEncoder` a `condicion_clima`, serializado en `logibrain_model.pkl` con `joblib`.

---

## 11. Agente IA (Text-to-SQL)

El agente usa **LangChain SQLDatabaseToolkit** con el modelo `llama-3.3-70b-versatile` de Groq. Tiene acceso de **solo lectura** a las 5 tablas del Data Warehouse y responde exclusivamente en español.

**Ejemplos de preguntas que puedes hacerle:**

- *¿Cuántas entregas tuvieron más de 60 minutos de retraso el último mes?*
- *¿Qué ruta tiene el mayor promedio de retraso?*
- *Dame el top 5 de conductores con mejor calificación y sus retrasos promedio.*
- *¿Cuántos vehículos tienen más de 10 años de antigüedad?*
- *¿Hay diferencia significativa en retrasos entre clima soleado y tormentoso?*

> **Nota para usuarios de Anaconda en Windows:** El entorno Anaconda puede tener problemas con los certificados SSL al conectarse a la API de Groq. El código ya incluye la corrección automática usando `certifi` (`os.environ["SSL_CERT_FILE"] = certifi.where()`), por lo que no se requiere ninguna acción adicional.
