"""
ml/train_model.py
─────────────────
Pipeline de entrenamiento del modelo predictivo de LogiBrain.

Objetivo → predecir `minutos_retraso` (regresión continua).

Flujo:
  1. Extrae datos desde PostgreSQL con un JOIN de estrella (Pandas + SQLAlchemy).
  2. Ingeniería de características:
       • antigüedad del vehículo  (año_actual - anio_fabricacion)
       • codificación ordinal del clima (soleado < lluvioso < nevado < tormentoso)
  3. Divide en train/test (80/20, estratificado por clima).
  4. Construye un sklearn Pipeline:
       ColumnTransformer → passthrough + OrdinalEncoder
       RandomForestRegressor (150 árboles, profundidad 12)
  5. Evalúa con MAE y RMSE.
  6. Serializa el pipeline completo con joblib → logibrain_model.pkl

Uso:
  cd backend
  python ml/train_model.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ── Ajuste sys.path para ejecución directa ────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.config import engine  # noqa: E402

# ─────────────────────────────────────────────────────────────────
#  Rutas de artefactos
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logibrain_model.pkl")

# Orden de las categorías de clima (de menor a mayor severidad)
# → el OrdinalEncoder asigna 0, 1, 2, 3 respectivamente, lo que
#   proporciona información ordinal al modelo de árbol.
CLIMA_CATEGORIES = ["soleado", "lluvioso", "nevado", "tormentoso"]

FEATURE_COLS_NUM = [
    "distancia_km",
    "experiencia_anios",
    "calificacion",
    "capacidad_kg",
    "antiguedad_vehiculo",
    "temp_promedio",
]
FEATURE_COLS_CAT = ["condicion_clima"]
TARGET_COL = "minutos_retraso"


# ─────────────────────────────────────────────────────────────────
#  1. Extracción de datos (ETL hacia DataFrame)
# ─────────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    """
    Ejecuta un JOIN entre la tabla de hechos y las 4 dimensiones,
    devolviendo un DataFrame plano listo para el entrenamiento.
    La consulta evita cargar columnas innecesarias (id, nombre, etc.)
    para mantener el conjunto de features limpio.
    """
    query = text("""
        SELECT
            f.minutos_retraso,
            r.distancia_km,
            c.experiencia_anios,
            c.calificacion,
            v.capacidad_kg,
            (2024 - v.anio)         AS antiguedad_vehiculo,
            cl.temp_promedio,
            cl.condicion            AS condicion_clima
        FROM  fact_entregas   f
        JOIN  dim_rutas       r  ON f.id_ruta      = r.id
        JOIN  dim_conductores c  ON f.id_conductor = c.id
        JOIN  dim_vehiculos   v  ON f.id_vehiculo  = v.id
        JOIN  dim_clima       cl ON f.id_clima     = cl.id
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# ─────────────────────────────────────────────────────────────────
#  2. Pipeline de sklearn
# ─────────────────────────────────────────────────────────────────

def _build_pipeline() -> Pipeline:
    """
    Construye el pipeline de preprocesamiento + modelo.

    ColumnTransformer:
      • numéricas  → passthrough (ya están en escala correcta para RF)
      • categórica → OrdinalEncoder con orden semántico de severidad

    RandomForestRegressor:
      • 150 estimadores → buen balance bias-varianza con 10 k registros
      • max_depth=12    → evita sobreajuste en árbol profundo
      • n_jobs=-1       → paralelismo completo en CPU
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURE_COLS_NUM),
            (
                "cat",
                OrdinalEncoder(
                    categories=[CLIMA_CATEGORIES],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                FEATURE_COLS_CAT,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ─────────────────────────────────────────────────────────────────
#  3. Entrenamiento, evaluación y serialización
# ─────────────────────────────────────────────────────────────────

def train() -> Pipeline:
    print("═" * 60)
    print("  LogiBrain ML — Entrenamiento del modelo predictivo")
    print("═" * 60)

    # ── Carga ──────────────────────────────────────────────────────
    print("\n[1/4] Extrayendo datos desde PostgreSQL...")
    df = _load_data()
    print(f"  Dataset: {len(df):,} registros × {df.shape[1]} columnas.")
    print(f"  Target mean/std: {df[TARGET_COL].mean():.2f} / {df[TARGET_COL].std():.2f} min")

    # ── Ingeniería de features ─────────────────────────────────────
    X = df[FEATURE_COLS_NUM + FEATURE_COLS_CAT]
    y = df[TARGET_COL]

    # ── Split ──────────────────────────────────────────────────────
    print("\n[2/4] Dividiendo en train (80%) / test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── Ajuste ─────────────────────────────────────────────────────
    print("\n[3/4] Entrenando RandomForestRegressor (150 árboles)...")
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    # ── Evaluación ─────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = pipeline.score(X_test, y_test)

    print("\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  MAE  (Error Absoluto Medio): {mae:>7.2f} minutos   │")
    print(f"  │  RMSE (Raíz del ECM):         {rmse:>7.2f} minutos   │")
    print(f"  │  R²   (Coef. determinación):  {r2:>7.4f}             │")
    print("  └──────────────────────────────────────────────────┘")

    # ── Feature importances ────────────────────────────────────────
    feature_names = FEATURE_COLS_NUM + FEATURE_COLS_CAT
    importances = pipeline.named_steps["model"].feature_importances_
    fi = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    print("\n  Importancia de variables:")
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"    {row['feature']:<25} {row['importance']:.4f}  {bar}")

    # ── Serialización ──────────────────────────────────────────────
    print(f"\n[4/4] Guardando modelo en: {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)
    print("\n  Modelo guardado exitosamente ✅")
    print("═" * 60)

    return pipeline


if __name__ == "__main__":
    train()
