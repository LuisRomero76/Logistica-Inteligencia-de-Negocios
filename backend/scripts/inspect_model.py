"""
scripts/inspect_model.py
────────────────────────
Utilidad para inspeccionar y visualizar el modelo serializado logibrain_model.pkl.

Muestra:
  • Parámetros del RandomForestRegressor
  • Importancia de cada variable (feature importance)
  • Estadísticas de los árboles del bosque
  • Una predicción de ejemplo

Uso:
  cd backend
  python scripts/inspect_model.py
"""

from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "logibrain_model.pkl")

FEATURE_NAMES = [
    "distancia_km",
    "experiencia_anios",
    "calificacion",
    "capacidad_kg",
    "antiguedad_vehiculo",
    "temp_promedio",
    "condicion_clima",
]


def main() -> None:
    print("═" * 60)
    print("  LogiBrain — Inspección del modelo logibrain_model.pkl")
    print("═" * 60)

    # ── Carga ──────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Modelo no encontrado en: {MODEL_PATH}")
        print("   Ejecuta primero:  python ml/train_model.py")
        return

    pipeline = joblib.load(MODEL_PATH)
    rf_model = pipeline.named_steps["model"]

    # ── Info general ───────────────────────────────────────────────
    print(f"\n📦 Archivo:       {os.path.abspath(MODEL_PATH)}")
    print(f"   Tamaño:        {os.path.getsize(MODEL_PATH) / 1024:.1f} KB")
    print(f"   Tipo:          {type(rf_model).__name__}")

    print(f"\n📐 Hiperparámetros:")
    print(f"   n_estimators:    {rf_model.n_estimators}")
    print(f"   max_depth:       {rf_model.max_depth}")
    print(f"   min_samples_leaf:{rf_model.min_samples_leaf}")
    print(f"   random_state:    {rf_model.random_state}")
    print(f"   n_jobs:          {rf_model.n_jobs}")

    # ── Estadísticas de los árboles ────────────────────────────────
    depths = [estimator.get_depth() for estimator in rf_model.estimators_]
    leaves = [estimator.get_n_leaves() for estimator in rf_model.estimators_]
    print(f"\n🌲 Estadísticas del bosque ({rf_model.n_estimators} árboles):")
    print(f"   Profundidad media:  {np.mean(depths):.1f}  (min {min(depths)}, max {max(depths)})")
    print(f"   Hojas promedio:     {np.mean(leaves):.0f}  (min {min(leaves)}, max {max(leaves)})")

    # ── Feature Importances ────────────────────────────────────────
    importances = rf_model.feature_importances_
    fi = pd.DataFrame({"feature": FEATURE_NAMES, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n📊 Importancia de variables (mayor = más determinante en la predicción):")
    print(f"   {'Variable':<25} {'Importancia':>10}   Barra")
    print("   " + "─" * 55)
    for _, row in fi.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"   {row['feature']:<25} {row['importance']:>10.4f}   {bar}")

    # ── Predicciones de ejemplo ────────────────────────────────────
    print("\n🔮 Predicciones de ejemplo:")
    ejemplos = [
        {
            "descripcion": "Ruta corta, conductor experto, clima soleado",
            "distancia_km": 80.0, "experiencia_anios": 20, "calificacion": 4.8,
            "capacidad_kg": 5000.0, "antiguedad_vehiculo": 2,
            "temp_promedio": 25.0, "condicion_clima": "soleado",
        },
        {
            "descripcion": "Ruta larga, conductor inexperto, tormenta",
            "distancia_km": 1100.0, "experiencia_anios": 1, "calificacion": 2.1,
            "capacidad_kg": 15000.0, "antiguedad_vehiculo": 18,
            "temp_promedio": 8.0, "condicion_clima": "tormentoso",
        },
        {
            "descripcion": "Ruta media, clima lluvioso, conductor promedio",
            "distancia_km": 400.0, "experiencia_anios": 7, "calificacion": 3.5,
            "capacidad_kg": 8000.0, "antiguedad_vehiculo": 6,
            "temp_promedio": 14.0, "condicion_clima": "lluvioso",
        },
    ]

    print(f"   {'Escenario':<46} {'Predicción':>12}")
    print("   " + "─" * 62)
    for ej in ejemplos:
        desc = ej.pop("descripcion")
        df = pd.DataFrame([ej], columns=FEATURE_NAMES)
        pred = float(pipeline.predict(df)[0])
        print(f"   {desc:<46} {pred:>8.1f} min")

    print("\n═" * 60)
    print("  Inspección completada ✅")
    print("═" * 60)


if __name__ == "__main__":
    main()
