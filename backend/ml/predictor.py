"""
ml/predictor.py
───────────────
Clase que encapsula la inferencia del modelo LogiBrain.

Diseño:
  • Carga el pipeline serializado (preprocessing + RF) una sola vez al
    instanciar la clase (patrón Singleton lazy vía routes_ml.py).
  • El método `predecir_retraso` recibe un dict con los mismos nombres
    de columna usados durante el entrenamiento, construye un DataFrame
    de 1 fila y devuelve el retraso predicho en minutos (float ≥ 0).

Ventaja del pipeline completo:
  El OrdinalEncoder está incluido en el pkl → no hay desajuste de
  transformaciones entre entrenamiento e inferencia.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "logibrain_model.pkl")

# Orden explícito de columnas: debe coincidir con el entrenamiento
_FEATURE_ORDER = [
    "distancia_km",
    "experiencia_anios",
    "calificacion",
    "capacidad_kg",
    "antiguedad_vehiculo",
    "temp_promedio",
    "condicion_clima",
]

_VALID_CLIMAS = {"soleado", "lluvioso", "nevado", "tormentoso"}


class LogiBrainPredictor:
    """
    Wrapper de inferencia para el modelo RandomForest de LogiBrain.

    Uso:
        predictor = LogiBrainPredictor()
        minutos = predictor.predecir_retraso({
            "distancia_km": 450.0,
            "experiencia_anios": 3,
            "calificacion": 3.2,
            "capacidad_kg": 8000.0,
            "antiguedad_vehiculo": 8,
            "temp_promedio": 12.5,
            "condicion_clima": "lluvioso",
        })
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._load()

    def _load(self) -> None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modelo no encontrado en '{MODEL_PATH}'. "
                "Ejecuta primero:  python ml/train_model.py"
            )
        self._pipeline = joblib.load(MODEL_PATH)

    def predecir_retraso(self, features: dict) -> float:
        """
        Predice el retraso de una entrega en minutos.

        Parameters
        ----------
        features : dict
            Claves requeridas (ver _FEATURE_ORDER arriba).

        Returns
        -------
        float
            Minutos de retraso predichos (≥ 0).

        Raises
        ------
        ValueError
            Si faltan columnas requeridas o el clima no es válido.
        """
        # ── Validación de entrada ──────────────────────────────────
        missing = [col for col in _FEATURE_ORDER if col not in features]
        if missing:
            raise ValueError(f"Faltan las siguientes columnas: {missing}")

        clima = features.get("condicion_clima", "").lower()
        if clima not in _VALID_CLIMAS:
            raise ValueError(
                f"condicion_clima='{clima}' no válido. "
                f"Opciones: {sorted(_VALID_CLIMAS)}"
            )

        # ── Construcción del DataFrame de inferencia ───────────────
        row = {col: features[col] for col in _FEATURE_ORDER}
        row["condicion_clima"] = clima          # normalizado a minúsculas
        df = pd.DataFrame([row], columns=_FEATURE_ORDER)

        # ── Inferencia y clip a 0 ──────────────────────────────────
        prediccion = float(self._pipeline.predict(df)[0])
        return max(0.0, round(prediccion, 2))

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
