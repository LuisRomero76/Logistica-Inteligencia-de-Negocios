from __future__ import annotations

import os

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "logibrain_model.pkl")

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
        missing = [col for col in _FEATURE_ORDER if col not in features]
        if missing:
            raise ValueError(f"Faltan las siguientes columnas: {missing}")

        clima = features.get("condicion_clima", "").lower()
        if clima not in _VALID_CLIMAS:
            raise ValueError(
                f"condicion_clima='{clima}' no válido. "
                f"Opciones: {sorted(_VALID_CLIMAS)}"
            )

        row = {col: features[col] for col in _FEATURE_ORDER}
        row["condicion_clima"] = clima
        df = pd.DataFrame([row], columns=_FEATURE_ORDER)

        prediccion = float(self._pipeline.predict(df)[0])
        return max(0.0, round(prediccion, 2))

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None
