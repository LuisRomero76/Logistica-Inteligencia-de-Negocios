"""
scripts/generate_data.py
────────────────────────
Pipeline ETL para poblar el Data Warehouse con datos sintéticos realistas.

Lógica de correlaciones (negocio logístico simulado):
  • Clima tormentoso  → base de retraso alta  (~55 min)
  • Clima nevado      → base de retraso media (~35 min)
  • Clima lluvioso    → base de retraso baja  (~18 min)
  • Clima soleado     → base de retraso mínima(~ 2 min)
  • Conductor inexperto (< 5 años) → penalización extra de hasta +22 min
  • Calificación baja               → penalización extra de hasta +12 min
  • Ruta larga (> 800 km)          → penalización proporcional de hasta +25 min
  • Vehículo antiguo (> 10 años)   → penalización por desgaste de hasta +15 min
  • Ruido gaussiano σ=6            → variabilidad realista

Uso:
  cd backend
  python scripts/generate_data.py
"""

from __future__ import annotations

import os
import sys
import random
import datetime

import numpy as np
from faker import Faker
from sqlalchemy.orm import Session

# ── Ajuste de sys.path para ejecutar el script directamente ───────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.config import SessionLocal, engine  # noqa: E402
from database.models import (  # noqa: E402
    Base,
    DimClima,
    DimConductores,
    DimRutas,
    DimVehiculos,
    FactEntregas,
)

# ─────────────────────────────────────────────────────────────────
#  Configuración de seeds para reproducibilidad
# ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker("es_ES")
Faker.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────
#  Parámetros de negocio
# ─────────────────────────────────────────────────────────────────
NUM_CONDUCTORES = 50
NUM_VEHICULOS = 20
NUM_RUTAS = 10
NUM_ENTREGAS = 10_000

ANO_ACTUAL = 2024

# Configuración de cada condición climática:
#   base_retraso → minutos base que añade ese clima
#   temp_range   → rango de temperatura representativo (°C)
CLIMA_CONFIG: dict[str, dict] = {
    "soleado":    {"base_retraso": 2,  "temp_range": (18, 35)},
    "lluvioso":   {"base_retraso": 18, "temp_range": (8,  20)},
    "nevado":     {"base_retraso": 35, "temp_range": (-8,  3)},
    "tormentoso": {"base_retraso": 55, "temp_range": (4,  14)},
}

MARCAS_VEHICULOS = ["Toyota", "Mercedes-Benz", "Volvo", "Scania", "MAN", "Iveco", "DAF"]

CIUDADES_ESPANA = [
    "Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao",
    "Zaragoza", "Málaga", "Murcia", "Valladolid", "Alicante",
    "Córdoba", "Granada", "Vitoria", "Pamplona", "Santander",
]


# ─────────────────────────────────────────────────────────────────
#  Funciones de seeding (dimensiones)
# ─────────────────────────────────────────────────────────────────

def _seed_clima(db: Session) -> list[DimClima]:
    """Inserta las 4 condiciones climáticas con temperatura representativa."""
    climas: list[DimClima] = []
    for condicion, cfg in CLIMA_CONFIG.items():
        temp = round(random.uniform(*cfg["temp_range"]), 1)
        clima = DimClima(condicion=condicion, temp_promedio=temp)
        db.add(clima)
        climas.append(clima)
    db.flush()
    print(f"  ✓ {len(climas)} climas insertados.")
    return climas


def _seed_vehiculos(db: Session) -> list[DimVehiculos]:
    """Genera la flota de 20 vehículos con variedad de antigüedad y capacidad."""
    vehiculos: list[DimVehiculos] = []
    for _ in range(NUM_VEHICULOS):
        v = DimVehiculos(
            marca=random.choice(MARCAS_VEHICULOS),
            anio=random.randint(2004, 2022),
            capacidad_kg=round(random.uniform(1_000, 22_000), 0),
        )
        db.add(v)
        vehiculos.append(v)
    db.flush()
    print(f"  ✓ {len(vehiculos)} vehículos insertados.")
    return vehiculos


def _seed_conductores(db: Session) -> list[DimConductores]:
    """
    Genera 50 conductores con distribución realista de experiencia.
    La experiencia sigue una distribución sesgada a la derecha (más conductores
    con 1-10 años que con 20-25 años) para simular la rotación laboral.
    """
    conductores: list[DimConductores] = []
    for _ in range(NUM_CONDUCTORES):
        # Distribución sesgada: mayoría con poca-media experiencia
        experiencia = int(np.clip(np.random.exponential(scale=7), 1, 25))
        # Calificación correlacionada positivamente con la experiencia + ruido
        base_calificacion = 2.5 + (experiencia / 25) * 2.0
        calificacion = round(
            np.clip(base_calificacion + np.random.normal(0, 0.4), 1.0, 5.0), 1
        )
        c = DimConductores(
            nombre=fake.name(),
            experiencia_anios=experiencia,
            calificacion=calificacion,
        )
        db.add(c)
        conductores.append(c)
    db.flush()
    print(f"  ✓ {len(conductores)} conductores insertados.")
    return conductores


def _seed_rutas(db: Session) -> list[DimRutas]:
    """
    Genera 10 rutas únicas entre ciudades españolas.
    La distancia es aleatoria y uniforme en [50, 1 200] km.
    """
    rutas: list[DimRutas] = []
    pares_usados: set[tuple[str, str]] = set()

    intentos = 0
    while len(rutas) < NUM_RUTAS and intentos < 200:
        intentos += 1
        origen = random.choice(CIUDADES_ESPANA)
        candidatas = [c for c in CIUDADES_ESPANA if c != origen]
        destino = random.choice(candidatas)
        if (origen, destino) not in pares_usados:
            pares_usados.add((origen, destino))
            r = DimRutas(
                origen=origen,
                destino=destino,
                distancia_km=round(random.uniform(50, 1_200), 1),
            )
            db.add(r)
            rutas.append(r)

    db.flush()
    print(f"  ✓ {len(rutas)} rutas insertadas.")
    return rutas


# ─────────────────────────────────────────────────────────────────
#  Función de negocio: cálculo de retraso con correlaciones reales
# ─────────────────────────────────────────────────────────────────

def _calcular_minutos_retraso(
    ruta: DimRutas,
    conductor: DimConductores,
    vehiculo: DimVehiculos,
    clima: DimClima,
) -> float:
    """
    Función determinista con ruido que simula el retraso de una entrega.

    Aportaciones al retraso (todas positivas, acumulables):
    ┌─────────────────────────────────────────────────────────────┐
    │ Factor           │ Fórmula                │ Rango aprox.   │
    ├─────────────────────────────────────────────────────────────┤
    │ Clima            │ base fija por condición │  2 –  55 min  │
    │ Inexperto        │ max(0, 15-exp)*1.8      │  0 –  25 min  │
    │ Baja calificacion│ (5-cal)*3.0             │  0 –  12 min  │
    │ Distancia larga  │ (dist/1200)*25          │  1 –  25 min  │
    │ Vehículo antiguo │ antiguedad*0.8          │  2 –  16 min  │
    │ Ruido gaussiano  │ N(0, 6)                 │ variable      │
    └─────────────────────────────────────────────────────────────┘
    Total mínimo garantizado: 0 min (clip).
    """
    cfg = CLIMA_CONFIG[clima.condicion]

    clima_factor = cfg["base_retraso"]
    experiencia_factor = max(0.0, (15 - conductor.experiencia_anios) * 1.8)
    calificacion_factor = (5.0 - conductor.calificacion) * 3.0
    distancia_factor = (ruta.distancia_km / 1_200) * 25.0
    antiguedad = ANO_ACTUAL - vehiculo.anio
    vehiculo_factor = antiguedad * 0.8
    ruido = np.random.normal(0, 6)

    total = (
        clima_factor
        + experiencia_factor
        + calificacion_factor
        + distancia_factor
        + vehiculo_factor
        + ruido
    )
    return max(0.0, round(total, 2))


# ─────────────────────────────────────────────────────────────────
#  Seeding de la tabla de hechos
# ─────────────────────────────────────────────────────────────────

def _seed_fact_entregas(
    db: Session,
    climas: list[DimClima],
    vehiculos: list[DimVehiculos],
    conductores: list[DimConductores],
    rutas: list[DimRutas],
    n: int = NUM_ENTREGAS,
) -> None:
    """
    Genera n registros de entregas con fechas distribuidas en los
    últimos 2 años.  Inserta en lotes de 1 000 para optimizar I/O.
    """
    fecha_inicio = datetime.datetime(2022, 1, 1)
    batch: list[FactEntregas] = []
    BATCH_SIZE = 1_000

    for i in range(n):
        vehiculo = random.choice(vehiculos)
        conductor = random.choice(conductores)
        ruta = random.choice(rutas)
        clima = random.choice(climas)

        minutos = _calcular_minutos_retraso(ruta, conductor, vehiculo, clima)
        dias_offset = random.randint(0, 730)
        fecha = fecha_inicio + datetime.timedelta(days=dias_offset)

        entrega = FactEntregas(
            id_vehiculo=vehiculo.id,
            id_conductor=conductor.id,
            id_ruta=ruta.id,
            id_clima=clima.id,
            fecha_entrega=fecha,
            minutos_retraso=minutos,
        )
        batch.append(entrega)

        if len(batch) == BATCH_SIZE:
            db.add_all(batch)
            db.flush()
            batch.clear()
            print(f"  → {i + 1:,}/{n:,} entregas procesadas...")

    if batch:
        db.add_all(batch)
        db.flush()

    print(f"  ✓ {n:,} entregas insertadas en fact_entregas.")


# ─────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("═" * 60)
    print("  LogiBrain ETL — Generación de datos sintéticos")
    print("═" * 60)

    # Asegura que las tablas existan antes de insertar
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        # Idempotencia: verifica si ya hay datos para evitar duplicados
        if db.query(DimClima).count() > 0:
            print(
                "\n⚠️  La base de datos ya contiene registros.\n"
                "   Si deseas regenerarlos, vacía las tablas primero y vuelve a ejecutar."
            )
            return

        print("\n[1/4] Insertando dimensiones...")
        climas = _seed_clima(db)
        vehiculos = _seed_vehiculos(db)
        conductores = _seed_conductores(db)
        rutas = _seed_rutas(db)

        print(f"\n[2/4] Generando {NUM_ENTREGAS:,} registros en Fact_Entregas...")
        _seed_fact_entregas(db, climas, vehiculos, conductores, rutas)

        print("\n[3/4] Committing transacción...")
        db.commit()

    print("\n[4/4] ETL completado exitosamente ✅")
    print("═" * 60)
    print("  Siguiente paso: ejecuta  python ml/train_model.py")
    print("═" * 60)


if __name__ == "__main__":
    main()
