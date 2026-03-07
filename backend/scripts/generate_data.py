from __future__ import annotations

import os
import sys
import random
import datetime

import numpy as np
from faker import Faker
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from database.config import SessionLocal, engine
from database.models import (
    Base,
    DimClima,
    DimConductores,
    DimRutas,
    DimVehiculos,
    FactEntregas,
)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker("es_ES")
Faker.seed(RANDOM_SEED)

NUM_CONDUCTORES = 50
NUM_VEHICULOS = 20
NUM_RUTAS = 10
NUM_ENTREGAS = 10_000

ANO_ACTUAL = 2024

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


def _seed_clima(db: Session) -> list[DimClima]:
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
    conductores: list[DimConductores] = []
    for _ in range(NUM_CONDUCTORES):
        experiencia = int(np.clip(np.random.exponential(scale=7), 1, 25))
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


def _calcular_minutos_retraso(
    ruta: DimRutas,
    conductor: DimConductores,
    vehiculo: DimVehiculos,
    clima: DimClima,
) -> float:
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


def _seed_fact_entregas(
    db: Session,
    climas: list[DimClima],
    vehiculos: list[DimVehiculos],
    conductores: list[DimConductores],
    rutas: list[DimRutas],
    n: int = NUM_ENTREGAS,
) -> None:
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


def main() -> None:
    print("═" * 60)
    print("  LogiBrain ETL — Generación de datos sintéticos")
    print("═" * 60)

    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
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
