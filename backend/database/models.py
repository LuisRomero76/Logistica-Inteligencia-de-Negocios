from __future__ import annotations

import datetime
from typing import List

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.config import Base


class DimVehiculos(Base):
    __tablename__ = "dim_vehiculos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    marca: Mapped[str] = mapped_column(String(50), nullable=False)
    anio: Mapped[int] = mapped_column(Integer, nullable=False)
    capacidad_kg: Mapped[float] = mapped_column(Float, nullable=False)

    entregas: Mapped[List["FactEntregas"]] = relationship(
        "FactEntregas", back_populates="vehiculo"
    )

    def __repr__(self) -> str:
        return f"<Vehiculo id={self.id} marca={self.marca} anio={self.anio}>"


class DimConductores(Base):
    __tablename__ = "dim_conductores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False)
    experiencia_anios: Mapped[int] = mapped_column(Integer, nullable=False)
    calificacion: Mapped[float] = mapped_column(Float, nullable=False)

    entregas: Mapped[List["FactEntregas"]] = relationship(
        "FactEntregas", back_populates="conductor"
    )

    def __repr__(self) -> str:
        return f"<Conductor id={self.id} nombre={self.nombre}>"


class DimRutas(Base):
    __tablename__ = "dim_rutas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    origen: Mapped[str] = mapped_column(String(80), nullable=False)
    destino: Mapped[str] = mapped_column(String(80), nullable=False)
    distancia_km: Mapped[float] = mapped_column(Float, nullable=False)

    entregas: Mapped[List["FactEntregas"]] = relationship(
        "FactEntregas", back_populates="ruta"
    )

    def __repr__(self) -> str:
        return f"<Ruta id={self.id} {self.origen}→{self.destino} {self.distancia_km}km>"


class DimClima(Base):
    __tablename__ = "dim_clima"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    condicion: Mapped[str] = mapped_column(String(30), nullable=False, unique=True)
    temp_promedio: Mapped[float] = mapped_column(Float, nullable=False)

    entregas: Mapped[List["FactEntregas"]] = relationship(
        "FactEntregas", back_populates="clima"
    )

    def __repr__(self) -> str:
        return f"<Clima condicion={self.condicion} temp={self.temp_promedio}°C>"


class FactEntregas(Base):
    __tablename__ = "fact_entregas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    id_vehiculo: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_vehiculos.id"), nullable=False, index=True
    )
    id_conductor: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_conductores.id"), nullable=False, index=True
    )
    id_ruta: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_rutas.id"), nullable=False, index=True
    )
    id_clima: Mapped[int] = mapped_column(
        Integer, ForeignKey("dim_clima.id"), nullable=False, index=True
    )

    fecha_entrega: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        server_default=func.now(),
    )
    minutos_retraso: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    vehiculo: Mapped["DimVehiculos"] = relationship(
        "DimVehiculos", back_populates="entregas"
    )
    conductor: Mapped["DimConductores"] = relationship(
        "DimConductores", back_populates="entregas"
    )
    ruta: Mapped["DimRutas"] = relationship(
        "DimRutas", back_populates="entregas"
    )
    clima: Mapped["DimClima"] = relationship(
        "DimClima", back_populates="entregas"
    )

    def __repr__(self) -> str:
        return (
            f"<Entrega id={self.id} retraso={self.minutos_retraso:.1f}min "
            f"fecha={self.fecha_entrega}>"
        )
