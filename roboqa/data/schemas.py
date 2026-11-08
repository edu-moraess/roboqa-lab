"""
RoboQA Data Schemas — Contratos fundamentais de dados.
Fase 1 — Data Engine (hardened)
ARQTECH LABS / RoboQA Lab

Decisões de validação:
- Arrays internos são copiados na construção (imutabilidade efetiva).
- Timestamps devem ser finitos, monotônicos não-decrescentes.
- Duplicatas de timestamp são rejeitadas (tolerância 1e-12 s).
- NaN / Inf são rejeitados em todos os campos numéricos.
- Covariância R deve ser simétrica e semidefinida positiva (eig >= -1e-8).
- source é enumerado estritamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class SchemaError(Exception):
    """Erro de validação de schema de dados do RoboQA."""
    pass


_ALLOWED_SOURCES = frozenset({"kitti_real", "synthetic", "derived"})


def _as_1d_float64(arr: np.ndarray, name: str) -> np.ndarray:
    """Converte e valida array 1D float64, rejeitando não-finitos."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 1:
        raise SchemaError(f"{name} deve ser 1D, recebido shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise SchemaError(f"{name} contém NaN ou Inf")
    return a.copy()  # cópia defensiva


def _as_2d_float64(arr: np.ndarray, name: str, expected_cols: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise SchemaError(f"{name} deve ser 2D, recebido shape {a.shape}")
    if expected_cols is not None and a.shape[1] != expected_cols:
        raise SchemaError(f"{name} deve ter {expected_cols} colunas, recebido {a.shape[1]}")
    if not np.all(np.isfinite(a)):
        raise SchemaError(f"{name} contém NaN ou Inf")
    return a.copy()


@dataclass(frozen=True)
class GroundTruth:
    """
    Trajetória de referência (ground truth).

    Attributes
    ----------
    t : np.ndarray, shape (N,)
        Timestamps (s). Monotônicos, sem duplicatas, finitos.
    pos : np.ndarray, shape (N, 2)
        Posição [x, y] (m).
    yaw : np.ndarray, shape (N,)
        Orientação (rad).
    speed : np.ndarray, shape (N,)
        Velocidade linear (m/s).
    yaw_rate : np.ndarray, shape (N,)
        Taxa de guinada (rad/s).
    source : str
        Proveniência: "kitti_real" | "synthetic" | "derived".
    """
    t: np.ndarray
    pos: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    yaw_rate: np.ndarray
    source: str

    def __post_init__(self) -> None:
        # Validação e normalização (cópias defensivas)
        object.__setattr__(self, "t", _as_1d_float64(self.t, "t"))
        object.__setattr__(self, "pos", _as_2d_float64(self.pos, "pos", expected_cols=2))
        object.__setattr__(self, "yaw", _as_1d_float64(self.yaw, "yaw"))
        object.__setattr__(self, "speed", _as_1d_float64(self.speed, "speed"))
        object.__setattr__(self, "yaw_rate", _as_1d_float64(self.yaw_rate, "yaw_rate"))

        n = self.t.shape[0]
        if n == 0:
            raise SchemaError("GroundTruth vazio (N=0).")

        if self.pos.shape[0] != n:
            raise SchemaError(f"pos shape {self.pos.shape} incompatível com t (N={n})")
        for name, arr in [("yaw", self.yaw), ("speed", self.speed), ("yaw_rate", self.yaw_rate)]:
            if arr.shape[0] != n:
                raise SchemaError(f"{name} shape {arr.shape} incompatível com t (N={n})")

        # Timestamps: monotônicos e sem duplicatas
        dt = np.diff(self.t)
        if np.any(dt < 0):
            raise SchemaError("Timestamps regressivos detectados.")
        if np.any(dt < 1e-12):
            raise SchemaError("Timestamps duplicados (ou quase) detectados.")

        if self.source not in _ALLOWED_SOURCES:
            raise SchemaError(
                f"source inválido: '{self.source}'. "
                f"Permitidos: {sorted(_ALLOWED_SOURCES)}"
            )

    @property
    def n(self) -> int:
        return int(self.t.shape[0])

    def copy(self) -> "GroundTruth":
        """Retorna cópia profunda (arrays novos)."""
        return GroundTruth(
            t=self.t.copy(),
            pos=self.pos.copy(),
            yaw=self.yaw.copy(),
            speed=self.speed.copy(),
            yaw_rate=self.yaw_rate.copy(),
            source=self.source,
        )


@dataclass(frozen=True)
class Measurement:
    """
    Medição de um sensor.

    Attributes
    ----------
    t : float
        Timestamp (s), finito.
    sensor : str
        Identificador não-vazio do sensor.
    z : np.ndarray, shape (M,)
        Vetor de medição.
    R : np.ndarray, shape (M, M)
        Matriz de covariância (simétrica, PSD).
    """
    t: float
    sensor: str
    z: np.ndarray
    R: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.t):
            raise SchemaError(f"Timestamp não-finito: {self.t}")

        if not isinstance(self.sensor, str) or not self.sensor.strip():
            raise SchemaError("sensor deve ser string não-vazia.")

        z = np.asarray(self.z, dtype=np.float64)
        if z.ndim != 1:
            raise SchemaError(f"z deve ser 1D, recebido shape {z.shape}")
        if z.size == 0:
            raise SchemaError("z não pode ser vazio.")
        if not np.all(np.isfinite(z)):
            raise SchemaError("z contém NaN ou Inf.")

        R = np.asarray(self.R, dtype=np.float64)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise SchemaError(f"R deve ser matriz quadrada, recebido {R.shape}")
        if R.shape[0] != z.shape[0]:
            raise SchemaError(
                f"Dimensão de R ({R.shape[0]}) incompatível com z ({z.shape[0]})"
            )
        if not np.all(np.isfinite(R)):
            raise SchemaError("R contém NaN ou Inf.")
        if not np.allclose(R, R.T, atol=1e-8):
            raise SchemaError("R não é simétrica (atol=1e-8).")

        # Semidefinição positiva
        eig = np.linalg.eigvalsh((R + R.T) / 2.0)  # força simetria numérica
        if np.any(eig < -1e-8):
            raise SchemaError(
                f"R não é semidefinida positiva (min eig={float(eig.min()):.3e})"
            )

        # Armazena cópias
        object.__setattr__(self, "z", z.copy())
        object.__setattr__(self, "R", R.copy())
        object.__setattr__(self, "sensor", self.sensor.strip())


def validate_measurement_sequence(ms: Sequence[Measurement]) -> None:
    """
    Valida sequência de medições.

    - Aceita sequência vazia.
    - Exige ordem temporal não-decrescente (tolerância 1e-9 s).
    - Rejeita timestamps não-finitos (já garantido pelo construtor).
    """
    if not ms:
        return
    times = np.array([m.t for m in ms], dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise SchemaError("Sequência contém timestamps não-finitos.")
    dt = np.diff(times)
    if np.any(dt < -1e-9):
        raise SchemaError("Medições não estão em ordem temporal não-decrescente.")
