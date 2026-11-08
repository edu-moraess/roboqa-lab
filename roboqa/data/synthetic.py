"""
Gerador de trajetórias sintéticas determinísticas.
Fase 1 — Data Engine (hardened)
ARQTECH LABS

Modelo matemático:
- Trajetória circular nominal de raio `radius` e velocidade linear `speed`.
- yaw(t) = (speed/radius)*t + yaw_rate_amp * sin(0.3*t)
- Posição obtida por integração de Euler de primeira ordem:
    x_{k+1} = x_k + speed * cos(yaw_k) * dt
    y_{k+1} = y_k + speed * sin(yaw_k) * dt
- speed e yaw_rate recebem ruído gaussiano pequeno (processo).
- A posição em si NÃO recebe ruído adicional de processo
  (o ruído está apenas em speed e yaw_rate).
- source é sempre "synthetic".
"""

from __future__ import annotations

import numpy as np

from .schemas import GroundTruth, SchemaError


def make_synthetic_ground_truth(
    n: int = 200,
    dt: float = 0.1,
    speed: float = 5.0,
    seed: int = 42,
    radius: float = 30.0,
    yaw_rate_amp: float = 0.15,
) -> GroundTruth:
    """
    Gera trajetória sintética do tipo círculo com variação angular controlada.

    Parameters
    ----------
    n : int
        Número de amostras (>= 2).
    dt : float
        Intervalo de tempo (s) (> 0).
    speed : float
        Velocidade linear nominal (m/s) (> 0).
    seed : int
        Seed para reprodutibilidade.
    radius : float
        Raio nominal da curva (m) (> 0).
    yaw_rate_amp : float
        Amplitude da oscilação adicional em yaw (rad).
        Controla a variação de yaw_rate além do valor nominal speed/radius.
        Deve ser >= 0.

    Returns
    -------
    GroundTruth
        Trajetória com source="synthetic".
    """
    # Validação de parâmetros
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise SchemaError(f"n deve ser inteiro >= 2, recebido {n}")
    if not np.isfinite(dt) or dt <= 0:
        raise SchemaError(f"dt deve ser > 0 e finito, recebido {dt}")
    if not np.isfinite(speed) or speed <= 0:
        raise SchemaError(f"speed deve ser > 0 e finito, recebido {speed}")
    if not np.isfinite(radius) or radius <= 0:
        raise SchemaError(f"radius deve ser > 0 e finito, recebido {radius}")
    if not np.isfinite(yaw_rate_amp) or yaw_rate_amp < 0:
        raise SchemaError(f"yaw_rate_amp deve ser >= 0 e finito, recebido {yaw_rate_amp}")
    if not isinstance(seed, (int, np.integer)):
        raise SchemaError(f"seed deve ser inteiro, recebido {type(seed)}")

    rng = np.random.default_rng(int(seed))

    t = np.arange(n, dtype=np.float64) * float(dt)

    # Orientação: contribuição circular + oscilação controlada
    omega0 = speed / radius
    yaw = omega0 * t + yaw_rate_amp * np.sin(0.3 * t)
    yaw = np.unwrap(yaw)

    # Integração de Euler (posição determinística a partir de yaw e speed nominal)
    dx = speed * np.cos(yaw) * dt
    dy = speed * np.sin(yaw) * dt
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    # Corrige o primeiro ponto para origem
    x = x - x[0]
    y = y - y[0]
    pos = np.stack([x, y], axis=1)

    # Ruído de processo apenas em speed e yaw_rate (não na posição)
    speed_arr = np.full(n, speed, dtype=np.float64) + rng.normal(0.0, 0.05, size=n)
    # yaw_rate derivado + ruído
    yaw_rate = np.gradient(yaw, dt) + rng.normal(0.0, 0.01, size=n)

    return GroundTruth(
        t=t,
        pos=pos,
        yaw=yaw,
        speed=speed_arr,
        yaw_rate=yaw_rate,
        source="synthetic",
    )
