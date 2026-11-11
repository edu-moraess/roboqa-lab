"""
Carregador de trajetórias KITTI OXTS.
Fase 1 — Data Engine (hardened)
ARQTECH LABS

Política de proveniência e fallback:
- load_kitti_oxts() levanta SchemaError em qualquer falha.
- get_trajectory(prefer_kitti=True) NÃO faz fallback silencioso.
  Se o caminho for inválido ou a leitura falhar, a exceção propaga.
- Fallback sintético ocorre somente quando prefer_kitti=False
  ou quando kitti_path é None.
- download_kitti_drive() é opt-in e, por política de segurança (R7),
  levanta NotImplementedError (sem download automático).

Temporalidade:
- Quando timestamps reais não estão disponíveis nos arquivos .txt
  do OXTS (formato clássico), usa-se aproximação de 10 Hz (dt=0.1 s).
- Esta aproximação é documentada e identificada.
- yaw_rate é calculado via np.gradient com np.unwrap para respeitar
  a descontinuidade angular.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .schemas import GroundTruth, SchemaError
from .synthetic import make_synthetic_ground_truth


def latlon_to_local(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converte latitude/longitude (graus) para coordenadas cartesianas locais (m).

    Utiliza aproximação equiretangular (adequada para trajetórias curtas,
    < ~10 km). Referência: centro da primeira amostra ou (lat0, lon0) fornecidos.

    Raises
    ------
    SchemaError
        Em entradas vazias, não-finitas ou dimensionalmente incompatíveis.
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    if lat.size == 0 or lon.size == 0:
        raise SchemaError("lat/lon não podem ser vazios.")
    if lat.shape != lon.shape:
        raise SchemaError(f"lat e lon devem ter mesmo shape: {lat.shape} vs {lon.shape}")
    if not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise SchemaError("lat/lon contêm valores não-finitos.")

    if lat0 is None:
        lat0 = float(lat.flat[0])
    if lon0 is None:
        lon0 = float(lon.flat[0])

    if not (np.isfinite(lat0) and np.isfinite(lon0)):
        raise SchemaError("lat0/lon0 devem ser finitos.")

    R = 6378137.0  # raio equatorial WGS84 (m)
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    x = R * dlon * np.cos(np.deg2rad(lat0))
    y = R * dlat
    return x.astype(np.float64), y.astype(np.float64)


def load_kitti_oxts(
    oxts_dir: Union[str, Path],
    max_frames: Optional[int] = None,
    dt_approx: float = 0.1,
) -> GroundTruth:
    """
    Carrega arquivos .txt do diretório oxts/data de uma sequência KITTI.

    Formato esperado (por linha):
    lat, lon, alt, roll, pitch, yaw, vn, ve, vf, vl, vu, ...

    Índices utilizados:
    - 0: lat (deg)
    - 1: lon (deg)
    - 5: yaw (rad)
    - 8: vf  (forward speed, m/s)

    Temporalidade:
    - Timestamps reais não estão presentes nos arquivos .txt clássicos.
    - Utiliza-se t = arange(N) * dt_approx (padrão 0.1 s = 10 Hz).
    - Esta é uma aproximação documentada; precisão típica do KITTI é ~10 Hz.

    Raises
    ------
    SchemaError
        Diretório inexistente, nenhum arquivo válido, poucos frames, etc.
    """
    oxts_dir = Path(oxts_dir)
    if not oxts_dir.exists():
        raise SchemaError(f"Diretório OXTS não encontrado: {oxts_dir}")
    if not oxts_dir.is_dir():
        raise SchemaError(f"Caminho não é diretório: {oxts_dir}")

    files = sorted(oxts_dir.glob("*.txt"))
    if not files:
        raise SchemaError(f"Nenhum arquivo .txt em {oxts_dir}")

    if max_frames is not None:
        if max_frames < 2:
            raise SchemaError("max_frames deve ser >= 2")
        files = files[:max_frames]

    rows = []
    for f in files:
        try:
            row = np.loadtxt(f)
        except Exception as e:
            raise SchemaError(f"Falha ao ler {f.name}: {e}") from e
        row = np.atleast_1d(row)
        if row.size < 9:
            raise SchemaError(
                f"Arquivo {f.name} incompleto (campos < 9). Conteúdo: {row}"
            )
        if not np.all(np.isfinite(row[:9])):
            raise SchemaError(f"Arquivo {f.name} contém valores não-finitos nos campos usados.")
        rows.append(row)

    if len(rows) < 2:
        raise SchemaError("Poucos frames válidos no OXTS (< 2).")

    arr = np.vstack(rows)

    lat = arr[:, 0]
    lon = arr[:, 1]
    yaw = arr[:, 5]
    speed = arr[:, 8]

    x, y = latlon_to_local(lat, lon)
    pos = np.stack([x, y], axis=1)

    # Timestamps aproximados (documentado)
    t = np.arange(len(arr), dtype=np.float64) * float(dt_approx)

    # yaw_rate respeitando unwrap
    yaw_unwrapped = np.unwrap(yaw)
    yaw_rate = np.gradient(yaw_unwrapped, dt_approx)

    # Preserve original WGS84 geodetic observations alongside the derived
    # local Cartesian trajectory. The local projection remains an approximation
    # and is not an ENU declaration or a validated GeoCore transform.
    geodetic_position = np.column_stack([lat, lon, arr[:, 2]])

    return GroundTruth(
        t=t,
        pos=pos,
        yaw=yaw_unwrapped,
        speed=speed.astype(np.float64),
        yaw_rate=yaw_rate,
        source="kitti_real",
        geodetic_position=geodetic_position,
        geodetic_datum="WGS84",
    )


def download_kitti_drive(
    drive: str = "2011_09_26_drive_0001",
    target_dir: Union[str, Path] = "data/kitti",
    force: bool = False,
) -> Path:
    """
    Download opt-in de uma sequência KITTI.

    Por política de segurança (R7) esta função NÃO realiza download automático.
    Levanta NotImplementedError com instruções claras.
    """
    raise NotImplementedError(
        "Download automático desabilitado por política de segurança (R7). "
        "Baixe manualmente a sequência desejada do site oficial do KITTI "
        "(http://www.cvlibs.net/datasets/kitti/raw_data.php) e aponte o "
        "caminho do diretório oxts/data para load_kitti_oxts(). "
        "Use make_synthetic_ground_truth() ou get_trajectory(prefer_kitti=False) "
        "como fonte sintética identificada."
    )


def get_trajectory(
    prefer_kitti: bool = False,
    kitti_path: Optional[Union[str, Path]] = None,
    n: int = 200,
    seed: int = 42,
    **synthetic_kwargs,
) -> GroundTruth:
    """
    Obtém trajetória.

    Comportamento:
    - prefer_kitti=False  → sempre sintético (source="synthetic").
    - prefer_kitti=True e kitti_path fornecido → tenta load_kitti_oxts.
      Em caso de falha, a exceção propaga (SEM fallback silencioso).
    - prefer_kitti=True e kitti_path=None → SchemaError explícito.
    """
    if prefer_kitti:
        if kitti_path is None:
            raise SchemaError(
                "prefer_kitti=True exige kitti_path válido. "
                "Não há fallback silencioso. "
                "Use prefer_kitti=False para trajetória sintética."
            )
        # Qualquer falha propaga
        return load_kitti_oxts(kitti_path)

    return make_synthetic_ground_truth(n=n, seed=seed, **synthetic_kwargs)
