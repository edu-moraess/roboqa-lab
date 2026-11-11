"""Explicit, auditable association between measurements and reference samples.

No array-position or nearest-time association is performed unless the caller
declares matching sequence identities, comparable clock domains/time bases,
and an explicit tolerance. Ambiguous and unmatched observations stay unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np

from .schemas import GroundTruth, Measurement, SchemaError

AssociationStatus = Literal["matched", "unmatched", "ambiguous"]


@dataclass(frozen=True)
class MeasurementGroundTruthAssociation:
    measurement_index: int
    ground_truth_index: int | None
    measurement_sequence_id: str
    ground_truth_sequence_id: str
    status: AssociationStatus
    method: str
    delta_t_s: float | None
    tolerance_s: float
    evidence: str

    def __post_init__(self) -> None:
        if self.measurement_index < 0:
            raise SchemaError("measurement_index must be >= 0.")
        if self.ground_truth_index is not None and self.ground_truth_index < 0:
            raise SchemaError("ground_truth_index must be >= 0 or None.")
        if self.status == "matched" and self.ground_truth_index is None:
            raise SchemaError("matched association requires ground_truth_index.")
        if self.status != "matched" and self.ground_truth_index is not None:
            raise SchemaError("unresolved association cannot reference a ground-truth index.")
        if not self.measurement_sequence_id.strip() or not self.ground_truth_sequence_id.strip():
            raise SchemaError("Both sequence identifiers must be explicit.")
        if not np.isfinite(self.tolerance_s) or self.tolerance_s < 0:
            raise SchemaError("tolerance_s must be finite and >= 0.")
        if self.delta_t_s is not None and not np.isfinite(self.delta_t_s):
            raise SchemaError("delta_t_s must be finite or None.")
        if not self.evidence.strip():
            raise SchemaError("Association evidence must be explicit.")


def associate_by_timestamp(
    measurements: Sequence[Measurement],
    ground_truth: GroundTruth,
    *,
    measurement_sequence_id: str,
    ground_truth_sequence_id: str,
    measurement_clock_domain: str,
    ground_truth_clock_domain: str,
    measurement_time_basis: str,
    ground_truth_time_basis: str,
    tolerance_s: float,
) -> list[MeasurementGroundTruthAssociation]:
    """Associate only unique timestamp candidates under declared comparable clocks.

    Sequence identifiers may differ (for example, a derived measurement stream
    and its source reference), but both must be explicitly supplied. Clock
    domain and time basis must match exactly. Each reference sample can be used
    at most once; collisions are marked ambiguous rather than silently reused.
    """
    declarations = {
        "measurement_sequence_id": measurement_sequence_id,
        "ground_truth_sequence_id": ground_truth_sequence_id,
        "measurement_clock_domain": measurement_clock_domain,
        "ground_truth_clock_domain": ground_truth_clock_domain,
        "measurement_time_basis": measurement_time_basis,
        "ground_truth_time_basis": ground_truth_time_basis,
    }
    for name, value in declarations.items():
        if not isinstance(value, str) or not value.strip():
            raise SchemaError(f"{name} must be an explicit non-empty string.")
    if measurement_clock_domain != ground_truth_clock_domain:
        raise SchemaError("Clock domains differ; timestamp association is not permitted.")
    if measurement_time_basis != ground_truth_time_basis:
        raise SchemaError("Time bases differ; timestamp association is not permitted.")
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise SchemaError("tolerance_s must be finite and >= 0.")

    candidates: list[list[tuple[int, float]]] = []
    for measurement in measurements:
        deltas = np.abs(ground_truth.t - float(measurement.t))
        indexes = np.flatnonzero(deltas <= tolerance_s)
        candidates.append([(int(i), float(ground_truth.t[i] - measurement.t)) for i in indexes])

    # Enforce one-to-one matching. If multiple measurements claim the same
    # reference sample, all claimants remain unresolved.
    claimants: dict[tuple[str, int], list[int]] = {}
    for measurement_index, matches in enumerate(candidates):
        if len(matches) == 1:
            key = (measurements[measurement_index].sensor, matches[0][0])
            claimants.setdefault(key, []).append(measurement_index)

    results: list[MeasurementGroundTruthAssociation] = []
    for measurement_index, matches in enumerate(candidates):
        if len(matches) == 0:
            results.append(MeasurementGroundTruthAssociation(
                measurement_index, None, measurement_sequence_id, ground_truth_sequence_id,
                "unmatched", "timestamp_nearest_unique", None, float(tolerance_s),
                "No ground-truth timestamp within declared tolerance.",
            ))
        elif len(matches) > 1:
            results.append(MeasurementGroundTruthAssociation(
                measurement_index, None, measurement_sequence_id, ground_truth_sequence_id,
                "ambiguous", "timestamp_nearest_unique", None, float(tolerance_s),
                "Multiple ground-truth samples fall within declared tolerance.",
            ))
        else:
            gt_index, delta = matches[0]
            claim_key = (measurements[measurement_index].sensor, gt_index)
            if len(claimants.get(claim_key, [])) > 1:
                results.append(MeasurementGroundTruthAssociation(
                    measurement_index, None, measurement_sequence_id, ground_truth_sequence_id,
                    "ambiguous", "timestamp_nearest_unique", delta, float(tolerance_s),
                    "Multiple measurements claim the same ground-truth sample.",
                ))
            else:
                results.append(MeasurementGroundTruthAssociation(
                    measurement_index, gt_index, measurement_sequence_id, ground_truth_sequence_id,
                    "matched", "timestamp_nearest_unique", delta, float(tolerance_s),
                    "Unique timestamp candidate under matching declared clock and time basis.",
                ))
    return results
