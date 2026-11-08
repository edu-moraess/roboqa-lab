"""Immutable evaluation contracts for Phase 4.4-D.

These types describe evaluation evidence only. They do not fit, calibrate, or
score anomaly detectors and do not make Quality Gate decisions.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from roboqa.ml.contracts import DatasetSplit


class AnomalyEvaluationError(ValueError):
    """Raised when anomaly evaluation evidence violates its contract."""


class EvaluationScope(str, Enum):
    OVERALL = "overall"
    RUN = "run"
    SCENARIO = "scenario"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnomalyEvaluationError(f"{name} must be a non-empty string")
    return value.strip()


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnomalyEvaluationError(f"{name} must be a non-negative integer")
    return value


def _ratio(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise AnomalyEvaluationError(f"{name} must be finite and in [0, 1] or None")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnomalyEvaluationError(
            f"{name} must be finite and in [0, 1] or None"
        ) from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise AnomalyEvaluationError(f"{name} must be finite and in [0, 1] or None")
    return result


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise AnomalyEvaluationError(
            f"{name} must be a 64-character SHA-256 hex digest"
        )
    digest = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise AnomalyEvaluationError(
            f"{name} must be a 64-character SHA-256 hex digest"
        )
    return digest


def _utc_timestamp(value: str) -> str:
    timestamp = _non_empty(value, "generated_at")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AnomalyEvaluationError(
            "generated_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AnomalyEvaluationError("generated_at must include UTC timezone")
    return timestamp


@dataclass(frozen=True)
class AnomalyConfusionMatrix:
    """Binary confusion counts over labeled, successfully scored samples."""

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    def __post_init__(self) -> None:
        for name in (
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.true_negative
            + self.false_negative
        )


@dataclass(frozen=True)
class AnomalyClassificationMetrics:
    """Optional classification ratios; undefined denominators remain None."""

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    specificity: float | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
    balanced_accuracy: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "precision",
            "recall",
            "f1",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "balanced_accuracy",
        ):
            object.__setattr__(self, name, _ratio(getattr(self, name), name))


@dataclass(frozen=True)
class AnomalyEvaluationCounts:
    """Explicit accounting for labeled, unknown, scored, and unscorable rows."""

    total_samples: int
    labeled_samples: int
    unknown_samples: int
    scored_labeled_samples: int
    unscorable_labeled_samples: int
    scored_unknown_samples: int
    unscorable_unknown_samples: int

    def __post_init__(self) -> None:
        for name in (
            "total_samples",
            "labeled_samples",
            "unknown_samples",
            "scored_labeled_samples",
            "unscorable_labeled_samples",
            "scored_unknown_samples",
            "unscorable_unknown_samples",
        ):
            object.__setattr__(self, name, _count(getattr(self, name), name))
        if self.total_samples != self.labeled_samples + self.unknown_samples:
            raise AnomalyEvaluationError(
                "total_samples must equal labeled_samples + unknown_samples"
            )
        if self.labeled_samples != (
            self.scored_labeled_samples + self.unscorable_labeled_samples
        ):
            raise AnomalyEvaluationError(
                "labeled_samples must equal scored + unscorable labeled samples"
            )
        if self.unknown_samples != (
            self.scored_unknown_samples + self.unscorable_unknown_samples
        ):
            raise AnomalyEvaluationError(
                "unknown_samples must equal scored + unscorable unknown samples"
            )


@dataclass(frozen=True)
class AnomalyEvaluationSlice:
    """Evaluation summary for the complete split, one run, or one scenario."""

    scope: EvaluationScope
    scope_id: str | None
    counts: AnomalyEvaluationCounts
    confusion_matrix: AnomalyConfusionMatrix
    metrics: AnomalyClassificationMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.scope, EvaluationScope):
            raise AnomalyEvaluationError("scope must be an EvaluationScope")
        if self.scope is EvaluationScope.OVERALL:
            if self.scope_id is not None:
                raise AnomalyEvaluationError("overall scope_id must be None")
        else:
            object.__setattr__(
                self, "scope_id", _non_empty(self.scope_id, "scope_id")
            )
        if not isinstance(self.counts, AnomalyEvaluationCounts):
            raise AnomalyEvaluationError("counts must be AnomalyEvaluationCounts")
        if not isinstance(self.confusion_matrix, AnomalyConfusionMatrix):
            raise AnomalyEvaluationError(
                "confusion_matrix must be AnomalyConfusionMatrix"
            )
        if not isinstance(self.metrics, AnomalyClassificationMetrics):
            raise AnomalyEvaluationError(
                "metrics must be AnomalyClassificationMetrics"
            )
        if self.counts.scored_labeled_samples != self.confusion_matrix.total:
            raise AnomalyEvaluationError(
                "confusion matrix total must equal scored_labeled_samples"
            )


@dataclass(frozen=True)
class AnomalyEvaluationReport:
    """Versioned, provenance-bound evaluation report; never a quality decision."""

    dataset_id: str
    evaluation_split: DatasetSplit
    method: str
    generated_at: str
    dataset_hash: str
    prediction_hash: str
    slices: tuple[AnomalyEvaluationSlice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _non_empty(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "method", _non_empty(self.method, "method"))
        if not isinstance(self.evaluation_split, DatasetSplit) or self.evaluation_split not in (
            DatasetSplit.VALIDATION,
            DatasetSplit.TEST,
        ):
            raise AnomalyEvaluationError(
                "evaluation_split must be VALIDATION or TEST; TRAIN is prohibited"
            )
        object.__setattr__(self, "generated_at", _utc_timestamp(self.generated_at))
        object.__setattr__(self, "dataset_hash", _sha256(self.dataset_hash, "dataset_hash"))
        object.__setattr__(
            self, "prediction_hash", _sha256(self.prediction_hash, "prediction_hash")
        )
        try:
            slices = tuple(self.slices)
        except TypeError as exc:
            raise AnomalyEvaluationError("slices must be iterable") from exc
        if not slices:
            raise AnomalyEvaluationError(
                "report must contain exactly one overall evaluation slice"
            )
        if any(not isinstance(item, AnomalyEvaluationSlice) for item in slices):
            raise AnomalyEvaluationError(
                "slices must contain AnomalyEvaluationSlice values"
            )
        overall = [item for item in slices if item.scope is EvaluationScope.OVERALL]
        if len(overall) != 1:
            raise AnomalyEvaluationError(
                "report must contain exactly one overall evaluation slice"
            )
        identities = [(item.scope, item.scope_id) for item in slices]
        if len(set(identities)) != len(identities):
            raise AnomalyEvaluationError("evaluation slice identities must be unique")
        object.__setattr__(self, "slices", slices)
