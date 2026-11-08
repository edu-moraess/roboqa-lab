"""Immutable, leakage-aware contracts for ML feature datasets (Phase 4.4-A).

These contracts do not train models, impute missing values, or make quality
decisions. Reference labels are evaluation metadata and must not be used as
model features.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class FeatureDatasetError(ValueError):
    """Raised when feature or dataset evidence violates its public contract."""


class FeatureAvailability(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    INVALID = "invalid"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class AnomalyReference(str, Enum):
    """Optional evaluation-only annotation; never a model input feature."""

    NORMAL = "normal"
    ANOMALOUS = "anomalous"
    UNKNOWN = "unknown"


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeatureDatasetError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise FeatureDatasetError(f"{name} must be finite or None")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureDatasetError(f"{name} must be finite or None") from exc
    if not math.isfinite(result):
        raise FeatureDatasetError(f"{name} must be finite or None")
    return result


def _utc_timestamp(value: str, name: str) -> str:
    timestamp = _non_empty(value, name)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FeatureDatasetError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise FeatureDatasetError(f"{name} must include UTC timezone")
    return timestamp


def _sha256(value: str, name: str) -> str:
    digest = _non_empty(value, name).lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise FeatureDatasetError(f"{name} must be a 64-character SHA-256 hex digest")
    return digest


@dataclass(frozen=True)
class FeatureValue:
    """One named scalar feature with explicit unit and availability."""

    name: str
    value: float | None
    unit: str
    availability: FeatureAvailability = FeatureAvailability.AVAILABLE
    sample_count: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "name"))
        object.__setattr__(self, "unit", _non_empty(self.unit, "unit"))
        if not isinstance(self.availability, FeatureAvailability):
            raise FeatureDatasetError("availability must be a FeatureAvailability")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise FeatureDatasetError("sample_count must be a non-negative integer")
        value = _finite_optional(self.value, "value")
        if self.availability is FeatureAvailability.AVAILABLE and value is None:
            raise FeatureDatasetError("available features require a finite value")
        if self.availability is not FeatureAvailability.AVAILABLE and value is not None:
            raise FeatureDatasetError("missing/invalid features must not carry a numeric value")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class FeatureSample:
    """One run/window feature row with an explicit dataset split."""

    sample_id: str
    run_id: str
    split: DatasetSplit
    features: tuple[FeatureValue, ...]
    scenario_id: str | None = None
    window_start: float | None = None
    window_end: float | None = None
    reference_label: AnomalyReference = AnomalyReference.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _non_empty(self.sample_id, "sample_id"))
        object.__setattr__(self, "run_id", _non_empty(self.run_id, "run_id"))
        if not isinstance(self.split, DatasetSplit):
            raise FeatureDatasetError("split must be a DatasetSplit")
        if self.scenario_id is not None:
            object.__setattr__(self, "scenario_id", _non_empty(self.scenario_id, "scenario_id"))
        if not isinstance(self.reference_label, AnomalyReference):
            raise FeatureDatasetError("reference_label must be an AnomalyReference")
        start = _finite_optional(self.window_start, "window_start")
        end = _finite_optional(self.window_end, "window_end")
        if (start is None) != (end is None):
            raise FeatureDatasetError("window_start and window_end must be supplied together")
        if start is not None and start >= end:
            raise FeatureDatasetError("window_start must be less than window_end")
        object.__setattr__(self, "window_start", start)
        object.__setattr__(self, "window_end", end)
        try:
            features = tuple(self.features)
        except TypeError as exc:
            raise FeatureDatasetError("features must be iterable") from exc
        if not features or any(not isinstance(item, FeatureValue) for item in features):
            raise FeatureDatasetError("features must contain at least one FeatureValue")
        names = [item.name for item in features]
        if len(set(names)) != len(names):
            raise FeatureDatasetError("feature names must be unique within a sample")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class FeatureDataset:
    """Versioned dataset manifest with split-leakage and schema validation.

    All rows from one run_id must belong to exactly one split. Feature names
    and units must be consistent across rows. Missing/invalid values remain
    explicit; this contract never imputes or drops them.
    """

    dataset_id: str
    schema_version: str
    created_at: str
    samples: tuple[FeatureSample, ...]
    source_hash: str
    configuration_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _non_empty(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "schema_version", _non_empty(self.schema_version, "schema_version"))
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "source_hash", _sha256(self.source_hash, "source_hash"))
        object.__setattr__(
            self, "configuration_hash", _sha256(self.configuration_hash, "configuration_hash")
        )
        try:
            samples = tuple(self.samples)
        except TypeError as exc:
            raise FeatureDatasetError("samples must be iterable") from exc
        if not samples or any(not isinstance(item, FeatureSample) for item in samples):
            raise FeatureDatasetError("samples must contain at least one FeatureSample")

        sample_ids = [item.sample_id for item in samples]
        if len(set(sample_ids)) != len(sample_ids):
            raise FeatureDatasetError("sample_id values must be unique")

        run_splits: dict[str, DatasetSplit] = {}
        feature_schema: dict[str, str] = {}
        expected_names: set[str] | None = None
        for sample in samples:
            previous_split = run_splits.setdefault(sample.run_id, sample.split)
            if previous_split is not sample.split:
                raise FeatureDatasetError(
                    f"run_id {sample.run_id!r} appears in multiple dataset splits"
                )
            sample_names = {feature.name for feature in sample.features}
            if expected_names is None:
                expected_names = sample_names
            elif sample_names != expected_names:
                raise FeatureDatasetError(
                    "all samples must contain the same feature names; use MISSING "
                    "or INVALID availability instead of omitting a feature"
                )
            for feature in sample.features:
                previous_unit = feature_schema.setdefault(feature.name, feature.unit)
                if previous_unit != feature.unit:
                    raise FeatureDatasetError(
                        f"feature {feature.name!r} has inconsistent units"
                    )
        object.__setattr__(self, "samples", samples)
