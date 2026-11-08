"""Leakage-aware advisory anomaly baselines for Phase 4.4-C.

Models are fitted only on TRAIN rows. Reference labels are never read. The
statistical baseline uses robust univariate scaling and a max-|z| score;
Isolation Forest is an optional scikit-learn-backed advisory detector.
Neither detector changes Quality Gate decisions.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from roboqa.ml.contracts import (
    DatasetSplit,
    FeatureAvailability,
    FeatureDataset,
)


class AnomalyDetectionError(ValueError):
    """Raised when anomaly model training or scoring evidence is invalid."""


class AnomalyDecision(str, Enum):
    NORMAL = "normal"
    ANOMALOUS = "anomalous"
    UNSCORABLE = "unscorable"


@dataclass(frozen=True)
class AnomalyPrediction:
    """Advisory score for one sample; not a Quality Gate decision."""

    sample_id: str
    run_id: str
    split: DatasetSplit
    method: str
    score: float | None
    threshold: float | None
    decision: AnomalyDecision
    feature_names: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.run_id or not self.method:
            raise AnomalyDetectionError("prediction identities and method must be non-empty")
        if not isinstance(self.split, DatasetSplit):
            raise AnomalyDetectionError("prediction split must be a DatasetSplit")
        if not isinstance(self.decision, AnomalyDecision):
            raise AnomalyDetectionError("decision must be an AnomalyDecision")
        if self.decision is AnomalyDecision.UNSCORABLE:
            if self.score is not None:
                raise AnomalyDetectionError("unscorable predictions must not carry a score")
        elif self.score is None or not math.isfinite(float(self.score)):
            raise AnomalyDetectionError("scored predictions require a finite score")
        if self.threshold is not None and not math.isfinite(float(self.threshold)):
            raise AnomalyDetectionError("threshold must be finite or None")
        object.__setattr__(self, "feature_names", tuple(self.feature_names))


@dataclass(frozen=True)
class _TrainingTransform:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    feature_units: tuple[str, ...]
    training_sample_ids: tuple[str, ...]
    training_run_ids: tuple[str, ...]
    training_data_hash: str


@dataclass(frozen=True)
class StatisticalAnomalyBaseline:
    """Fitted robust statistical baseline; immutable training parameters."""

    transform: _TrainingTransform
    contamination: float
    threshold: float
    model_hash: str


@dataclass(frozen=True)
class IsolationForestBaseline:
    """Fitted optional Isolation Forest with a training-only transform."""

    transform: _TrainingTransform
    contamination: float
    random_state: int
    n_estimators: int
    library_version: str
    threshold: float
    model_hash: str
    estimator: Any


def _validate_contamination(value: float) -> float:
    if isinstance(value, bool):
        raise AnomalyDetectionError("contamination must be finite and in (0, 0.5]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnomalyDetectionError("contamination must be finite and in (0, 0.5]") from exc
    if not math.isfinite(result) or not 0.0 < result <= 0.5:
        raise AnomalyDetectionError("contamination must be finite and in (0, 0.5]")
    return result


def _sample_values(sample: object) -> dict[str, object]:
    return {feature.name: feature for feature in sample.features}


def _training_matrix(dataset: FeatureDataset) -> tuple[_TrainingTransform, np.ndarray]:
    if not isinstance(dataset, FeatureDataset):
        raise AnomalyDetectionError("dataset must be a FeatureDataset")
    training = tuple(sample for sample in dataset.samples if sample.split is DatasetSplit.TRAIN)
    if len(training) < 3:
        raise AnomalyDetectionError("at least three TRAIN samples are required")
    names = tuple(sorted(feature.name for feature in training[0].features))
    maps = [_sample_values(sample) for sample in training]
    eligible = tuple(
        name for name in names
        if all(
            mapping[name].availability is FeatureAvailability.AVAILABLE
            and mapping[name].value is not None
            for mapping in maps
        )
    )
    if not eligible:
        raise AnomalyDetectionError(
            "no eligible features: each candidate must be AVAILABLE in every TRAIN sample"
        )
    matrix = np.asarray(
        [[float(mapping[name].value) for name in eligible] for mapping in maps],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all():
        raise AnomalyDetectionError("TRAIN feature matrix contains non-finite values")

    medians = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - medians), axis=0)
    scales = 1.4826 * mad
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    iqr_scales = (q75 - q25) / 1.349
    std_scales = np.std(matrix, axis=0, ddof=0)
    scales = np.where(scales > 0.0, scales, np.where(
        iqr_scales > 0.0, iqr_scales, std_scales
    ))
    variable = scales > 0.0
    if not np.any(variable):
        raise AnomalyDetectionError("TRAIN features are constant; no model can be fitted")
    eligible = tuple(name for name, keep in zip(eligible, variable) if keep)
    medians = medians[variable]
    scales = scales[variable]
    matrix = matrix[:, variable]
    standardized = (matrix - medians) / scales

    feature_units = tuple(maps[0][name].unit for name in eligible)
    train_document = [
        {
            "sample_id": sample.sample_id,
            "run_id": sample.run_id,
            "features": [
                [name, unit, float(value)]
                for name, unit, value in zip(eligible, feature_units, row)
            ],
        }
        for sample, row in zip(training, matrix)
    ]
    train_hash = hashlib.sha256(json.dumps(
        train_document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    transform = _TrainingTransform(
        feature_names=eligible,
        medians=tuple(float(value) for value in medians),
        scales=tuple(float(value) for value in scales),
        feature_units=feature_units,
        training_sample_ids=tuple(sample.sample_id for sample in training),
        training_run_ids=tuple(sorted({sample.run_id for sample in training})),
        training_data_hash=train_hash,
    )
    return transform, standardized


def _matrix_for_scoring(
    dataset: FeatureDataset, transform: _TrainingTransform
) -> tuple[list[object], np.ndarray, list[str]]:
    if not isinstance(dataset, FeatureDataset):
        raise AnomalyDetectionError("dataset must be a FeatureDataset")
    rows = []
    scorable = []
    reasons = []
    medians = np.asarray(transform.medians, dtype=np.float64)
    scales = np.asarray(transform.scales, dtype=np.float64)
    for sample in dataset.samples:
        mapping = _sample_values(sample)
        missing = [
            name for name in transform.feature_names
            if name not in mapping
            or mapping[name].availability is not FeatureAvailability.AVAILABLE
            or mapping[name].value is None
        ]
        unit_mismatches = [
            name for name, unit in zip(transform.feature_names, transform.feature_units)
            if name in mapping and mapping[name].unit != unit
        ]
        if missing or unit_mismatches:
            scorable.append(False)
            details = []
            if missing:
                details.append("required training feature unavailable: " + ", ".join(missing))
            if unit_mismatches:
                details.append("feature unit mismatch: " + ", ".join(unit_mismatches))
            reasons.append("; ".join(details))
            rows.append(np.zeros(len(transform.feature_names), dtype=np.float64))
            continue
        values = np.asarray(
            [float(mapping[name].value) for name in transform.feature_names],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            scorable.append(False)
            reasons.append("required training feature is non-finite")
            rows.append(np.zeros(len(transform.feature_names), dtype=np.float64))
            continue
        scorable.append(True)
        reasons.append("")
        rows.append((values - medians) / scales)
    return list(dataset.samples), np.asarray(rows, dtype=np.float64), [
        reason if not ok else "" for ok, reason in zip(scorable, reasons)
    ]


def _model_hash(method: str, transform: _TrainingTransform, parameters: dict[str, object]) -> str:
    document = {
        "method": method,
        "training_data_hash": transform.training_data_hash,
        "feature_names": transform.feature_names,
        "medians": transform.medians,
        "scales": transform.scales,
        "feature_units": transform.feature_units,
        "parameters": parameters,
    }
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def fit_statistical_baseline(
    dataset: FeatureDataset, *, contamination: float = 0.05
) -> StatisticalAnomalyBaseline:
    """Fit a robust max-|z| baseline on TRAIN samples only."""
    fraction = _validate_contamination(contamination)
    transform, train_matrix = _training_matrix(dataset)
    training_scores = np.max(np.abs(train_matrix), axis=1)
    threshold = float(np.quantile(training_scores, 1.0 - fraction, method="higher"))
    model_hash = _model_hash(
        "robust_max_z_v1", transform, {"contamination": fraction, "threshold": threshold}
    )
    return StatisticalAnomalyBaseline(
        transform, fraction, threshold, model_hash
    )


def score_statistical_baseline(
    model: StatisticalAnomalyBaseline, dataset: FeatureDataset
) -> tuple[AnomalyPrediction, ...]:
    """Score rows with a fitted baseline; unavailable features remain UNSCORABLE."""
    if not isinstance(model, StatisticalAnomalyBaseline):
        raise AnomalyDetectionError("model must be a StatisticalAnomalyBaseline")
    samples, matrix, reasons = _matrix_for_scoring(dataset, model.transform)
    predictions = []
    for sample, row, reason in zip(samples, matrix, reasons):
        if reason:
            predictions.append(AnomalyPrediction(
                sample.sample_id, sample.run_id, sample.split, "robust_max_z_v1",
                None, model.threshold, AnomalyDecision.UNSCORABLE,
                model.transform.feature_names, reason,
            ))
            continue
        score = float(np.max(np.abs(row)))
        predictions.append(AnomalyPrediction(
            sample.sample_id, sample.run_id, sample.split, "robust_max_z_v1",
            score, model.threshold,
            AnomalyDecision.ANOMALOUS if score > model.threshold else AnomalyDecision.NORMAL,
            model.transform.feature_names,
        ))
    return tuple(predictions)


def fit_isolation_forest(
    dataset: FeatureDataset,
    *,
    contamination: float = 0.05,
    random_state: int = 42,
    n_estimators: int = 100,
) -> IsolationForestBaseline:
    """Fit optional scikit-learn Isolation Forest on TRAIN samples only."""
    fraction = _validate_contamination(contamination)
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or not 0 <= random_state <= 2**32 - 1
    ):
        raise AnomalyDetectionError("random_state must be an integer in [0, 2**32 - 1]")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
        raise AnomalyDetectionError("n_estimators must be a positive integer")
    try:
        import sklearn
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise AnomalyDetectionError(
            "Isolation Forest requires the optional 'ml' dependency: scikit-learn"
        ) from exc
    transform, train_matrix = _training_matrix(dataset)
    estimator = IsolationForest(
        n_estimators=n_estimators,
        contamination=fraction,
        random_state=random_state,
        n_jobs=1,
    )
    estimator.fit(train_matrix)
    threshold = float(-estimator.offset_)
    model_hash = _model_hash(
        "isolation_forest_v1",
        transform,
        {
            "contamination": fraction,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "library_version": sklearn.__version__,
            "threshold": threshold,
        },
    )
    return IsolationForestBaseline(
        transform, fraction, random_state, n_estimators, sklearn.__version__,
        threshold, model_hash, estimator
    )


def score_isolation_forest(
    model: IsolationForestBaseline, dataset: FeatureDataset
) -> tuple[AnomalyPrediction, ...]:
    """Score rows with a fitted Isolation Forest; missing inputs are not imputed."""
    if not isinstance(model, IsolationForestBaseline):
        raise AnomalyDetectionError("model must be an IsolationForestBaseline")
    samples, matrix, reasons = _matrix_for_scoring(dataset, model.transform)
    valid_indices = [index for index, reason in enumerate(reasons) if not reason]
    raw_scores: dict[int, float] = {}
    if valid_indices:
        values = -model.estimator.score_samples(matrix[valid_indices])
        raw_scores = {
            index: float(score) for index, score in zip(valid_indices, values)
        }
    predictions = []
    for index, (sample, reason) in enumerate(zip(samples, reasons)):
        if reason:
            predictions.append(AnomalyPrediction(
                sample.sample_id, sample.run_id, sample.split, "isolation_forest_v1",
                None, model.threshold, AnomalyDecision.UNSCORABLE,
                model.transform.feature_names, reason,
            ))
            continue
        score = raw_scores[index]
        predictions.append(AnomalyPrediction(
            sample.sample_id, sample.run_id, sample.split, "isolation_forest_v1",
            score, model.threshold,
            AnomalyDecision.ANOMALOUS if score > model.threshold else AnomalyDecision.NORMAL,
            model.transform.feature_names,
        ))
    return tuple(predictions)
