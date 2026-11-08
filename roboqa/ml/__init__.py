"""Advisory ML contracts and tooling for RoboQA Lab."""

from roboqa.ml.anomaly import (
    AnomalyDecision,
    AnomalyDetectionError,
    AnomalyPrediction,
    IsolationForestBaseline,
    StatisticalAnomalyBaseline,
    fit_isolation_forest,
    fit_statistical_baseline,
    score_isolation_forest,
    score_statistical_baseline,
)
from roboqa.ml.contracts import (
    AnomalyReference,
    DatasetSplit,
    FeatureAvailability,
    FeatureDataset,
    FeatureDatasetError,
    FeatureSample,
    FeatureValue,
)
from roboqa.ml.evaluation import (
    AnomalyClassificationMetrics,
    AnomalyConfusionMatrix,
    AnomalyEvaluationCounts,
    AnomalyEvaluationError,
    AnomalyEvaluationReport,
    AnomalyEvaluationSlice,
    EvaluationScope,
)
from roboqa.ml.evaluation_engine import evaluate_anomaly_predictions
from roboqa.ml.features import FeatureEngineeringError, extract_temporal_features

__all__ = [
    "AnomalyDecision",
    "AnomalyClassificationMetrics",
    "AnomalyConfusionMatrix",
    "AnomalyEvaluationCounts",
    "AnomalyEvaluationError",
    "AnomalyEvaluationReport",
    "AnomalyEvaluationSlice",
    "EvaluationScope",
    "AnomalyDetectionError",
    "AnomalyPrediction",
    "AnomalyReference",
    "DatasetSplit",
    "FeatureAvailability",
    "FeatureDataset",
    "FeatureDatasetError",
    "FeatureEngineeringError",
    "FeatureSample",
    "FeatureValue",
    "IsolationForestBaseline",
    "StatisticalAnomalyBaseline",
    "fit_isolation_forest",
    "fit_statistical_baseline",
    "score_isolation_forest",
    "score_statistical_baseline",
    "extract_temporal_features",
    "evaluate_anomaly_predictions",
]
