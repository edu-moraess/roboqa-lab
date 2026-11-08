"""RoboQA quality metric calculation and report contracts."""

from roboqa.qa.contracts import (
    FindingSeverity,
    MetricAvailability,
    MetricObservation,
    MetricScope,
    QualityContractError,
    QualityDecision,
    QualityFinding,
    QualityReport,
)
from roboqa.qa.gate import (
    MetricRule,
    QualityGateError,
    QualityGatePolicy,
    evaluate_quality_gate,
)
from roboqa.qa.metrics import QualityMetricError, compute_quality_metrics
from roboqa.qa.integrated import (
    IntegratedQualityError,
    compute_integrated_quality_metrics,
    evaluate_integrated_quality,
)

__all__ = [
    "FindingSeverity",
    "MetricAvailability",
    "MetricObservation",
    "MetricScope",
    "QualityContractError",
    "QualityDecision",
    "QualityFinding",
    "QualityReport",
    "QualityMetricError",
    "compute_quality_metrics",
    "IntegratedQualityError",
    "compute_integrated_quality_metrics",
    "evaluate_integrated_quality",
    "MetricRule",
    "QualityGateError",
    "QualityGatePolicy",
    "evaluate_quality_gate",
]
