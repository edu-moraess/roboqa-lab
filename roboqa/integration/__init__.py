"""Fault-to-QA integration contracts, replay, adaptation, and scheduling."""
from roboqa.integration.contracts import (
    DeliveryDecision,
    DeliveryDisposition,
    EnvelopeIdentity,
    IntegrationContractError,
    TransportOrderingPolicy,
    TransportReplaySummary,
)
from roboqa.integration.measurement_adapter import (
    AdaptedMeasurement,
    MeasurementAdapterError,
    adapt_accepted_envelopes,
)
from roboqa.integration.replay import TransportReplayResult, replay_transport
from roboqa.integration.scheduler import (
    FusionSchedule,
    FusionSchedulingError,
    FusionSchedulingPolicy,
    build_fusion_schedule,
)
from roboqa.integration.execution import (
    ExecutionDisposition,
    FusionExecutionError,
    FusionExecutionRecord,
    FusionExecutionResult,
    MeasurementExecutionOutcome,
    execute_fusion_schedule,
)

__all__ = [
    "DeliveryDecision",
    "DeliveryDisposition",
    "EnvelopeIdentity",
    "IntegrationContractError",
    "TransportOrderingPolicy",
    "TransportReplaySummary",
    "TransportReplayResult",
    "replay_transport",
    "AdaptedMeasurement",
    "MeasurementAdapterError",
    "adapt_accepted_envelopes",
    "FusionSchedule",
    "FusionSchedulingError",
    "FusionSchedulingPolicy",
    "build_fusion_schedule",
    "ExecutionDisposition",
    "FusionExecutionError",
    "FusionExecutionRecord",
    "FusionExecutionResult",
    "MeasurementExecutionOutcome",
    "execute_fusion_schedule",
]
