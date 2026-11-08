"""Fault Injection engine and immutable contracts — Phase 3.2."""
from roboqa.fault_injection.contracts import (
    ContractError, FaultEvent, FaultOperator, FaultResult, FaultScenario,
    FaultStatus, FaultType, RecoveryRecord, RecoveryStatus,
)
from roboqa.fault_injection.engine import FaultExecution, FaultInjectionEngine
from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fault_injection.operators import (
    ArrivalShiftOperator, BiasOperator, DelayOperator, DropoutOperator,
    DuplicateOperator, EnvelopeFaultOperator, InvalidPayloadOperator,
    OutlierOperator, PacketLossOperator,
)
from roboqa.fault_injection.provenance import (
    FaultProvenance, build_configuration_hash, canonical_json, sha256_digest,
    validate_sha256, validate_utc_timestamp,
)

__all__ = [
    "ContractError", "FaultEvent", "FaultOperator", "FaultResult", "FaultScenario",
    "FaultStatus", "FaultType", "RecoveryRecord", "RecoveryStatus",
    "FaultExecution", "FaultInjectionEngine", "RawMeasurementEnvelope",
    "ArrivalShiftOperator", "BiasOperator", "DelayOperator", "DropoutOperator",
    "DuplicateOperator", "EnvelopeFaultOperator", "InvalidPayloadOperator",
    "OutlierOperator", "PacketLossOperator", "FaultProvenance",
    "build_configuration_hash", "canonical_json", "sha256_digest",
    "validate_sha256", "validate_utc_timestamp",
]
