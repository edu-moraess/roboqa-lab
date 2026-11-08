"""Fault Injection engine and immutable contracts — Phase 3.2."""
from roboqa.fault_injection.contracts import (
    ContractError, FaultEvent, FaultOperator, FaultResult, FaultScenario,
    FaultStatus, FaultType, RecoveryRecord, RecoveryStatus,
)
from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.fault_injection.provenance import (
    FaultProvenance, build_configuration_hash, canonical_json, sha256_digest,
    validate_sha256, validate_utc_timestamp,
)
from roboqa.fault_injection.operators import (
    ArrivalShiftOperator, BiasOperator, DelayOperator, DropoutOperator,
    DuplicateOperator, EnvelopeFaultOperator, InvalidPayloadOperator,
    OutlierOperator, PacketLossOperator,
)
from roboqa.fault_injection.engine import FaultExecution, FaultInjectionEngine

__all__ = [
    "ContractError", "FaultEvent", "FaultOperator", "FaultResult", "FaultScenario",
    "FaultStatus", "FaultType", "RecoveryRecord", "RecoveryStatus",
    "RawMeasurementEnvelope", "FaultProvenance", "build_configuration_hash",
    "canonical_json", "sha256_digest", "validate_sha256", "validate_utc_timestamp",
    "EnvelopeFaultOperator", "DropoutOperator", "PacketLossOperator", "DelayOperator",
    "ArrivalShiftOperator", "BiasOperator", "OutlierOperator", "InvalidPayloadOperator",
    "DuplicateOperator", "FaultExecution", "FaultInjectionEngine",
]
