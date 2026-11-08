"""Canonical serialization and SHA-256 provenance helpers."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

from dataclasses import dataclass

from roboqa.fault_injection.contracts import ContractError, _non_empty, thaw_json


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically and reject NaN/Inf."""
    normalized = thaw_json(value)
    try:
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical-JSON serializable: {exc}") from exc


def sha256_digest(value: Any) -> str:
    """Hash canonical JSON, never Python's process-dependent object hash."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def validate_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractError(f"{name} must be hexadecimal") from exc
    return value.lower()


def validate_utc_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("created_at must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("created_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError("created_at must include a timezone")
    return value


def build_configuration_hash(configuration: Mapping[str, Any]) -> str:
    if not isinstance(configuration, Mapping):
        raise ContractError("configuration must be a mapping")
    return sha256_digest(configuration)


@dataclass(frozen=True)
class FaultProvenance:
    """Immutable provenance record for one fault-injection execution."""

    run_id: str
    scenario_id: str
    source_hash: str
    output_hash: str
    operator_version: str
    configuration_hash: str
    seed: int
    created_at: str
    parent_run_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "scenario_id", "operator_version"):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        for name in ("source_hash", "output_hash", "configuration_hash"):
            object.__setattr__(self, name, validate_sha256(getattr(self, name), name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractError("seed must be an integer")
        if not 0 <= self.seed <= (2**64 - 1):
            raise ContractError("seed must be in [0, 2**64 - 1]")
        object.__setattr__(self, "created_at", validate_utc_timestamp(self.created_at))
        if self.parent_run_id is not None:
            object.__setattr__(self, "parent_run_id", _non_empty(self.parent_run_id, "parent_run_id"))
