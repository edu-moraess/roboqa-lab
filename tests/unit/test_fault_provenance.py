import pytest

from roboqa.fault_injection import (
    ContractError,
    FaultProvenance,
    build_configuration_hash,
    canonical_json,
    sha256_digest,
    validate_sha256,
    validate_utc_timestamp,
)


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    assert sha256_digest({"b": 2, "a": 1}) == sha256_digest({"a": 1, "b": 2})


def test_canonical_json_rejects_nonfinite_values():
    with pytest.raises(ContractError):
        canonical_json({"invalid": float("nan")})


def test_configuration_hash_is_sha256():
    digest = build_configuration_hash({"seed": 42, "mode": "test"})
    assert validate_sha256(digest, "configuration_hash") == digest


@pytest.mark.parametrize("value", ["", "xyz", "g" * 64, "a" * 63])
def test_invalid_sha256_is_rejected(value):
    with pytest.raises(ContractError):
        validate_sha256(value, "source_hash")


def test_timestamp_requires_timezone():
    assert validate_utc_timestamp("2026-11-08T02:00:00Z")
    with pytest.raises(ContractError):
        validate_utc_timestamp("2026-11-08T02:00:00")


def test_fault_provenance_validates_hashes_seed_and_timezone():
    digest = "a" * 64
    record = FaultProvenance(
        run_id="run-1",
        scenario_id="scenario-1",
        source_hash=digest,
        output_hash="b" * 64,
        operator_version="delay@1.0",
        configuration_hash="c" * 64,
        seed=42,
        created_at="2026-11-08T02:00:00Z",
    )
    assert record.seed == 42
    with pytest.raises(ContractError):
        FaultProvenance(
            run_id="run-1", scenario_id="scenario-1", source_hash="bad",
            output_hash=digest, operator_version="v1", configuration_hash=digest,
            seed=42, created_at="2026-11-08T02:00:00Z",
        )
    with pytest.raises(ContractError):
        FaultProvenance(
            run_id="run-1", scenario_id="scenario-1", source_hash=digest,
            output_hash=digest, operator_version="v1", configuration_hash=digest,
            seed=42, created_at="2026-11-08T02:00:00",
        )

    with pytest.raises(ContractError):
        validate_utc_timestamp("2026-11-08T05:00:00+03:00")
