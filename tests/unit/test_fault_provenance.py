import pytest

from roboqa.fault_injection import (
    ContractError,
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
