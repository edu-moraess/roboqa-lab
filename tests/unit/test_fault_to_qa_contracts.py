"""Unit tests for Phase 4.5-A Fault-to-QA integration contracts."""
import pytest

from roboqa.integration.contracts import (
    DeliveryDecision,
    DeliveryDisposition,
    EnvelopeIdentity,
    IntegrationContractError,
    TransportOrderingPolicy,
    TransportReplaySummary,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def identity(sequence=1, sensor="GPS", source="device-1"):
    return EnvelopeIdentity(sensor_id=sensor, source_id=source, sequence_number=sequence)


def decision(disposition=DeliveryDisposition.ACCEPTED, sequence=1, **kwargs):
    defaults = dict(
        identity=identity(sequence),
        disposition=disposition,
        acquisition_time=1.0,
        arrival_time=1.01,
        processing_time=1.02,
    )
    if disposition is DeliveryDisposition.DROPPED:
        defaults["arrival_time"] = None
        defaults["processing_time"] = None
    if disposition is not DeliveryDisposition.ACCEPTED and "reason" not in kwargs:
        defaults["reason"] = "test disposition"
    defaults.update(kwargs)
    return DeliveryDecision(**defaults)


def summary(decisions, **overrides):
    values = dict(
        run_id="run-1",
        ordering_policy=TransportOrderingPolicy.ARRIVAL_ORDER,
        source_count=1,
        delivered_count=1,
        accepted_count=1,
        duplicate_count=0,
        late_count=0,
        invalid_count=0,
        dropped_count=0,
        source_hash=HASH_A,
        delivered_hash=HASH_B,
        configuration_hash=HASH_C,
        decisions=tuple(decisions),
    )
    values.update(overrides)
    return TransportReplaySummary(**values)


def test_identity_normalizes_sensor_and_exposes_stable_key():
    item = identity(sensor=" GPS ")
    assert item.sensor_id == "gps"
    assert item.key == ("gps", "device-1", 1)


@pytest.mark.parametrize("value", ["", "  ", None])
def test_identity_rejects_blank_sensor(value):
    with pytest.raises(IntegrationContractError):
        EnvelopeIdentity(sensor_id=value, sequence_number=0)


@pytest.mark.parametrize("value", [-1, True, 1.2])
def test_identity_rejects_invalid_sequence(value):
    with pytest.raises(IntegrationContractError):
        EnvelopeIdentity(sensor_id="gps", sequence_number=value)


def test_identity_rejects_blank_source_id():
    with pytest.raises(IntegrationContractError):
        EnvelopeIdentity(sensor_id="gps", sequence_number=0, source_id=" ")


def test_accepted_decision_accepts_valid_three_clock_evidence():
    item = decision()
    assert item.disposition is DeliveryDisposition.ACCEPTED
    assert item.processing_time == 1.02


@pytest.mark.parametrize("field", ["acquisition_time", "arrival_time"])
def test_accepted_requires_acquisition_and_arrival(field):
    with pytest.raises(IntegrationContractError):
        decision(**{field: None})


@pytest.mark.parametrize("field", ["acquisition_time", "arrival_time", "processing_time"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_accepted_rejects_non_finite_timestamps(field, value):
    with pytest.raises(IntegrationContractError):
        decision(**{field: value})


def test_accepted_rejects_processing_before_arrival():
    with pytest.raises(IntegrationContractError):
        decision(processing_time=1.0)


@pytest.mark.parametrize("disposition", [
    DeliveryDisposition.DUPLICATE,
    DeliveryDisposition.LATE,
    DeliveryDisposition.INVALID,
    DeliveryDisposition.DROPPED,
])
def test_nonaccepted_requires_reason(disposition):
    with pytest.raises(IntegrationContractError):
        DeliveryDecision(
            identity=identity(), disposition=disposition,
            acquisition_time=1.0, arrival_time=1.01,
        )


@pytest.mark.parametrize("field", ["arrival_time", "processing_time"])
def test_dropped_cannot_claim_arrival_or_processing(field):
    with pytest.raises(IntegrationContractError):
        decision(DeliveryDisposition.DROPPED, **{field: 1.01})


def test_nonaccepted_can_preserve_nonfinite_raw_timestamps():
    item = decision(
        DeliveryDisposition.INVALID,
        acquisition_time=float("nan"),
        arrival_time=float("inf"),
    )
    assert item.disposition is DeliveryDisposition.INVALID


def test_valid_summary_accounts_for_accepted_delivery():
    item = decision()
    result = summary([item])
    assert result.delivered_count == 1
    assert result.decisions == (item,)


def test_summary_accounts_for_duplicate_and_dropped_packet():
    accepted = decision(sequence=1)
    duplicate = decision(DeliveryDisposition.DUPLICATE, sequence=1)
    dropped = decision(DeliveryDisposition.DROPPED, sequence=2)
    result = summary(
        [accepted, duplicate, dropped],
        source_count=2, delivered_count=2, accepted_count=1,
        duplicate_count=1, dropped_count=1,
    )
    assert result.delivered_count == 2
    assert result.dropped_count == 1


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_summary_rejects_invalid_counts(value):
    with pytest.raises(IntegrationContractError):
        summary([decision()], source_count=value)


def test_summary_rejects_inconsistent_delivery_accounting():
    with pytest.raises(IntegrationContractError):
        summary([decision()], delivered_count=2)


def test_summary_rejects_drop_count_above_source_count():
    with pytest.raises(IntegrationContractError):
        summary([decision()], dropped_count=2)


def test_summary_rejects_wrong_number_of_decision_records():
    with pytest.raises(IntegrationContractError):
        summary([], delivered_count=1)


def test_summary_rejects_unaccounted_source_packets():
    with pytest.raises(IntegrationContractError):
        summary([decision()], source_count=2)


def test_summary_rejects_decision_count_mismatch():
    duplicate = decision(DeliveryDisposition.DUPLICATE)
    with pytest.raises(IntegrationContractError):
        summary([duplicate])


@pytest.mark.parametrize("field", ["source_hash", "delivered_hash", "configuration_hash"])
@pytest.mark.parametrize("value", ["", "xyz", "a" * 63, "g" * 64])
def test_summary_rejects_invalid_hashes(field, value):
    with pytest.raises(IntegrationContractError):
        summary([decision()], **{field: value})


def test_summary_requires_ordering_policy_enum():
    with pytest.raises(IntegrationContractError):
        summary([decision()], ordering_policy="arrival_order")


def test_summary_rejects_non_decision_items():
    with pytest.raises(IntegrationContractError):
        summary([object()])
