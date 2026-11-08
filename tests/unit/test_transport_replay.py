"""Unit and adversarial tests for deterministic transport replay (Phase 4.5-B)."""
import math

import pytest

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.integration import (
    DeliveryDisposition,
    IntegrationContractError,
    TransportOrderingPolicy,
    replay_transport,
)


def packet(sequence, acquisition, arrival, *, sensor="gps", source="unit-1",
           processing=None, payload=(1.0, 2.0)):
    return RawMeasurementEnvelope(
        sensor_id=sensor,
        source_id=source,
        sequence_number=sequence,
        acquisition_time=acquisition,
        arrival_time=arrival,
        processing_time=processing,
        payload=payload,
    )


def dispositions(result):
    return [item.disposition for item in result.summary.decisions]


def test_arrival_order_uses_watermark_and_emits_only_accepted():
    source = (
        packet(1, 1.0, 1.1),
        packet(2, 2.0, 1.2),
        packet(3, 1.5, 1.3),
    )
    result = replay_transport(source, source, allowed_lateness_s=0.25)

    assert result.summary.ordering_policy is TransportOrderingPolicy.ARRIVAL_ORDER
    assert result.summary.accepted_count == 2
    assert result.summary.late_count == 1
    assert [item.sequence_number for item in result.accepted_envelopes] == [1, 2]
    late = [item for item in result.summary.decisions
            if item.disposition is DeliveryDisposition.LATE]
    assert len(late) == 1
    assert "watermark" in late[0].reason


def test_allowed_lateness_can_retain_out_of_order_event_time_packet():
    source = (
        packet(1, 1.0, 1.1),
        packet(2, 2.0, 1.2),
        packet(3, 1.5, 1.3),
    )
    result = replay_transport(source, source, allowed_lateness_s=0.6)

    assert result.summary.accepted_count == 3
    assert result.summary.late_count == 0
    assert [item.sequence_number for item in result.accepted_envelopes] == [1, 2, 3]


def test_acquisition_order_reorders_and_does_not_classify_late():
    source = (
        packet(1, 1.0, 1.1),
        packet(2, 2.0, 1.2),
        packet(3, 1.5, 1.3),
    )
    result = replay_transport(
        source, source, ordering_policy=TransportOrderingPolicy.ACQUISITION_ORDER
    )

    assert [item.sequence_number for item in result.accepted_envelopes] == [1, 3, 2]
    assert result.summary.late_count == 0


def test_duplicate_is_suppressed_and_missing_source_is_dropped():
    first = packet(1, 1.0, 1.1)
    duplicate = packet(1, 1.0, 1.2, payload=(9.0, 9.0))
    second = packet(2, 2.0, 2.1)
    missing = packet(3, 3.0, 3.1)
    result = replay_transport(
        (first, second, missing), (first, duplicate, second)
    )

    assert result.summary.source_count == 3
    assert result.summary.delivered_count == 3
    assert result.summary.accepted_count == 2
    assert result.summary.duplicate_count == 1
    assert result.summary.dropped_count == 1
    assert [item.sequence_number for item in result.accepted_envelopes] == [1, 2]
    assert dispositions(result).count(DeliveryDisposition.DROPPED) == 1


def test_first_invalid_delivery_wins_and_later_copy_is_duplicate():
    invalid = packet(1, 1.0, 1.1, payload=(math.nan, 2.0))
    later_valid_copy = packet(1, 1.0, 1.2)
    result = replay_transport((invalid,), (invalid, later_valid_copy))

    assert result.summary.invalid_count == 1
    assert result.summary.duplicate_count == 1
    assert result.summary.accepted_count == 0
    assert not result.accepted_envelopes


@pytest.mark.parametrize(
    "item",
    [
        packet(1, 2.0, 1.0),
        packet(1, 1.0, 1.1, processing=1.0),
        packet(1, 1.0, 1.1, processing=math.inf),
        packet(1, 1.0, 1.1, payload=(math.inf,)),
    ],
)
def test_invalid_clock_or_payload_is_rejected(item):
    result = replay_transport((item,), (item,))
    assert result.summary.invalid_count == 1
    assert result.summary.accepted_count == 0
    assert dispositions(result) == [DeliveryDisposition.INVALID]


def test_replay_is_independent_of_input_collection_order():
    a = packet(1, 1.0, 1.1, sensor="gps", source="source-a")
    b = packet(1, 1.0, 1.1, sensor="speed", source=None)
    c = packet(2, 2.0, 2.1, sensor="gps", source="source-a")
    left = replay_transport((a, b, c), (a, b, c))
    right = replay_transport((c, a, b), (c, b, a))

    assert left.summary.source_hash == right.summary.source_hash
    assert left.summary.delivered_hash == right.summary.delivered_hash
    assert left.summary.configuration_hash == right.summary.configuration_hash
    assert left.summary.run_id == right.summary.run_id
    assert [
        (item.sensor_id, item.sequence_number)
        for item in left.accepted_envelopes
    ] == [
        (item.sensor_id, item.sequence_number)
        for item in right.accepted_envelopes
    ]


def test_rejects_duplicate_identity_in_source_snapshot():
    item = packet(1, 1.0, 1.1)
    duplicate = packet(1, 1.0, 1.2)
    with pytest.raises(IntegrationContractError, match="source snapshot"):
        replay_transport((item, duplicate), (item,))


def test_rejects_delivered_identity_missing_from_source():
    with pytest.raises(IntegrationContractError, match="absent from source"):
        replay_transport((packet(1, 1.0, 1.1),), (packet(2, 2.0, 2.1),))


@pytest.mark.parametrize("value", [-1, math.inf, math.nan, True, "bad"])
def test_rejects_invalid_allowed_lateness(value):
    item = packet(1, 1.0, 1.1)
    with pytest.raises(IntegrationContractError, match="allowed_lateness_s"):
        replay_transport((item,), (item,), allowed_lateness_s=value)


def test_empty_delivery_accounts_for_every_source_as_dropped():
    source = (packet(1, 1.0, 1.1), packet(2, 2.0, 2.1))
    result = replay_transport(source, ())

    assert result.summary.accepted_count == 0
    assert result.summary.dropped_count == 2
    assert len(result.summary.decisions) == 2
    assert all(item.disposition is DeliveryDisposition.DROPPED
               for item in result.summary.decisions)


def test_empty_source_and_delivery_is_a_valid_deterministic_replay():
    result = replay_transport((), ())
    assert result.summary.source_count == 0
    assert result.summary.delivered_count == 0
    assert result.summary.accepted_count == 0
    assert result.summary.run_id.startswith("replay-")
