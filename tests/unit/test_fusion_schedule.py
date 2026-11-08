"""Unit tests for deterministic event-time Fusion scheduling (Phase 4.5-D1)."""

import numpy as np
import pytest

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.integration import (
    TransportOrderingPolicy,
    adapt_accepted_envelopes,
    replay_transport,
)
from roboqa.integration.measurement_adapter import AdaptedMeasurement
from roboqa.integration.scheduler import (
    FusionSchedulingError,
    FusionSchedulingPolicy,
    build_fusion_schedule,
)


def packet(sequence, sensor, acquisition, arrival, payload, *, source="schedule-test",
           processing=None):
    return RawMeasurementEnvelope(
        sensor_id=sensor,
        source_id=source,
        sequence_number=sequence,
        acquisition_time=acquisition,
        arrival_time=arrival,
        processing_time=processing,
        payload=payload,
    )


def adapt(replay):
    return adapt_accepted_envelopes(
        replay,
        sensor_covariances={"gps": np.eye(2), "speed": np.array([[0.25]])},
    )


def test_arrival_order_replay_is_rescheduled_into_monotonic_event_time():
    source = (
        packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),
        packet(2, "gps", 2.0, 2.2, (2.0, 3.0)),
        packet(3, "gps", 1.5, 2.3, (1.5, 2.5)),
    )
    replay = replay_transport(
        source, source,
        ordering_policy=TransportOrderingPolicy.ARRIVAL_ORDER,
        allowed_lateness_s=0.6,
    )
    assert [item.sequence_number for item in replay.accepted_envelopes] == [1, 2, 3]

    schedule = build_fusion_schedule(replay, adapt(replay))

    assert schedule.policy is FusionSchedulingPolicy.ACQUISITION_TIME
    assert [item.identity.sequence_number for item in schedule.entries] == [1, 3, 2]
    assert [item.t for item in schedule.measurements] == [1.0, 1.5, 2.0]
    assert schedule.replay_summary is replay.summary


def test_schedule_is_independent_of_adapter_result_collection_order():
    source = (
        packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),
        packet(2, "speed", 1.0, 1.15, (4.0,)),
        packet(3, "gps", 1.0, 1.2, (3.0, 4.0)),
    )
    replay = replay_transport(
        source, source,
        ordering_policy=TransportOrderingPolicy.ARRIVAL_ORDER,
    )
    adapted = adapt(replay)

    left = build_fusion_schedule(replay, adapted)
    right = build_fusion_schedule(replay, tuple(reversed(adapted)))

    assert [item.identity for item in left.entries] == [
        item.identity for item in right.entries
    ]
    assert left.schedule_hash == right.schedule_hash
    assert [item.identity.sensor_id for item in left.entries] == [
        "gps", "gps", "speed"
    ]
    assert [item.identity.sequence_number for item in left.entries] == [1, 3, 2]


def test_same_sensor_and_timestamp_use_source_then_sequence_tiebreakers():
    source = (
        packet(2, "gps", 1.0, 1.1, (2.0, 2.0), source="b"),
        packet(1, "gps", 1.0, 1.2, (1.0, 1.0), source="a"),
        packet(1, "gps", 1.0, 1.3, (3.0, 3.0), source="b"),
    )
    replay = replay_transport(source, source)
    schedule = build_fusion_schedule(replay, adapt(replay))

    assert [
        (item.identity.source_id, item.identity.sequence_number)
        for item in schedule.entries
    ] == [("a", 1), ("b", 1), ("b", 2)]


def test_schedule_hash_includes_explicit_covariance():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    first = adapt_accepted_envelopes(replay, sensor_covariances={"gps": np.eye(2)})
    second = adapt_accepted_envelopes(
        replay, sensor_covariances={"gps": np.eye(2) * 2.0}
    )

    left = build_fusion_schedule(replay, first)
    right = build_fusion_schedule(replay, second)

    assert left.schedule_hash != right.schedule_hash


def test_schedule_preserves_processing_clock_and_replay_accounting():
    accepted = packet(1, "gps", 1.0, 1.1, (1.0, 2.0), processing=1.2)
    dropped = packet(2, "gps", 2.0, 2.1, (2.0, 3.0))
    replay = replay_transport((accepted, dropped), (accepted,))
    schedule = build_fusion_schedule(replay, adapt(replay))

    assert schedule.entries[0].arrival_time == 1.1
    assert schedule.entries[0].processing_time == 1.2
    assert schedule.replay_summary.dropped_count == 1
    assert schedule.replay_summary.accepted_count == 1


def test_empty_accepted_batch_produces_empty_schedule_with_replay_evidence():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, ())
    schedule = build_fusion_schedule(replay, ())

    assert schedule.entries == ()
    assert schedule.measurements == ()
    assert schedule.replay_summary.dropped_count == 1


@pytest.mark.parametrize(
    "adapted",
    [
        (),
        (object(),),
    ],
)
def test_rejects_missing_or_invalid_adapter_records(adapted):
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    with pytest.raises(FusionSchedulingError):
        build_fusion_schedule(replay, adapted)


def test_rejects_duplicate_adapter_identity():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    adapted = adapt(replay)
    with pytest.raises(FusionSchedulingError, match="unique"):
        build_fusion_schedule(replay, adapted + adapted)


def test_rejects_adapter_identity_not_present_in_replay():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    adapted = adapt(replay)
    forged = AdaptedMeasurement(
        identity=adapted[0].identity.__class__(
            sensor_id="gps", source_id="other-source", sequence_number=1
        ),
        measurement=adapted[0].measurement,
        arrival_time=adapted[0].arrival_time,
        processing_time=adapted[0].processing_time,
    )
    with pytest.raises(FusionSchedulingError, match="exactly one"):
        build_fusion_schedule(replay, (forged,))


def test_rejects_adapter_arrival_clock_mismatch():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    adapted = adapt(replay)
    forged = AdaptedMeasurement(
        identity=adapted[0].identity,
        measurement=adapted[0].measurement,
        arrival_time=1.15,
        processing_time=None,
    )
    with pytest.raises(FusionSchedulingError, match="arrival_time"):
        build_fusion_schedule(replay, (forged,))


def test_rejects_adapter_payload_mismatch():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    adapted = adapt(replay)
    forged_measurement = type(adapted[0].measurement)(
        t=1.0, sensor="gps", z=np.array([9.0, 9.0]), R=np.eye(2)
    )
    forged = AdaptedMeasurement(
        identity=adapted[0].identity,
        measurement=forged_measurement,
        arrival_time=adapted[0].arrival_time,
        processing_time=adapted[0].processing_time,
    )
    with pytest.raises(FusionSchedulingError, match="Measurement.z"):
        build_fusion_schedule(replay, (forged,))


def test_rejects_invalid_schedule_hash_on_direct_construction():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    schedule = build_fusion_schedule(replay, adapt(replay))
    with pytest.raises(FusionSchedulingError, match="schedule_hash"):
        type(schedule)(
            replay_summary=schedule.replay_summary,
            entries=schedule.entries,
            schedule_hash="0" * 64,
        )


def test_rejects_measurement_acquisition_time_mismatch():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0)),)
    replay = replay_transport(source, source)
    adapted = adapt(replay)
    forged_measurement = type(adapted[0].measurement)(
        t=1.05, sensor="gps", z=np.array([1.0, 2.0]), R=np.eye(2)
    )
    forged = AdaptedMeasurement(
        identity=adapted[0].identity,
        measurement=forged_measurement,
        arrival_time=adapted[0].arrival_time,
        processing_time=adapted[0].processing_time,
    )
    with pytest.raises(FusionSchedulingError, match="acquisition_time"):
        build_fusion_schedule(replay, (forged,))


def test_rejects_adapter_processing_clock_mismatch():
    source = (packet(1, "gps", 1.0, 1.1, (1.0, 2.0), processing=1.2),)
    replay = replay_transport(source, source)
    adapted = adapt(replay)
    forged = AdaptedMeasurement(
        identity=adapted[0].identity,
        measurement=adapted[0].measurement,
        arrival_time=adapted[0].arrival_time,
        processing_time=1.25,
    )
    with pytest.raises(FusionSchedulingError, match="processing_time"):
        build_fusion_schedule(replay, (forged,))
