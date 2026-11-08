"""Unit tests for the strict transport-to-Measurement adapter."""

import numpy as np
import pytest

from roboqa.fault_injection.envelopes import RawMeasurementEnvelope
from roboqa.integration import TransportOrderingPolicy, replay_transport
from roboqa.integration.measurement_adapter import (
    MeasurementAdapterError,
    adapt_accepted_envelopes,
)


def envelope(sequence, sensor, acquisition, arrival, payload, processing=None):
    return RawMeasurementEnvelope(
        sensor_id=sensor,
        source_id="adapter-test",
        sequence_number=sequence,
        acquisition_time=acquisition,
        arrival_time=arrival,
        processing_time=processing,
        payload=payload,
    )


def replay(items):
    return replay_transport(
        tuple(items), tuple(items),
        ordering_policy=TransportOrderingPolicy.ACQUISITION_ORDER,
    )


def test_adapts_only_accepted_envelopes_and_preserves_identity_and_clocks():
    gps = envelope(1, "GPS", 1.0, 1.1, (10.0, 20.0), 1.2)
    speed = envelope(2, "speed", 1.0, 1.15, (4.5,))
    result = replay((gps, speed))
    adapted = adapt_accepted_envelopes(
        result, sensor_covariances={"gps": np.eye(2), "SPEED": np.array([[0.25]])}
    )
    assert len(adapted) == 2
    assert adapted[0].identity.sensor_id == "gps"
    assert adapted[0].identity.sequence_number == 1
    assert adapted[0].measurement.sensor == "gps"
    assert adapted[0].measurement.t == 1.0
    np.testing.assert_array_equal(adapted[0].measurement.z, [10.0, 20.0])
    assert adapted[0].arrival_time == 1.1
    assert adapted[0].processing_time == 1.2
    assert adapted[1].processing_time is None
    assert not adapted[0].measurement.z.flags.writeable
    assert not adapted[0].measurement.R.flags.writeable


def test_replay_nonaccepted_packets_are_not_adapted():
    first = envelope(1, "gps", 1.0, 1.1, (1.0, 2.0))
    duplicate = envelope(1, "gps", 1.0, 1.2, (9.0, 9.0))
    result = replay_transport((first,), (first, duplicate))
    assert result.summary.duplicate_count == 1
    adapted = adapt_accepted_envelopes(result, sensor_covariances={"gps": np.eye(2)})
    assert len(adapted) == 1
    np.testing.assert_array_equal(adapted[0].measurement.z, [1.0, 2.0])


def test_requires_explicit_covariance_for_each_accepted_sensor():
    result = replay((envelope(1, "gps", 1.0, 1.1, (1.0, 2.0)),))
    with pytest.raises(MeasurementAdapterError, match="missing explicit R"):
        adapt_accepted_envelopes(result, sensor_covariances={})


@pytest.mark.parametrize("covariance", [np.array([1.0, 1.0]), np.array([[np.nan]])])
def test_rejects_non_square_or_nonfinite_covariance(covariance):
    result = replay((envelope(1, "speed", 1.0, 1.1, (2.0,)),))
    with pytest.raises(MeasurementAdapterError):
        adapt_accepted_envelopes(result, sensor_covariances={"speed": covariance})


def test_rejects_covariance_that_is_not_positive_semidefinite():
    result = replay((envelope(1, "speed", 1.0, 1.1, (2.0,)),))
    with pytest.raises(MeasurementAdapterError, match="semidefinida positiva"):
        adapt_accepted_envelopes(result, sensor_covariances={"speed": np.array([[-1.0]])})


def test_rejects_unknown_sensor_even_when_covariance_is_supplied():
    result = replay((envelope(1, "lidar", 1.0, 1.1, (2.0,)),))
    with pytest.raises(MeasurementAdapterError, match="Sensor desconhecido"):
        adapt_accepted_envelopes(result, sensor_covariances={"lidar": np.eye(1)})


def test_rejects_payload_dimension_mismatch_for_known_sensor():
    result = replay((envelope(1, "gps", 1.0, 1.1, (2.0,)),))
    with pytest.raises(MeasurementAdapterError, match="dim"):
        adapt_accepted_envelopes(result, sensor_covariances={"gps": np.eye(2)})


def test_rejects_covariance_keys_that_collide_after_normalization():
    result = replay((envelope(1, "gps", 1.0, 1.1, (1.0, 2.0)),))
    with pytest.raises(MeasurementAdapterError, match="unique after normalization"):
        adapt_accepted_envelopes(
            result, sensor_covariances={"gps": np.eye(2), " GPS ": np.eye(2)}
        )


def test_empty_accepted_set_returns_empty_without_inventing_measurements():
    source = envelope(1, "gps", 1.0, 1.1, (1.0, 2.0))
    result = replay_transport((source,), ())
    assert result.summary.dropped_count == 1
    assert adapt_accepted_envelopes(result, sensor_covariances={}) == ()

