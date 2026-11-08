import math

import numpy as np
import pytest

from roboqa.fault_injection import ContractError, RawMeasurementEnvelope


def test_raw_envelope_preserves_invalid_payload_for_boundary_diagnostics():
    envelope = RawMeasurementEnvelope(
        sensor_id="gps",
        acquisition_time=1.0,
        arrival_time=1.1,
        sequence_number=0,
        payload={"z": [1.0, float("nan")], "R": [[1.0, 0.0], [0.0, 1.0]]},
    )
    assert not envelope.payload_finite
    assert math.isnan(envelope.payload["z"][1])
    assert not envelope.is_valid_candidate


def test_raw_envelope_preserves_nonfinite_timestamp():
    envelope = RawMeasurementEnvelope(
        sensor_id="imu",
        acquisition_time=float("inf"),
        arrival_time=2.0,
        sequence_number=1,
        payload={"z": [0.0]},
    )
    assert not envelope.timestamps_finite
    assert not envelope.is_valid_candidate


def test_raw_envelope_payload_is_immutable():
    envelope = RawMeasurementEnvelope(
        sensor_id="gps", acquisition_time=1.0, arrival_time=1.1,
        sequence_number=0, payload={"z": [1.0, 2.0]},
    )
    with pytest.raises(TypeError):
        envelope.payload["z"] = [9.0]
    assert envelope.payload["z"] == (1.0, 2.0)


def test_raw_envelope_rejects_invalid_transport_identity():
    with pytest.raises(ContractError):
        RawMeasurementEnvelope(
            sensor_id="", acquisition_time=1.0, arrival_time=1.0,
            sequence_number=0, payload={},
        )


def test_raw_envelope_accepts_numpy_payload_without_mutable_aliases():
    source = np.array([1.0, 2.0])
    envelope = RawMeasurementEnvelope(
        sensor_id="gps", acquisition_time=1.0, arrival_time=1.1,
        sequence_number=0, payload={"z": source},
    )
    source[0] = 99.0
    assert envelope.payload["z"] == (1.0, 2.0)



@pytest.mark.parametrize("kwargs", [
    {"acquisition_time": True},
    {"arrival_time": "1.0"},
    {"sequence_number": True},
    {"sequence_number": -1},
    {"source_id": ""},
])
def test_raw_envelope_rejects_invalid_scalar_fields(kwargs):
    values = {
        "sensor_id": "gps",
        "acquisition_time": 1.0,
        "arrival_time": 1.1,
        "sequence_number": 0,
        "payload": {},
    }
    values.update(kwargs)
    with pytest.raises(ContractError):
        RawMeasurementEnvelope(**values)


def test_raw_envelope_rejects_non_string_mapping_keys_and_unsupported_objects():
    with pytest.raises(ContractError, match="keys must be strings"):
        RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {1: "bad"})
    with pytest.raises(ContractError, match="unsupported type"):
        RawMeasurementEnvelope("gps", 1.0, 1.1, 0, {"bad": object()})
