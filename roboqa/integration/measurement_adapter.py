"""Strict adapter from accepted transport envelopes to Fusion measurements.

Phase 4.5-C deliberately does not schedule or execute the Fusion Engine.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from roboqa.data.schemas import Measurement, SchemaError
from roboqa.fusion.sensors import validate_measurement_for_fusion
from roboqa.integration.contracts import EnvelopeIdentity, IntegrationContractError
from roboqa.integration.replay import TransportReplayResult


class MeasurementAdapterError(ValueError):
    """Raised when accepted transport evidence cannot form a valid Measurement."""


@dataclass(frozen=True)
class AdaptedMeasurement:
    """A normalized Measurement with its transport identity and clock evidence."""

    identity: EnvelopeIdentity
    measurement: Measurement
    arrival_time: float
    processing_time: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EnvelopeIdentity):
            raise MeasurementAdapterError("identity must be an EnvelopeIdentity")
        if not isinstance(self.measurement, Measurement):
            raise MeasurementAdapterError("measurement must be a Measurement")
        arrival = float(self.arrival_time)
        if not np.isfinite(arrival) or arrival < self.measurement.t:
            raise MeasurementAdapterError("arrival_time must be finite and >= acquisition time")
        object.__setattr__(self, "arrival_time", arrival)
        if self.processing_time is not None:
            processing = float(self.processing_time)
            if not np.isfinite(processing) or processing < arrival:
                raise MeasurementAdapterError("processing_time must be finite and >= arrival_time")
            object.__setattr__(self, "processing_time", processing)
        if self.identity.sensor_id != self.measurement.sensor.strip().lower():
            raise MeasurementAdapterError("identity sensor does not match Measurement sensor")


def adapt_accepted_envelopes(
    replay: TransportReplayResult,
    *,
    sensor_covariances: Mapping[str, np.ndarray],
) -> tuple[AdaptedMeasurement, ...]:
    """Adapt only replay-accepted envelopes using explicit per-sensor R matrices.

    Payload is interpreted as measurement vector z. Measurement time is
    acquisition_time. Arrival/processing clocks and source identity remain
    alongside the normalized record for downstream provenance.

    Covariance is never inferred: callers must provide R for every sensor
    present in accepted envelopes. Keys are normalized to lowercase.
    """
    if not isinstance(replay, TransportReplayResult):
        raise MeasurementAdapterError("replay must be a TransportReplayResult")
    if not isinstance(sensor_covariances, Mapping):
        raise MeasurementAdapterError("sensor_covariances must be a mapping")

    covariance_by_sensor: dict[str, np.ndarray] = {}
    for sensor, covariance in sensor_covariances.items():
        if not isinstance(sensor, str) or not sensor.strip():
            raise MeasurementAdapterError("covariance sensor IDs must be non-empty strings")
        key = sensor.strip().lower()
        if key in covariance_by_sensor:
            raise MeasurementAdapterError("covariance sensor IDs must be unique after normalization")
        try:
            candidate = np.asarray(covariance, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MeasurementAdapterError(f"invalid covariance for sensor {key}") from exc
        if candidate.ndim != 2 or candidate.shape[0] != candidate.shape[1]:
            raise MeasurementAdapterError(f"R for sensor {key} must be a square matrix")
        if not np.all(np.isfinite(candidate)):
            raise MeasurementAdapterError(f"R for sensor {key} must be finite")
        covariance_by_sensor[key] = candidate.copy()

    adapted: list[AdaptedMeasurement] = []
    for envelope in replay.accepted_envelopes:
        sensor = envelope.sensor_id.strip().lower()
        if sensor not in covariance_by_sensor:
            raise MeasurementAdapterError(f"missing explicit R covariance for sensor {sensor}")
        try:
            measurement = Measurement(
                t=envelope.acquisition_time,
                sensor=sensor,
                z=np.asarray(envelope.payload, dtype=np.float64),
                R=covariance_by_sensor[sensor],
            )
            validate_measurement_for_fusion(measurement)
            identity = EnvelopeIdentity(
                sensor_id=envelope.sensor_id,
                source_id=envelope.source_id,
                sequence_number=envelope.sequence_number,
            )
            adapted.append(AdaptedMeasurement(
                identity=identity,
                measurement=measurement,
                arrival_time=envelope.arrival_time,
                processing_time=envelope.processing_time,
            ))
        except (SchemaError, IntegrationContractError, TypeError, ValueError, OverflowError) as exc:
            raise MeasurementAdapterError(
                f"cannot adapt accepted envelope {sensor}#{envelope.sequence_number}: {exc}"
            ) from exc

    return tuple(adapted)
