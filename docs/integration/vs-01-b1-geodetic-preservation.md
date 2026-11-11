# VS-01-B1 — Preserve original geodetic coordinates in RoboQA

Status: implementation candidate; manual Colab validation pending.

## Scope

This change preserves the original KITTI OXTS geodetic samples in the RoboQA
`GroundTruth` artifact while retaining the existing local Cartesian
`pos` representation and its current behavior.

The new optional fields are:

- `geodetic_position`: an immutable defensive-copy array with shape (N, 3),
  ordered as latitude degrees, longitude degrees, altitude metres.
- `geodetic_datum`: explicit non-empty datum identifier; KITTI OXTS loader
  currently declares `WGS84`.

Datasets without geodetic source observations leave both fields as `None`.
A datum without coordinates is rejected. Coordinates without an explicit
datum are rejected. Latitude and longitude ranges, array dimensions, sample
count, and finite values are validated.

## Important semantic boundary

The existing `GroundTruth.pos` remains the output of RoboQA's
`latlon_to_local()` equirectangular approximation. Preserving source
latitude/longitude/altitude does **not** convert that approximation into ENU,
does not validate a geodetic transform, and does not synchronize timestamps.

The KITTI classic OXTS loader still uses an approximate 10 Hz sequence clock
(`t = arange(N) * dt_approx`). It is not represented as synchronized UTC or
as a hardware clock.

No changes are made to the native `Measurement(t, sensor, z, R)` contract,
the RoboQA EKF, or the Core adapter in this substage. The next substage must
define how an individual GPS measurement references its original geodetic
sample without fabricating a relationship between independently generated
measurements and ground-truth samples.

## Validation required

- Exact branch/commit identity in Colab.
- Compile RoboQA source and tests with SyntaxWarnings treated as errors.
- Run the full existing RoboQA test suite.
- Verify KITTI source geodetic values survive loading and defensive copying.
- Verify synthetic GroundTruth remains compatible with absent geodetic data.
- Verify invalid datum, dimensions, non-finite values, and coordinate ranges fail closed.

No dataset download, dependency installation, training, push, merge, or
GitHub Actions execution is part of this substage.
