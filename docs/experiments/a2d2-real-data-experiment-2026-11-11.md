# VYRA — A2D2 real-data experiment evidence

**Date:** 2026-11-11  
**Repository:** `edu-moraess/vyra-roboqa-intell`  
**Experiment mode:** read-only  
**Data access:** anonymous public S3  
**Training:** disabled  
**Repository modifications during experiment:** none

## Purpose

Run a bounded exploratory experiment against real A2D2 vehicle data to verify that the VYRA workflow can discover a public sequence, retrieve vehicle telemetry and camera samples, parse geodetic GPS signals, align latitude/longitude observations, and produce auditable diagnostic outputs.

This experiment is complementary evidence for Stage 5 VS-01. It does not replace the contract/integration milestones and does not establish end-to-end sensor fusion.

## Dataset selection and acquisition

- Discovered public sequences: **3**
- Selected sequence: `camera_lidar/20180810_150607/`
- Vehicle bus file: `camera_lidar/20180810_150607/bus/20180810150607_bus_signals.json`
- Bus file size reported by the script: **100.69 MB**
- Downloaded bus file SHA-256: `99dc8bcef028ef4e52164a789b693ddc1d2bd3b67ff7bdc2f0c808a7b2da82d0`
- Camera frames discovered: **15,688**
- Three sampled front-center images downloaded successfully:
  - `20180810150607_camera_frontcenter_000000060.png`
  - `20180810150607_camera_frontcenter_000007904.png`
  - `20180810150607_camera_frontcenter_000015747.png`

## Parsed telemetry

- Latitude signal: `latitude_degree`
- Longitude signal: `longitude_degree`
- Latitude observations: **2,629**
- Longitude observations: **2,629**
- Valid GPS pairs: **2,629**
- Maximum pairing delta reported: **0 microseconds**
- Recorded experiment duration: **525.602 seconds**
- Vehicle speed signal: `vehicle_speed`
- Aligned speed samples: **2,629**

The zero pairing delta describes the pairing performed by this script; it does not prove synchronization between GPS, camera, and other sensor clocks.

## Generated outputs reported by the experiment

Output directory in the Colab session: `/content/vyra_a2d2_real_experiment`

- CSV: `reports/a2d2_gps_observations.csv`
- Report: `reports/experiment_report.txt`
- Manifest: `reports/experiment_manifest.json`
- Five plots:
  - `01_gps_trajectory.png`
  - `02_coordinates_over_time.png`
  - `03_speed_or_sampling.png`
  - `04_sampling_distribution.png`
  - `05_derived_displacement.png`
- Three sampled camera images listed above.

The Colab script reported PASS for GPS integrity, coordinate ranges, timestamp ordering, CSV generation, all five plots, all three camera images, manifest, and report.

## Visual observations

The supplied plots show a nontrivial GPS trajectory, variation in latitude and longitude over the approximately 525.6-second recording, vehicle-speed variation, two visible clusters in the sampling-interval histogram (approximately 0.16 s and 0.32 s), and consecutive-position displacement peaks approaching 20 m.

These are exploratory visual observations from the supplied plots, not a validated GPS error model or a scientific assessment of sensor accuracy. The displacement calculation must be interpreted in the context of the actual timestamp intervals and the dataset's sampling semantics.

## Explicit limitations and not-run items

The script explicitly reported:

- Geodetic altitude validation: **NOT PERFORMED**
- ENU transformation: **NOT PERFORMED**
- Full B1 N×3 validation: **NOT PERFORMED**

The experiment also did not demonstrate camera/GPS clock synchronization, geodetic-to-local transformation, ESKF ingestion, or end-to-end sensor fusion.

No model training, package installation, or repository modification occurred during the experiment.

## Evidence provenance and persistence boundary

This record preserves the experiment summary and the exact bus-file SHA-256 reported by the Colab script. The original CSV, JSON manifest, text report, plots, and downloaded camera files were generated under the Colab session path above. They were **not** transferred into this repository by this documentation commit. Their continued availability depends on the Colab runtime/storage state until separately archived.

## Roadmap relationship

This experiment is complementary to the validated Stage 5 work:

1. VS-01-A — native RoboQA → Core → Sensor Fusion bridge validated.
2. VS-01-B1 — geodetic source-coordinate preservation in RoboQA validated.
3. B1.1 — establish explicit and defensible association between source measurements and geodetic reference samples.
4. B2 — implement and validate an explicit, versioned geodetic transformation in GeoCore.
5. B3–B5 — resolve temporal semantics, complete adapter/tests, and perform exact-SHA Colab integration validation.

A2D2 does not change this sequence or authorize implicit LOCAL_CARTESIAN/ENU conversion.
