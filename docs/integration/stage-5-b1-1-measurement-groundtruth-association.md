# B1.1 — Measurement to GroundTruth association

Status: implementation candidate; manual Colab validation pending.

## Contract

RoboQA now has a separate association record rather than assuming that a
Measurement's position in a Python list equals a GroundTruth sample index.
Each result records both sequence identifiers, both sample indexes, status,
method, signed time delta, declared tolerance, and evidence.

The timestamp matcher requires explicit and exactly matching clock-domain and
time-basis declarations. It only accepts a unique candidate within the
caller-supplied tolerance. Unmatched samples remain unmatched; multiple
candidates and one-to-many claims remain ambiguous. A reference sample cannot
be silently reused.

## Important boundary

This helper does not infer UTC, hardware synchronization, or clock equivalence.
It does not alter the Measurement or GroundTruth base schemas and does not
change EKF behavior. The caller must establish that the declared time domain
and time basis are factually valid for the source data.

For synthetic measurements generated from a GroundTruth trajectory, the
generator copies each source timestamp. A caller may declare the common
synthetic sequence clock and use a strict tolerance. This does not generalize
to independent real sensor streams.

## Validation

Manual Colab validation must verify exact branch/commit identity, compile
RoboQA, run the complete RoboQA suite, and execute the new association tests.
No dataset download, dependency installation, training, GitHub Actions,
push to main, or merge is part of this candidate stage.
