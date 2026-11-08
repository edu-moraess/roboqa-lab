# RoboQA Lab

**Quality Assurance platform for sensor fusion and EKF state estimation systems.**

ARQTECH LABS — Operação RoboQA

## Status

| Component              | Version | Status        |
|------------------------|---------|---------------|
| Data Engine (Fase 1)   | 0.1.0   | PASS          |
| Fusion Engine (Fase 2) | 0.2.0   | Implemented   |
| Fault Injection        | —       | Pending       |
| Quality Gate           | —       | Pending       |
| Streamlit Control Center | —     | Pending       |

## Phase 1 — Data Engine

Contratos robustos para GroundTruth, Measurement, gerador sintético e loader KITTI com proveniência explícita e arrays somente leitura.

## Phase 2 — Fusion Engine

- Modelo CTRV com Jacobiano analítico validado por diferenças finitas
- EKF com atualização de Joseph e ruído de processo dependente de Δt
- Sensores: GPS, speed, yaw_rate
- Runner assíncrono com ordenação determinística (t, sensor)
- `simulate_measurements` + experimento sintético reproduzível

### Instalação rápida

```bash
pip install -e ".[dev]"
pytest -q
```

### Experimento sintético

```python
from roboqa.data.synthetic import make_synthetic_ground_truth
from roboqa.fusion import simulate_measurements, run_filter

gt = make_synthetic_ground_truth(n=100, seed=42)
ms = simulate_measurements(gt, seed=42)
result = run_filter(ms, provenance={"gt_source": gt.source, "seed": 42})
print(result.status, result.diagnostics.n_update)
```

### Proveniência

```
Trajetória baseada em dados reais do KITTI (quando disponível).
Medições sensoriais são simuladas para avaliação experimental.
Os resultados representam experimentos computacionais,
não validação de sensores físicos.
```

## License

MIT (a definir na Fase 6)
