# RoboQA Lab

**Quality Assurance platform for sensor fusion and EKF state estimation systems.**

ARQTECH LABS — Operação RoboQA

## Status

| Component              | Version | Status   |
|------------------------|---------|----------|
| Data Engine (Fase 1)   | 0.1.0   | Hardened |
| Fusion Engine (Fase 2) | —       | Pending  |
| Fault Injection        | —       | Pending  |
| Quality Gate           | —       | Pending  |
| Streamlit Control Center | —     | Pending  |

## Phase 1 — Data Engine

Contratos robustos para:

- `GroundTruth` — trajetória de referência com validação dimensional, temporal e de proveniência.
- `Measurement` — medição sensorial com validação de covariância (simetria + PSD).
- Gerador sintético determinístico (`make_synthetic_ground_truth`).
- Loader KITTI OXTS com política explícita de fallback (sem silêncio).
- Proveniência discriminada: `kitti_real` | `synthetic` | `derived`.

### Instalação rápida

```bash
pip install -e ".[dev]"
pytest -q
```

### Proveniência

Todos os objetos de dados carregam o campo `source`. Medições de sensores são sempre simuladas; a trajetória pode ser real (KITTI) ou sintética.

```
Trajetória baseada em dados reais do KITTI (quando disponível).
Medições sensoriais são simuladas para avaliação experimental.
Os resultados representam experimentos computacionais,
não validação de sensores físicos.
```

## License

MIT (a definir na Fase 6)
