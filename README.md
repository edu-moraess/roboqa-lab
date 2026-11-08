# RoboQA Lab

**Quality Assurance platform for sensor fusion and EKF state estimation systems.**

ARQTECH LABS — Operação RoboQA

## Status

| Component              | Version | Status        |
|------------------------|---------|---------------|
| Data Engine (Fase 1)   | 0.1.0   | PASS          |
| Fusion Engine (Fase 2) | 0.2.0   | Hardening     |
| Fault Injection        | —       | Pending       |
| Quality Gate           | —       | Pending       |
| Streamlit Control Center | —     | Pending       |

## Phase 1 — Data Engine

Contratos robustos para GroundTruth, Measurement, gerador sintético e loader KITTI com proveniência explícita e arrays somente leitura.

## Phase 2 — Fusion Engine

- Modelo CTRV com Jacobiano analítico e testes por diferenças finitas
- EKF com atualização de Joseph e validação de simetria/semidefinição positiva da covariância
- Sensores: GPS, speed, yaw_rate
- Runner assíncrono com ordenação determinística (t, sensor) e política temporal explícita
- simulate_measurements + experimento sintético reproduzível

### Modelo de ruído de processo

O ruído de processo usa um modelo discreto de aceleração longitudinal e angular aleatória, constante durante cada intervalo de integração (zero-order hold). sigma_a representa o desvio-padrão da aceleração longitudinal em m/s²; sigma_alpha, o desvio-padrão da aceleração angular em rad/s². A matriz é construída como Q = B diag(sigma_a², sigma_alpha²) B.T, incluindo covariâncias cruzadas posição–velocidade e yaw–taxa de yaw. Portanto, este contrato **não** deve ser interpretado como discretização de ruído branco contínuo.

A contribuição longitudinal de posição é projetada na direção do heading no início do intervalo. A perturbação angular é aplicada aos estados yaw e yaw-rate; sua influência direta na posição dentro do mesmo intervalo não é modelada separadamente. Trata-se de uma aproximação discreta split-step explícita, coerente com o mapa B documentado, e não de uma discretização contínua completa por integração de Van Loan.

### Instalação rápida

    pip install -e ".[dev]"
    pytest -q

### Experimento sintético

    from roboqa.data.synthetic import make_synthetic_ground_truth
    from roboqa.fusion import simulate_measurements, run_filter

    gt = make_synthetic_ground_truth(n=100, seed=42)
    ms = simulate_measurements(gt, seed=42)
    result = run_filter(ms, provenance={"gt_source": gt.source, "seed": 42})
    print(result.status, result.diagnostics.n_update)

### Proveniência

    Trajetória baseada em dados reais do KITTI (quando disponível).
    Medições sensoriais são simuladas para avaliação experimental.
    Os resultados representam experimentos computacionais,
    não validação de sensores físicos.

## License

MIT (a definir na Fase 6)
