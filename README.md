# RoboQA Lab

**Plataforma modular de Quality Assurance para sistemas de fusão sensorial e estimação de estado.**

**ArqTech Labs — Engineering Program**

RoboQA Lab não é apenas uma demonstração de EKF. O objetivo é construir uma plataforma de engenharia para testar, observar e avaliar pipelines de percepção/estimação sensorial sob dados normais, falhas controladas e condições temporais adversas.

> **Status do projeto:** implementação em andamento — base validada até a **Fase 4.1**; Fase 4.2 implementada; testes direcionados e cobertura de 100% em `metrics.py`; validação global pendente. O produto completo ainda não está concluído.

## Visão da plataforma

```text
Data Engine
    ↓
Fusion Engine
    ↓
Fault Injection ──→ Temporal Scheduler / Transport Faults
    ↓
Fusion Telemetry
    ↓
Quality Metrics
    ↓
Deterministic Quality Gate
    ↓
ML Intelligence (advisory only)
    ↓
Streamlit Control Center
```

O fluxo acima representa a arquitetura-alvo. Nem todos os módulos estão implementados ainda. O Streamlit Control Center é um componente central planejado, não uma entrega concluída.

## Estado atual

| Fase | Entrega | Estado |
|---|---|---|
| 1 | Data Engine — contratos, dados sintéticos e ingestão | **VALIDATED** |
| 2.2 | Fusion Engine — CTRV + EKF hardening | **VALIDATED** |
| 3.1–3.3-E | Fault Injection, temporal pipeline, campanhas e integração | **VALIDATED** |
| 4.0 | Quality Contracts | **VALIDATED** |
| 4.1 | Fusion Telemetry | **VALIDATED — 94% FUSION COVERAGE** |
| 4.2 | Quality Metrics Engine | **TARGETED COVERAGE 100% — FULL VALIDATION PENDING** |
| 4.3 | Deterministic Quality Gate | PLANNED |
| 4.4 | ML Intelligence | PLANNED |
| 4.5 | Fault-to-QA Integration | PLANNED |
| 4.6 | Streamlit Control Center | PLANNED |

### Relatório consolidado

O histórico técnico, as entregas por fase, as evidências de validação manual, os branches/PRs, os limites conhecidos e o plano de continuidade estão documentados em:

**[ROBOQA — Engineering Status Report (2026-11-08)](docs/ROBOQA_ENGINEERING_STATUS_2026-11-08.md)**

## Componentes implementados

### Data Engine — Fase 1

- Contratos `GroundTruth` e `Measurement`.
- Arrays somente leitura e cópias defensivas.
- Fundamentos de dados sintéticos, ingestão e proveniência.
- Testes unitários e adversariais.

### Fusion Engine — Fases 2 e 4.1

- Modelo de movimento CTRV: `[x, y, psi, v, omega]`.
- Transição CTRV, Jacobiano analítico e normalização angular.
- EKF com atualização de Joseph.
- Sensores GPS, velocidade e taxa de yaw.
- Ordenação determinística de eventos e validação temporal.
- Modelo discreto de ruído de processo com zero-order hold e limites de aproximação documentados.
- Atualizações atômicas e validação da covariância.
- **Fusion Telemetry:** registros imutáveis por atualização com vetor de inovação, covariância S, NIS, sensor, timestamp e graus de liberdade.
- NIS calculado por `numpy.linalg.solve`, sem inversa matricial explícita.
- `FilterResult.telemetry` agrega evidências das atualizações bem-sucedidas.

### Fault Injection — Fases 3.1 a 3.3

- Cenários, eventos, registros de execução e proveniência imutáveis.
- Raw measurement envelopes que preservam payloads malformados sem enfraquecer o contrato normalizado `Measurement`.
- Relógios separados de aquisição, chegada e processamento.
- Operadores determinísticos com sementes: bias, outliers, delay, reordenação, duplicação, dropout, packet loss e corrupção de payload.
- Packet loss probabilístico, burst loss e distribuições uniform/exponential de atraso.
- Scheduler para cenários one-shot e recorrentes.
- Campanhas sobrepostas com composição lógica determinística.
- Testes de integração e adversariais.

**Nota semântica:** campanhas sobrepostas usam composição serial determinística; não representam execução paralela por threads ou workers assíncronos.

### Quality Metrics Engine — Fase 4.2 (cobertura direcionada 100%; validação global pendente)

- RMSE com GroundTruth explícito, interpolação e tolerância de alinhamento.
- NIS médio e norma da inovação por sensor.
- Disponibilidade baseada em contagem esperada explícita.
- Latências de transporte, processamento e ponta a ponta usando os três relógios.
- Média do traço da covariância e taxa de rejeição temporal.
- Métricas ausentes/inválidas permanecem explicitamente classificadas; não são convertidas em zero.

### Quality Contracts — Fase 4.0

- `MetricObservation`: valor, unidade, escopo, amostras e disponibilidade explícita.
- `QualityFinding`: severidade, regra, mensagem e referências de evidência.
- `QualityReport`: relatório imutável com decisão, timestamp UTC, versão de política e hashes SHA-256.
- Estados de decisão definidos: `PASS`, `PASS_WITH_WARNINGS`, `FAIL`, `BLOCKED`.
- Validação de escopos, valores finitos, identidades únicas e proveniência.

**Limite:** a Phase 4.2 calcula métricas, mas não aplica limiares estatísticos nem emite decisões PASS/FAIL. O Quality Gate e ML ainda não foram implementados.

## Próximas etapas

1. **4.2 — Quality Metrics:** validar no Colab, fechar semântica e cobertura das métricas.
2. **4.3 — Deterministic Quality Gate:** decisões auditáveis com bloqueio quando faltar evidência obrigatória.
3. **4.4 — ML Intelligence:** detecção de anomalias consultiva, após validação dos dados e métricas.
4. **4.5 — Fault-to-QA Integration:** avaliar os efeitos das falhas no pipeline completo.
5. **4.6 — Streamlit Control Center:** interface operacional de experimentos, telemetria e relatórios.

## Validação e desenvolvimento

- Validação manual em Google Colab.
- Pytest para testes unitários, integração e adversariais.
- Ruff para lint.
- Coverage como evidência complementar, com gates definidos por fase.
- Sem GitHub Actions como autoridade de validação.
- Sem treinamento, downloads automáticos de datasets/pesos ou builds pesados antes da auditoria e dos gates pertinentes.

A Phase 4.1 foi validada manualmente no Colab (suíte completa aprovada, Ruff limpo e 94% de cobertura no Fusion Engine). Na Phase 4.2, os testes direcionados de `metrics.py` alcançaram 100% de cobertura de statements e branches. A validação global da Phase 4.2 (suíte completa, Ruff e compileall) ainda está pendente.

## Governança e integridade

- Auditoria read-only antes de mudanças relevantes.
- Commits e branches incrementais, preservando histórico.
- Nenhum merge sem autorização explícita.
- Nenhum reset destrutivo, force push ou recriação silenciosa de branches.
- Não fabricar métricas, cobertura, contagens de testes ou resultados.
- Dados brutos inválidos permanecem na fronteira de envelopes; contratos de domínio normalizados continuam estritos.
- O Quality Gate determinístico será autoritativo; ML terá papel consultivo.

## Instalação e execução

```bash
pip install -e ".[dev]"
pytest -q
ruff check roboqa tests
```

Exemplo do experimento sintético do Fusion Engine:

```python
from roboqa.data.synthetic import make_synthetic_ground_truth
from roboqa.fusion import simulate_measurements, run_filter

ground_truth = make_synthetic_ground_truth(n=100, seed=42)
measurements = simulate_measurements(ground_truth, seed=42)
result = run_filter(
    measurements,
    provenance={"gt_source": ground_truth.source, "seed": 42},
)
print(result.status, result.diagnostics.n_update)
print("Telemetry records:", len(result.telemetry))
```

## Proveniência experimental

- Trajetórias podem utilizar dados reais do KITTI quando disponíveis.
- Medições sensoriais de experimentos sintéticos são simuladas.
- Resultados computacionais não equivalem à validação de sensores físicos.
- Cada experimento deve preservar configuração, sementes, hashes e evidências relevantes.

## Licença

MIT — a definir formalmente na Fase 6.
