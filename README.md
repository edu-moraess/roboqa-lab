# RoboQA Lab

**Plataforma modular de Quality Assurance para sistemas de fusão sensorial e estimação de estado.**

**ArqTech Labs — Engineering Program**

RoboQA Lab não é apenas uma demonstração de EKF. O objetivo é construir uma plataforma de engenharia para testar, observar e avaliar pipelines de percepção/estimação sensorial sob dados normais, falhas controladas e condições temporais adversas.

> **Status do projeto:** implementação em andamento — Fases 1, 2.2, 3.1–3.3-E, 4.0, 4.1 e 4.2 validadas manualmente; Fase 4.3 também passou por pytest completo, Ruff e compileall no Colab. Fase 4.4-A está VALIDADA manualmente no Colab (30 testes direcionados, suíte completa, Ruff e compileall). Fase 4.5-A passou pela validação manual dos contratos de integração (52 testes direcionados, suíte completa, Ruff e compileall). Fase 4.5-B tem replay temporal determinístico VALIDADO manualmente no Colab (19 testes direcionados, suíte completa, Ruff e compileall). Fase 4.5-C tem adaptador estrito VALIDADO manualmente no Colab (10 testes direcionados, suíte completa, Ruff e compileall). Fase 4.5-D1 (15 testes direcionados) e D2 (32 testes direcionados; suíte completa, Ruff e compileall) também estão VALIDADA(S) manualmente. O produto completo ainda não está concluído.

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
| 4.2 | Quality Metrics Engine | **VALIDATED — FULL SUITE, RUFF, COMPILEALL; 100% TARGETED COVERAGE** |
| 4.3 | Deterministic Quality Gate | **VALIDATED — PYTEST, RUFF, COMPILEALL** |
| 4.4-A | ML Feature/Dataset Contracts | **VALIDATED — TARGETED + FULL SUITE, RUFF, COMPILEALL** |
| 4.5-A | Fault-to-QA Integration Contracts | **VALIDATED — 52 TARGETED TESTS, FULL SUITE, RUFF, COMPILEALL** |
| 4.5-B | Deterministic transport replay, watermark, deduplication and packet accounting | **VALIDATED — 19 TARGETED TESTS, FULL SUITE, RUFF, COMPILEALL** |
| 4.5-C | Strict Measurement adapter from accepted transport envelopes | **VALIDATED — 10 TARGETED TESTS, FULL SUITE, RUFF, COMPILEALL** |
| 4.5-D1 | Deterministic event-time Fusion scheduling boundary | **VALIDATED — 15 TARGETED TESTS, FULL SUITE, RUFF, COMPILEALL** |
| 4.5-D2 | Identity-preserving execution through Fusion runner | **VALIDATED — 32 TARGETED TESTS, FULL SUITE, RUFF, COMPILEALL** |
| 4.5-D3 | Replay/execution evidence propagation into Quality Metrics and Gate | PLANNED |
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

### Quality Metrics Engine — Fase 4.2 (VALIDATED)

- RMSE com GroundTruth explícito, interpolação e tolerância de alinhamento.
- NIS médio e norma da inovação por sensor.
- Disponibilidade baseada em contagem esperada explícita.
- Latências de transporte, processamento e ponta a ponta usando os três relógios.
- Média do traço da covariância e taxa de rejeição temporal.
- Métricas ausentes/inválidas permanecem explicitamente classificadas; não são convertidas em zero.

### Deterministic Quality Gate — Fase 4.3 (VALIDATED)

- Avaliação determinística de métricas por política versionada e explícita.
- Regras identificam métrica, escopo, sensor/janela e unidade.
- Limites rígidos geram FAIL; bandas de alerta geram PASS_WITH_WARNINGS.
- Evidência obrigatória ausente, inválida ou com unidade incompatível gera BLOCKED.
- Relatório imutável preserva métricas, findings, decisão, versão da política e hashes fornecidos.
- Nenhum limiar universal é presumido; ML não participa da decisão.

### ML Intelligence — Fase 4.4-A (VALIDATED)

- Contratos imutáveis para valores de features, amostras e datasets versionados.
- Disponibilidade AVAILABLE/MISSING/INVALID explícita; sem imputação ou sentinelas numéricas.
- Separação TRAIN/VALIDATION/TEST por execução, bloqueando vazamento de janelas do mesmo run.
- Schema de features e unidades consistente entre amostras.
- Anotações NORMAL/ANOMALOUS/UNKNOWN mantidas como metadados de avaliação, fora das features.
- Sem treinamento, inferência, downloads ou alteração da autoridade do Quality Gate.
- Validação manual no Colab: 30 testes direcionados; suíte completa sem falhas reportadas; Ruff limpo; compileall sem erros. A saída não expôs uma contagem agregada inequívoca da suíte.
- Especificação: [Phase 4.4-A — ML Feature/Dataset Contracts](docs/phase-4-4a-ml-feature-dataset-contracts.md).

### Quality Contracts — Fase 4.0

- `MetricObservation`: valor, unidade, escopo, amostras e disponibilidade explícita.
- `QualityFinding`: severidade, regra, mensagem e referências de evidência.
- `QualityReport`: relatório imutável com decisão, timestamp UTC, versão de política e hashes SHA-256.
- Estados de decisão definidos: `PASS`, `PASS_WITH_WARNINGS`, `FAIL`, `BLOCKED`.
- Validação de escopos, valores finitos, identidades únicas e proveniência.

**Limite:** a Phase 4.3 aplica somente regras explicitamente configuradas. Integração fault-to-QA, ML consultivo e Streamlit ainda não foram implementados.

## Próximas etapas

1. **4.4-B — Feature Engineering:** extrair features temporais a partir de telemetria e evidências validadas.
2. **4.4-C — Anomaly Baselines:** baseline estatístico e detector consultivo, sem autoridade sobre o Quality Gate.
3. **4.4-D/E — Evaluation and Advisory Report:** avaliação por cenário/run, calibração, explicabilidade e relatório independente.
4. **4.5 — Fault-to-QA Integration:** avaliar os efeitos das falhas no pipeline completo.
5. **4.6 — Streamlit Control Center:** interface operacional de experimentos, telemetria e relatórios.


### Phase 4.5-C — Measurement Adapter (VALIDATED)

- Adapts only replay-accepted envelopes into the strict normalized Measurement contract.
- Requires explicit caller-supplied covariance R per sensor; no inferred noise model.
- Maps acquisition_time to Measurement.t and preserves arrival/processing clocks and source identity as adapter metadata.
- Validates sensor registry, dimensions, finite values, and covariance through the existing contracts.
- Does not change run_filter or claim arrival-order Fusion execution.
- Manual Colab validation: 10 adapter tests passed; full pytest exit code 0 (aggregate count not exposed); Ruff clean; compileall exit code 0.
- Validation corrections were limited to a case-insensitive test assertion and import ordering; production adapter behavior was unchanged.
- Specification: [Phase 4.5-C — Strict Measurement Adapter](docs/phase-4-5c-measurement-adapter.md).

## Validação e desenvolvimento

- Validação manual em Google Colab.
- Pytest para testes unitários, integração e adversariais.
- Ruff para lint.
- Coverage como evidência complementar, com gates definidos por fase.
- Sem GitHub Actions como autoridade de validação.
- Sem treinamento, downloads automáticos de datasets/pesos ou builds pesados antes da auditoria e dos gates pertinentes.

A Phase 4.1 foi validada manualmente no Colab (suíte completa aprovada, Ruff limpo e 94% de cobertura no Fusion Engine). A Phase 4.2 foi validada manualmente: suíte completa aprovada, Ruff limpo, compileall sem erros e 100% de cobertura direcionada de statements e branches em `metrics.py`. A Phase 4.3 passou pela suíte completa de pytest, Ruff e compileall no Colab. A Fase 4.4-A foi validada manualmente no Colab: 30 testes direcionados, suíte completa sem falhas reportadas, Ruff limpo e compileall sem erros. A Fase 4.5-A foi validada manualmente: 52 testes direcionados, suíte completa sem falhas, Ruff limpo e compileall sem erros; a contagem agregada da suíte não foi inferida. A Fase 4.5-B foi validada manualmente no Colab: 19 testes direcionados passaram; a suíte completa terminou com exit code 0; Ruff direcionado e completo e compileall direcionado e completo passaram. A saída da suíte completa não expôs uma contagem agregada inequívoca, portanto nenhum total foi inferido. A validação cobre o replay offline, não a integração com o Fusion Engine.

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


Phase 4.5-C was manually validated in Colab on code checkpoint `77f0e6e267f08f86ddc7f1468da0b9d474d348ef`: 10 targeted adapter tests passed; the full pytest suite exited 0 without reported failures (aggregate count not exposed); Ruff passed; compileall exited 0. The adapter does not change `run_filter` scheduling or establish arrival-order Fusion execution.


### Phase 4.5-D1 — Deterministic Fusion Scheduling (VALIDATED)

- Adds an immutable `FusionSchedule` boundary between the accepted transport replay and the existing monotonic Fusion runner.
- Verifies a one-to-one match between replay-accepted envelope identities and adapted measurements.
- Validates sensor identity, acquisition/arrival/processing clocks, and normalized payload.
- Uses acquisition/event time for offline batch execution order, with deterministic sensor/source/sequence tie-breakers.
- Schedule SHA-256 binds replay provenance, clocks, payload, and caller-supplied covariance R.
- Replay summary remains attached, preserving duplicate, late, invalid, and dropped delivery accounting.
- Manual Colab validation: **15 targeted tests passed**; full pytest completed at 100%, exit code 0; full Ruff passed; full compileall exit code 0. The full-suite output did not expose an aggregate test count, so none is inferred.
- This is an offline batch schedule, not a live arrival-order scheduler. EKF and `run_filter` remain unchanged; no rewind/fixed-lag reprocessing is implemented.
- Specification: [Phase 4.5-D1 — Deterministic Fusion Scheduling](docs/phase-4-5d1-fusion-scheduling-contract.md).


### Phase 4.5-D2 — Identity-preserving Fusion execution (VALIDATED)

- Executes the immutable D1 schedule through the existing monotonic Fusion runner.
- Produces one terminal outcome per replay-accepted identity: UPDATED, FAILED, or NOT_EXECUTED.
- Associates successful InnovationTelemetry with the exact sensor/source/sequence identity and acquisition timestamp.
- Preserves the complete transport replay summary, runner diagnostics, configuration, state/covariance/time traces, and deterministic execution SHA-256.
- Numerical update failure is attributed to the failing scheduled item; remaining entries are explicitly NOT_EXECUTED. Invalid run configuration does not blame an individual measurement.
- No EKF mathematics, runner policy, Quality Metrics, or Quality Gate behavior was changed.
- Manual Colab validation: **32 targeted tests passed** across execution, scheduling, and adapter; full pytest reached 100% with exit code 0; full Ruff passed; full compileall exited 0. The full-suite output did not expose an unambiguous aggregate count.
- Validated branch checkpoint: `43de077cf01b16c3114d5f7dfeeeaf53a6d2057d`; subsequent changes in this documentation checkpoint do not alter production code.
- Specification: [Phase 4.5-D2 — Identity-preserving Fusion execution](docs/phase-4-5d2-identity-preserving-fusion-execution.md).
