# Plano de Melhoria — Detecção de Fraude em Transações PIX

Este documento consolida o diagnóstico do repositório e um plano de ação priorizado para
elevar o projeto ao nível de um sistema de detecção de fraude competitivo: correto,
reprodutível, sem vazamento de dados, com serving confiável e um modelo melhor alinhado
ao objetivo de negócio (priorizar os melhores casos para um time de análise com
capacidade limitada — ou seja, **Precision@K é a métrica rainha**, não F1).

---

## 1. Diagnóstico — o que existe hoje

| Componente | Estado |
|---|---|
| `notebooks/EDA.ipynb` | EDA completa (nulls, séries temporais, correlação, outliers) |
| `notebooks/Model development.ipynb` | Seleção de modelos (RF, XGBoost, AdaBoost) com under/oversampling, split temporal, análise de threshold por custo |
| `model/model_package.py` | Wrapper `FraudDetectionModel` (com bugs, ver 2.1) |
| `app/main.py` | API FastAPI de inferência (quebrada de ponta a ponta, ver 2.2) |
| `app/feature_store.py` | **Vazio** |
| `tests/` | 2 arquivos de teste, ambos quebrados (paths e schemas errados) |
| `Dockerfile` | Não builda / não roda (ver 2.4) |
| `eda_functions.py`, `sanity_functions.py` | Utilitários de EDA com muita duplicação e código não relacionado (SARIMAX etc.) |
| Artefatos | `.pkl`, `.png`, `.yaml`, model cards versionados por timestamp, commitados no git |

Métricas atuais (modelo "unshuffled" 20250214): Recall 0.92, Precision 0.20, F1 0.33,
ROC-AUC 0.978, **Precision@100 0.88**, threshold 0.7778 otimizado por custo.

---

## 2. Bugs críticos (P0 — corrigir antes de qualquer refatoração)

### 2.1 `model/model_package.py`
- **`XGBRegressor` + `predict_proba`**: `train()` instancia `xgb.XGBRegressor`, mas
  `predict()` chama `self.model.predict_proba(X)` — crash garantido. Trocar por
  `XGBClassifier` (e o problema é de classificação mesmo).
- `RandomUnderSampler` é aplicado sobre o dataset inteiro dentro de `train()`; deve
  ficar dentro de um `imblearn.Pipeline` para valer só no fit e nunca na inferência.
- `random_state=None` em tudo → resultados irreprodutíveis. Fixar seeds.

### 2.2 `app/main.py` — o endpoint `/api/v1/predict` nunca responde com sucesso
Encadeamento de erros que fazem qualquer request falhar:
1. `preprocess_transaction(transaction)` recebe um objeto Pydantic mas trata como dict —
   falta `transaction.model_dump()`.
2. `df.drop(['event_created_at', 'merchant_id'])` → `event_created_at` **não existe** no
   schema `Transaction` → `KeyError`.
3. `df.drop(columns=list_variables_to_drop)` → nenhuma dessas 7 colunas existe no input.
4. O schema `Transaction` só tem 10 features, mas o modelo foi treinado com ~94 —
   as colunas faltantes nunca são adicionadas (o TODO no código nunca foi feito).
5. O `response_model=FraudPredictionResponse` exige `is_fraudulent`, `risk_score`,
   `merchant_id`, mas o endpoint retorna `is_fraud`, `review_required` e omite os
   demais → erro de validação mesmo se a predição funcionasse.
6. `timestamp: str` no schema, mas recebe `datetime.utcnow()` (objeto, e API deprecada —
   usar `datetime.now(timezone.utc)`).
7. `df['amount'] = df['amount'] * 0.1` — escala mágica sem fonte única com o treino
   (na EDA o fator citado é /100). Qualquer divergência aqui destrói o modelo em produção
   silenciosamente.
8. `treat_missing_values` no código de serving referencia a coluna `infraction` (o
   label!) — código de treino copiado para a API; em serving não existe label.
9. Rota divergente da documentação: README e testes usam `/predict`, a API registra
   `/api/v1/predict`.
10. `class Config: schema_extra` é sintaxe Pydantic v1; em v2 é `model_config`/
    `json_schema_extra`.
11. Modelo carregado no import do módulo com path relativo hardcoded (e um path
    alternativo comentado) — usar variável de ambiente `MODEL_PATH` + evento de
    lifespan do FastAPI; o check `if model is None` hoje é código morto.
12. Threshold `0.7778` hardcoded duas vezes — mover para o YAML de config do modelo e
    carregar junto com o artefato.

### 2.3 `tests/`
- `test_model.py`: path `../models/...` (diretório real é `model/`), quebra se o pytest
  rodar da raiz; dummy input com 4 features vs ~94 do modelo.
- `unit_api.py`: payload com schema inexistente (`user_id`, `transaction_type`), rota
  `/predict` errada, e mistura teste unitário (TestClient) com script de integração que
  exige servidor rodando em localhost. Nome fora do padrão pytest (`test_*.py`).

### 2.4 `Dockerfile`
- `COPY / model`, `COPY / data`, `COPY / tests` — sintaxe sem sentido (copia a raiz do
  contexto para pastas erradas); não copia `app/`.
- `CMD uvicorn api.main:app` — o pacote é `app`, não `api`.
- `ENV MODEL_PATH=/models/...` aponta para um path que não existe e nem é lido pelo código.
- `python:3.9-slim` é incompatível com `numpy~=2.1` (exige Python ≥ 3.10). Subir para
  `python:3.12-slim`.
- Sem `.dockerignore` (imagem carregaria `.git`, notebooks, pickles antigos).

### 2.5 `readme.md`
- Estrutura de pastas descrita não corresponde ao repo (`data/`, `api/`, `config/`,
  `scripts/` não existem; `models/` vs `model/`).
- `pip install requirements.txt` → falta o `-r`.
- Rota `/predict` documentada ≠ rota real.
- Métricas no README (F1 ~0.44) divergem do model card mais recente (F1 0.33).

---

## 3. Problemas metodológicos de ML (P1 — impactam a qualidade do modelo)

### 3.1 Vazamento de dados (data leakage)
1. **Imputação orientada pelo target no dataset completo**: `treat_missing_values`
   decide a estratégia por coluna comparando a taxa de fraude entre nulos/não-nulos e
   imputa mediana **calculadas sobre o dataset inteiro (treino+teste)**. A decisão e as
   medianas devem ser aprendidas só no treino (transformer sklearn com `fit`/`transform`)
   e persistidas para o serving.
2. **CV estratificada em problema temporal**: o split principal é temporal
   (`shuffle=False`, correto), mas o `GridSearchCV` usa `StratifiedKFold`, que embaralha
   o tempo dentro do treino. Trocar por `TimeSeriesSplit` para que a validação de
   hiperparâmetros respeite a ordem temporal — fraude tem drift forte, e a CV atual
   superestima o desempenho.
3. **Features janeladas**: conferir que features `*_last_Xd` do parquet foram computadas
   apenas com dados anteriores ao evento (point-in-time correctness). Documentar essa
   garantia.

### 3.2 Consistência treino/serving (training-serving skew)
O maior risco estrutural do projeto: o pré-processamento existe **em três versões
divergentes** (notebook, `model_package.py`, `app/main.py`). Solução:
- Uma única `sklearn.Pipeline` serializada contendo: imputação → engenharia de features
  temporais → (sampler, só no fit) → modelo. O artefato salvo é a pipeline completa.
- O serving apenas monta o DataFrame de entrada e chama `pipeline.predict_proba`.
- Features derivadas do timestamp (`is_weekend`, `is_night`, `hour`...) devem ser
  computadas a partir do `event_created_at` **da transação** (enviado no payload), não
  do `datetime.now()` do servidor.

### 3.3 Melhorias de modelagem (para "competir pelo melhor modelo")
Em ordem de expectativa de ganho:

1. **Otimizar a métrica certa**: a capacidade do time é ~100 análises/dia → otimizar
   **Precision@K / Recall@K com K = capacidade diária** (ou average precision como
   proxy contínua) na seleção de modelo e de threshold, não F1. A análise de custo já
   existente no notebook (`analyze_xgboost_thresholds`) deve virar módulo versionado,
   com custo de FP/FN parametrizado, e o threshold resultante salvo no YAML do modelo.
2. **`scale_pos_weight` / class weights em vez de undersampling**: undersampling a 0.25
   descarta ~informação dos negativos e distorce as probabilidades. Gradient boosting
   lida bem com desbalanceamento via pesos; comparar (weights vs undersampling vs nada)
   com a métrica de negócio. Se mantiver sampling, calibrar probabilidades depois
   (`CalibratedClassifierCV` isotônica) — essencial porque o threshold é usado como
   corte de custo.
3. **Tuning sério**: Optuna já está no requirements e não é usado. Busca bayesiana
   (100+ trials, TimeSeriesSplit, pruning) sobre LightGBM e XGBoost supera o grid
   pequeno atual. Incluir `min_child_weight`, `subsample`, `colsample_bytree`,
   `reg_alpha/lambda`, `max_bin`.
4. **Poda de features**: ~45 das 94 features têm importância zero no config atual.
   Seleção por importância SHAP + remoção de colineares (já existe função para isso)
   → modelo menor, mais rápido (requisito de latência) e menos propenso a overfit.
5. **Feature engineering novo** (alto potencial em fraude):
   - Razões e velocidades: `amount / mc_tx_amount_succ_sum_last_30d`,
     `count_últimas_24h / count_últimos_30d` (aceleração de atividade);
   - Desvio do comportamento do merchant: z-score do `amount` vs histórico do merchant;
   - Interações com idade da conta: `amount * 1/mc_weeks_signup` (contas novas +
     valores altos é padrão clássico de fraude);
   - Flags de valor redondo (`amount % 100 == 0`) — as features `rounded_amounts` já
     sugerem que isso importa;
   - Razão declined/success (`decline_ext_rate` existe; criar em mais janelas).
6. **Ensemble supervisionado + não supervisionado**: usar o score do
   IsolationForest/ECOD (o notebook Unsupervised já explora PyOD) como **feature**
   do modelo supervisionado — captura padrões novos de fraude não rotulados.
7. **Validação final honesta**: manter um holdout temporal intocado (último mês) e
   reportar Precision@100/dia nele; avaliar estabilidade por semana para medir drift.
8. **Baseline obrigatório**: regressão logística com as top features como baseline
   reportada — dá régua de quanto o boosting agrega.

### 3.4 Artefatos e nomenclatura
- Model card diz "XGBoost", arquivo chama `ADA_model_*`, config tem parâmetros de
  AdaBoost (`algorithm: deprecated`, `estimator: null`) — os metadados salvos não
  correspondem ao modelo salvo. O `save_model_artifacts` deve extrair tipo e parâmetros
  do próprio objeto, nunca de strings fixas.
- Salvar no YAML: threshold escolhido, custo assumido de FP/FN, janela temporal de
  treino/teste, hash do dataset, versão das libs, seed.
- Tirar `.pkl`/`.png`/CSVs de resultados do git → usar DVC ou MLflow (ou, no mínimo,
  Git LFS). O repo hoje carrega 3 gerações de artefatos.

---

## 4. Refatoração de código (P1–P2)

### 4.1 Estrutura alvo do repositório
```
DS_home_test/
├── src/fraud_detection/
│   ├── __init__.py
│   ├── config.py            # paths, threshold, seeds — via pydantic-settings/env
│   ├── data.py              # load/validação de schema (pandera)
│   ├── features.py          # engenharia de features (única fonte, usada por treino e API)
│   ├── preprocessing.py     # imputer custom (fit no treino, transform no serving)
│   ├── train.py             # CLI de treino: pipeline + optuna + artefatos
│   ├── evaluate.py          # métricas de negócio, análise de threshold/custo
│   └── registry.py          # save/load de artefatos com metadados
├── app/
│   └── main.py              # só HTTP: schema, rota, chama a pipeline
├── notebooks/               # exploração; importam de src/, não duplicam lógica
├── tests/
│   ├── conftest.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_api.py          # TestClient, com modelo dummy fixture
│   └── test_train_smoke.py  # treino em amostra minúscula
├── model/                   # artefatos locais (gitignored; rastreados por DVC/MLflow)
├── pyproject.toml           # substitui requirements.txt; deps de api/train separadas
├── Dockerfile               # multi-stage, python:3.12-slim, só deps de serving
└── README.md                # corrigido e alinhado à estrutura real
```

### 4.2 Limpezas pontuais
- **`eda_functions.py`**: `corr_func`, `corr_movel`, `corr_deslocada`,
  `plotMovingAverage` e MAPE estão **duplicadas literalmente 2x no mesmo arquivo**
  (~350 linhas redundantes). Deduplicar; mover o que é de série temporal/SARIMAX (não
  usado no case) para fora ou deletar.
- **`sanity_functions.py`**: `iqr()` não calcula IQR — usa média±k·desvio (o código de
  IQR real está comentado). Renomear ou corrigir; `standard()` faz min-max, não
  padronização — renomear para `min_max_scale`. Imports dentro de função (`barplot`)
  → topo do módulo.
- **`app/feature_store.py`** vazio: implementar de fato (cache das features `mc_*` por
  merchant, já que o cliente da API não vai enviar 90 agregados) **ou remover**. Para o
  case, documentar a premissa: em produção a API consultaria uma feature store por
  `merchant_id`; no teste, aceitar as features no payload.
- **Notebooks**: fixar seeds, limpar células mortas/comentadas, extrair funções longas
  (`analyze_xgboost_thresholds`, `save_model_artifacts`, `treat_missing_values`) para
  `src/` e importar. Numerar (`1_EDA`, `2_modeling`, ...) como o README já promete.
- Typos em arquivos rastreados: `oulier_standarization_analysis.csv`,
  `model_selection_results_unshufled.csv`.

### 4.3 API de inferência (reescrita do `app/main.py`)
- Carregar pipeline + threshold + metadados via `lifespan`, path por env var.
- Schema `Transaction` gerado a partir do `feature_names` do artefato (ou validado
  contra ele no startup) — hoje o schema e o modelo estão dessincronizados.
- Resposta: `transaction_id`, `fraud_probability`, `review_required` (score ≥ threshold),
  `model_version`, `threshold`, `timestamp` — schema e retorno idênticos.
- Adicionar: logging estruturado de cada predição (input hash, score, versão — insumo
  para monitoramento), endpoint `/metrics` ou log para drift, tratamento de erro que
  não vaza stack trace (`str(e)` hoje vai para o cliente).
- Latência: mensurar de fato (o README alega ~100ms sem evidência); com pipeline leve e
  poda de features, a meta razoável é <20ms p99. Evitar recriar DataFrame/objetos por
  request no hot path.

### 4.4 Qualidade e automação (P2)
- `pyproject.toml` + `ruff` (lint+format) + `mypy` no `src/` + `pre-commit`.
- CI (GitHub Actions): lint → testes → build da imagem → smoke test do endpoint.
- Testes que faltam e importam: paridade treino/serving (mesma linha de dados pela
  pipeline de treino e pela API produz o mesmo score), teste de schema do artefato,
  teste do threshold aplicado, propriedade "sem `infraction` no serving".

---

## 5. Extensões de produto (P3 — diferenciais para o case)

1. **Fila priorizada em vez de flag binária**: como a restrição é capacidade de
   analistas, entregar um endpoint/relatório "top-K transações do dia por score ×
   valor em risco" (`expected_loss = P(fraude) × amount`) — maximiza R$ recuperado por
   hora de analista, argumento de negócio mais forte que threshold fixo.
2. **Monitoramento de drift**: PSI/KS das features e do score semana a semana (fraude
   muda rápido); alarme para retreino.
3. **Loop de feedback**: os vereditos dos analistas viram labels novos → retreino
   periódico versionado (base para a discussão de MLOps na apresentação).
4. **Explicabilidade por predição**: top-3 razões SHAP na resposta da API — analistas
   de fraude precisam saber *por que* revisar (o notebook SHAP já existe; produtizar).
5. **Regras duras complementares**: features de sanções/PEP têm sinal regulatório —
   transações com contraparte sancionada podem ir direto para revisão, independente do
   score (camada de regras antes do modelo).

---

## 6. Roadmap sugerido

| Fase | Escopo | Resultado |
|---|---|---|
| **P0 (1–2 dias)** | §2: consertar API, Dockerfile, testes, README, `model_package.py` | Projeto roda de ponta a ponta: `docker build && run` + request de exemplo retorna predição |
| **P1 (2–4 dias)** | §3.1–3.2 + §4.1–4.3: pipeline única sem leakage, TimeSeriesSplit, estrutura `src/`, API reescrita, artefatos consistentes | Modelo retreinado sem leakage com métricas honestas; paridade treino/serving testada |
| **P2 (3–5 dias)** | §3.3 + §4.4: Optuna, class weights, calibração, novas features, ensemble com score não supervisionado, CI | Ganho mensurável de Precision@100 no holdout temporal; qualidade automatizada |
| **P3 (contínuo)** | §5: fila priorizada por expected loss, drift, feedback loop, SHAP na API | Solução de negócio completa, além do modelo |

**Critério de sucesso do modelo**: superar o baseline atual de Precision@100 = 0.88 no
holdout temporal com métricas obtidas *sem* os vazamentos atuais (o número atual está
inflado pela imputação com informação do teste e pela CV embaralhada) — e reportar
também o valor monetário em risco capturado no top-100 diário.
