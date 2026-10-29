# DriftWatch — Drift Detection & Automated Retraining for ML in Production

A research-oriented MLOps project that compares **five drift detectors** and **two baseline policies** inside a complete **monitor → detect → retrain → validate → promote** pipeline, with MLflow tracking, a model registry, a FastAPI serving layer and CI.

**Research question:** *Which drift detectors detect data drift and concept drift fastest, with the fewest false alarms, and which retraining policy recovers model accuracy best?*

---

## Quick start (Windows)

1. Put this folder on your **D drive** (for example `D:\Projects\driftwatch`). Everything installs inside the folder.
2. Double-click **`run.bat`**. It will:
   - install the packages (first time only, 3–5 minutes),
   - run the **live demo** (watch drift hit and the model recover, batch by batch),
   - run the **full benchmark** (140 simulated streams, 2–5 minutes),
   - open **`results/report.html`** in your browser.
3. Double-click **`run_api.bat`**. The model API opens at http://127.0.0.1:8000/docs.
4. Double-click **`mlflow_ui.bat`**. The MLflow dashboard opens at http://127.0.0.1:5000.

You need Python 3.10 or newer. If `run.bat` says Python isn't found, install it from python.org and tick **"Add python.exe to PATH"**.

<details><summary>Manual commands (any OS)</summary>

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.demo                # live demo (default: concept drift + DDM)
python -m src.demo --scenario sudden_covariate --detector PSI
python -m src.experiment          # full benchmark (add --quick for 2 seeds)
uvicorn api.main:app --reload     # API -> http://127.0.0.1:8000/docs
mlflow ui --backend-store-uri sqlite:///mlflow.db
pytest -q                         # tests
```
</details>

---

## 1. Problem statement

A deployed model is trained on yesterday's data, and the world keeps changing. Incomes fall, customer segments shift, and policies change. Model accuracy then decays silently, a problem called **drift**. There are two kinds:

| Type | What changes | Example |
|---|---|---|
| **Covariate (data) drift** | the inputs, P(X) | an economic shock lowers incomes and credit scores |
| **Concept drift** | the input→output rule, P(y\|X) | a new lending policy changes who defaults, while the applicants look the same |

MLOps teams must decide **how to detect drift** and **when to retrain**. Retraining too rarely loses accuracy, and retraining too often wastes compute and risks pushing bad models. This project measures those trade-offs.

## 2. Objectives

1. Build an automated pipeline: serve → monitor → detect → retrain → champion/challenger validation → promote.
2. Benchmark 5 detectors (PSI, KS, Jensen–Shannon, MMD, DDM) against 2 baselines (no retraining, periodic retraining).
3. Measure **detection delay, detection rate, false alarms, retraining cost and accuracy recovery** across 4 controlled drift scenarios.
4. Make it production-shaped: experiment tracking and model registry (MLflow), a model API (FastAPI), tests and CI (GitHub Actions), and a container (Docker).

## 3. Methodology

**Data.** A synthetic loan-default stream with 6 features: age, income, loan amount, credit score, debt ratio and employment years. It is synthetic because measuring *detection delay* requires knowing exactly when drift starts. Real datasets don't record that, so drift research uses controlled generators. Each stream has **60 batches × 500 applications**, and drift starts at **batch 30**.

**Scenarios.**

| Scenario | Description |
|---|---|
| `no_drift` | Control. Every alarm is a false alarm. |
| `sudden_covariate` | Incomes −22%, debt ratio 0.29→0.41, credit score −40, all at once. |
| `gradual_covariate` | The same shock, phased in over 20 batches. |
| `concept` | The default rule changes (age and employment now dominate, credit score barely matters); inputs are unchanged. |

**Detectors.**

| Detector | Type | Alarm rule |
|---|---|---|
| PSI (Population Stability Index) | features, per column | PSI > 0.2 (credit-industry standard) |
| KS (Kolmogorov–Smirnov) | features, per column | p < 0.05 / 6 (Bonferroni) |
| JS (Jensen–Shannon distance) | features, per column | distance > 0.1 (Evidently default) |
| MMD (Maximum Mean Discrepancy) | features, **multivariate** | permutation p < 0.05 |
| DDM (Drift Detection Method, Gama 2004) | **model error rate** (needs labels) | error > min + 3σ |

**Pipeline.** A Random Forest (100 trees) serves each batch. When true labels arrive, the strategy decides whether to retrain:

- A **challenger** is trained on the last 3 batches.
- The challenger and the current **champion** are both scored on 30% of the newest batch. The challenger is **promoted only if it is at least as good** (a quality gate).
- After an alarm, one **follow-up retrain** runs when the window holds only post-alarm data.
- Each detector's reference distribution resets to the new training data.

**Evaluation.** 4 scenarios × 7 strategies × 5 seeds = **140 runs**. A drift counts as *caught* if an alarm fires within 10 batches of its start.

## 4. Results

All numbers are means over 5 seeds (`results/summary.csv`).

| Strategy | Concept drift: caught | Concept drift: accuracy after drift | Gradual covariate: delay | Wasted retrains (no drift) |
|---|---|---|---|---|
| No retraining | — | 0.629 | — | 0 |
| Periodic (every 10) | 100% | 0.808 | 5.0 | 6.0 |
| PSI | **0%** | 0.629 | 8.6 | 0 |
| KS test | 40% | 0.743 | **1.4** | 7.2 |
| Jensen–Shannon | **0%** | 0.629 | 4.0 | 0 |
| MMD | 20% | 0.693 | 3.4 | 4.4 |
| **DDM** | **100% (0.2 batches)** | **0.833** | 9.5 | **0** |

**Key findings**

1. **Feature-based detectors are blind to concept drift.** PSI and JS never fired. Their models stayed at 62.9% accuracy, the same as never retraining. KS and MMD "caught" it in 40% and 20% of runs, but that matches their false-alarm rates, so those hits are chance.
2. **DDM caught concept drift in every run within 0.2 batches, with zero false alarms.** It recovered accuracy from 0.629 to **0.833**, the best of all strategies and 2.5 points above periodic retraining, while running 2 retrains instead of 6.
3. **Not all drift is harmful.** Every feature detector flagged covariate drift instantly, but retraining improved accuracy by under 1 point (0.845 → 0.852). DDM stayed mostly quiet in these scenarios, which is the right call because the model was still working.
4. **Sensitive detectors cost compute.** In the no-drift control, KS triggered 7.2 needless retrains per stream and MMD 4.4. PSI, JS and DDM triggered none.
5. **Recommendation:** use a **hybrid**. PSI or JS gives a label-free early warning that the inputs have moved. DDM, once labels arrive, decides whether retraining is actually needed.

## 5. Conclusion

Drift monitoring is not one problem. Data-distribution tests answer *"has the input changed?"*, which is cheap, needs no labels and is good for early warning. Performance monitoring answers *"is the model getting worse?"*, which is what actually justifies retraining. A production pipeline needs both, plus a champion/challenger gate so a retrain can never push a worse model live.

**Limitations and future work:**
- The data is synthetic. Next steps are validating on a real stream with timestamps (e.g. the Lending Club loans data) and handling delayed labels.
- Add adaptive window sizes (ADWIN).
- Add carbon/compute cost per retrain (CodeCarbon).

---

## Project structure

```
driftwatch/
├── run.bat / run_api.bat / mlflow_ui.bat   one-click Windows launchers
├── src/
│   ├── data.py          drift-injected data stream (4 scenarios)
│   ├── detectors.py     PSI, KS, JS, MMD, DDM
│   ├── pipeline.py      monitor -> detect -> retrain -> validate -> promote
│   ├── experiment.py    140-run benchmark, charts, report, MLflow logging
│   └── demo.py          live single-stream demo + MLflow model registry
├── api/main.py          FastAPI: /predict, /model, /drift/check, /drift/simulate
├── tests/               unit tests (pytest)
├── results/             report.html, charts, CSVs
├── .github/workflows/   CI: tests + quick benchmark + quality gate
└── Dockerfile           containerised API
```

## MLOps components

| Concern | Implementation |
|---|---|
| Experiment tracking | MLflow: every run with per-batch accuracy, parameters and summary metrics |
| Model registry | MLflow: each promoted model becomes a new version of `loan-default-model` |
| Model serving | FastAPI with Swagger docs |
| Monitoring | 5 drift detectors; `/drift/check` endpoint for live batches |
| Automated retraining | detector-triggered with a champion/challenger quality gate |
| CI/CD | GitHub Actions runs tests, a benchmark, and an accuracy regression gate |
| Packaging | Docker image for the API |

## References

- Gama, J. et al. (2004). *Learning with Drift Detection.* SBIA. (DDM)
- Gama, J. et al. (2014). *A Survey on Concept Drift Adaptation.* ACM Computing Surveys.
- Gretton, A. et al. (2012). *A Kernel Two-Sample Test.* JMLR. (MMD)
- Lu, J. et al. (2018). *Learning under Concept Drift: A Review.* IEEE TKDE.
- Rabanser, S. et al. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS.
- Yurdakul, B. (2018). *Statistical Properties of Population Stability Index.* (PSI)
