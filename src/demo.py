"""
Live demo of the MLOps loop on ONE stream, printed batch by batch.

    python -m src.demo                                   # concept drift, DDM detector
    python -m src.demo --scenario sudden_covariate --detector PSI

Every promoted model is
  * saved to models/champion.joblib (the file the FastAPI service serves), and
  * registered in the MLflow Model Registry as "loan-default-model" (new version each time).
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib

from .data import FEATURES, SCENARIOS, initial_training_data
from .pipeline import STRATEGIES, run_stream

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def save_champion(model, version, batch, X_train, scenario, strategy):
    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "champion.joblib")
    X_train.to_csv(MODELS / "reference.csv", index=False)       # what the model was trained on
    info = {"version": version, "promoted_at_batch": batch, "scenario": scenario, "strategy": strategy,
            "features": FEATURES, "trained_rows": len(X_train), "saved_at": datetime.now().isoformat(timespec="seconds")}
    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="concept", choices=SCENARIOS)
    ap.add_argument("--detector", default="DDM", choices=STRATEGIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()

    mlflow = None
    if not a.no_mlflow:
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
            import mlflow
            import mlflow.sklearn
            mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
            mlflow.set_experiment("driftwatch-live-demo")
        except ImportError:
            mlflow = None

    print(f"\n=== DriftWatch live demo | scenario: {a.scenario} | monitoring: {a.detector} ===")
    print("Drift is injected at batch 30. Watch accuracy, alarms, retraining and promotions.\n")

    X0, _ = initial_training_data(a.seed)

    def on_promote(model, version, batch, X_train):
        save_champion(model, version, batch, X_train, a.scenario, a.detector)
        if mlflow:
            kwargs = dict(name=f"model-v{version}", input_example=X_train.head(3),
                          registered_model_name="loan-default-model")
            try:
                try:   # newer MLflow saves with skops and needs the tree type marked as trusted (we built it)
                    mlflow.sklearn.log_model(model, skops_trusted_types=["sklearn.tree._tree.Tree"], **kwargs)
                except TypeError:
                    mlflow.sklearn.log_model(model, **kwargs)
            except Exception as e:
                print(f"   (MLflow model registration skipped: {e.__class__.__name__})")

    run_ctx = mlflow.start_run(run_name=f"{a.scenario}-{a.detector}") if mlflow else None
    try:
        # version 1 = the model trained on historical data
        from .pipeline import new_model
        save_champion(new_model(a.seed).fit(*initial_training_data(a.seed)), 1, -1, X0, a.scenario, a.detector)
        rows, summary, _ = run_stream(a.scenario, a.detector, a.seed, log=print, on_promote=on_promote)
        if mlflow:
            mlflow.log_params({"scenario": a.scenario, "detector": a.detector, "seed": a.seed})
            for r in rows:
                mlflow.log_metric("accuracy", r["accuracy"], step=r["batch"])
            mlflow.log_metrics({k: float(v) for k, v in summary.items()
                                if isinstance(v, (int, float)) and v == v and not isinstance(v, bool)})
    finally:
        if run_ctx:
            mlflow.end_run()

    print("\n--- summary ---")
    for k in ["detection_delay", "false_alarms", "retrains", "promotions", "acc_pre_drift", "acc_post_drift", "acc_last_10"]:
        v = summary[k]
        print(f"{k:16s} {v:.3f}" if isinstance(v, float) else f"{k:16s} {v}")
    print(f"\nLatest champion saved to {MODELS / 'champion.joblib'} (served by the API).")


if __name__ == "__main__":
    main()
