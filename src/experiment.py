"""
Benchmark: every scenario x every strategy x several random seeds.

Writes to results/:
    runs.csv       one row per (scenario, strategy, seed) with the summary metrics
    batches.csv    per-batch accuracy for every run
    summary.csv    metrics averaged over seeds
    *.png          charts
    report.html    everything in one page (open it in a browser)
and, if MLflow is installed, logs every run to mlflow.db (view with `mlflow ui`).
"""
import argparse
import base64
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import BATCH_SIZE, DRIFT_START, GRADUAL_SPAN, N_BATCHES, SCENARIOS
from .pipeline import MAX_DELAY, STRATEGIES, WINDOW_BATCHES, run_stream

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SCENARIO_LABELS = {
    "no_drift": "No drift (control)",
    "sudden_covariate": "Sudden covariate drift",
    "gradual_covariate": "Gradual covariate drift",
    "concept": "Concept drift",
}
STRATEGY_LABELS = {
    "none": "No retraining", "periodic": "Periodic (every 10)",
    "PSI": "PSI", "KS": "KS test", "JS": "Jensen-Shannon", "MMD": "MMD", "DDM": "DDM",
}

# ---- chart styling (validated categorical palette, recessive axes) ----
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = {"none": "#8a8984", "periodic": "#eb6834", "PSI": "#2a78d6", "DDM": "#1baf7a"}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _one(scenario, strategy, seed):
    rows, summary, _ = run_stream(scenario, strategy, seed)
    return rows, summary


def run_all(seeds, n_jobs=2):
    jobs = [(sc, st, s) for s in seeds for sc in SCENARIOS for st in STRATEGIES]
    print(f"Running {len(jobs)} simulations ({len(SCENARIOS)} scenarios x {len(STRATEGIES)} strategies x "
          f"{len(seeds)} seeds) using {n_jobs} parallel worker(s) ...")
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(*j) for j in jobs)
    print(f"Done in {time.time() - t0:.0f}s")
    batches = pd.DataFrame([r for rows, _ in out for r in rows])
    runs = pd.DataFrame([s for _, s in out])
    return runs, batches


def summarise(runs):
    g = runs.groupby(["scenario", "strategy"])
    summary = g.agg(
        detection_rate=("detected", "mean"),
        mean_delay=("detection_delay", "mean"),
        false_alarms_per_100=("false_alarms_per_100", "mean"),
        retrains=("retrains", "mean"),
        promotions=("promotions", "mean"),
        acc_post_drift=("acc_post_drift", "mean"),
        acc_post_drift_std=("acc_post_drift", "std"),
        acc_last_10=("acc_last_10", "mean"),
        f1_post_drift=("f1_post_drift", "mean"),
    ).reset_index()
    order = {s: i for i, s in enumerate(STRATEGIES)}
    summary["_o"] = summary["scenario"].map({s: i for i, s in enumerate(SCENARIOS)}) * 10 + summary["strategy"].map(order)
    return summary.sort_values("_o").drop(columns="_o")


# ------------------------------------------------------------------ charts
def chart_accuracy_over_time(batches, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True)
    shown = ["none", "periodic", "PSI", "DDM"]
    for ax, sc in zip(axes.flat, SCENARIOS):
        if sc != "no_drift":
            end = DRIFT_START + (GRADUAL_SPAN if sc == "gradual_covariate" else 0)
            ax.axvspan(DRIFT_START - 0.5, max(end, DRIFT_START + 0.5) - 0.5, color="#f0efec", zorder=0)
            ax.axvline(DRIFT_START - 0.5, color=MUTED, lw=1, ls="--", zorder=1)
            ax.text(DRIFT_START + 0.3, 0.505, "drift starts", color=INK2, fontsize=8.5)
        for st in shown:
            d = batches[(batches.scenario == sc) & (batches.strategy == st)].groupby("batch")["accuracy"].mean()
            ax.plot(d.index, d.values, color=SERIES[st], lw=2, label=STRATEGY_LABELS[st],
                    zorder=3 if st != "none" else 2)
        if sc == "concept":
            ax.annotate("PSI never fires here, so it\nsits on top of 'No retraining'", xy=(48, 0.625),
                        xytext=(41, 0.70), fontsize=8.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.set_title(SCENARIO_LABELS[sc], loc="left")
        ax.set_ylim(0.5, 0.95)
    for ax in axes[1]:
        ax.set_xlabel("Production batch (500 loan applications each)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (mean over seeds)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(matrix, row_labels, col_labels, fmt, title, path, cmap_vals, low_is_good, note=None, texts=None):
    fig, ax = plt.subplots(figsize=(8.6, (0.62 if texts else 0.48) * len(row_labels) + 1.6))
    vals = matrix.values.astype(float)
    finite = vals[np.isfinite(vals)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", cmap_vals)
    norm_vals = (vals - lo) / (hi - lo + 1e-12)
    if low_is_good:
        norm_vals = 1 - norm_vals
    shown = np.where(np.isfinite(vals), norm_vals, np.nan)
    ax.imshow(np.ma.masked_invalid(shown), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if texts is not None and texts[i][j]:
                txt, dark_bg = texts[i][j], np.isfinite(v) and shown[i, j] > 0.55
            elif np.isfinite(v):
                txt, dark_bg = fmt(v), shown[i, j] > 0.55
            else:
                txt, dark_bg = "missed", False
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec"))
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="white" if dark_bg else INK)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.set_title(title, loc="left", pad=10)
    if note:
        fig.text(0.01, 0.01, note, color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_detection(summary, runs, path):
    """Colour = expected delay, counting a miss as MAX_DELAY+1 batches; text = mean delay when caught + hit rate."""
    drifted = [s for s in SCENARIOS if s != "no_drift"]
    dets = [s for s in STRATEGIES if s != "none"]
    r = runs[runs.scenario.isin(drifted)].copy()
    r["effective_delay"] = r["detection_delay"].fillna(MAX_DELAY + 1)
    eff = r.groupby(["strategy", "scenario"])["effective_delay"].mean().unstack().loc[dets, drifted]
    s = summary.set_index(["strategy", "scenario"])
    texts = []
    for st in dets:
        row = []
        for sc in drifted:
            rate, d = s.loc[(st, sc), "detection_rate"], s.loc[(st, sc), "mean_delay"]
            row.append("missed\n0% of runs" if rate == 0 else f"{d:.1f} batches\n{rate:.0%} of runs")
        texts.append(row)
    _heatmap(eff, [STRATEGY_LABELS[x] for x in dets], [SCENARIO_LABELS[x] for x in drifted],
             lambda v: f"{v:.1f}", "Detection delay and hit rate (darker = caught sooner and more reliably)", path,
             BLUES[:6], low_is_good=True, texts=texts,
             note=f"A run counts as 'caught' if an alarm fires within {MAX_DELAY} batches of the drift. Colour = mean delay with misses counted as {MAX_DELAY + 1}.")


def chart_accuracy_table(summary, path):
    m = summary.pivot(index="strategy", columns="scenario", values="acc_post_drift").loc[STRATEGIES, SCENARIOS]
    _heatmap(m, [STRATEGY_LABELS[s] for s in STRATEGIES], [SCENARIO_LABELS[s] for s in SCENARIOS],
             lambda v: f"{v:.3f}", "Mean accuracy after the drift point (higher is better)", path,
             BLUES[:6], low_is_good=False)


def chart_cost(summary, path):
    d = summary[summary.scenario == "no_drift"].set_index("strategy").loc[STRATEGIES]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d["retrains"], color="#2a78d6", height=0.6)
    for yi, v in zip(y, d["retrains"]):
        ax.text(v + 0.15, yi, f"{v:.1f}", va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(y, [STRATEGY_LABELS[s] for s in d.index])
    ax.set_xlabel("Retraining jobs per 60-batch stream")
    ax.set_title("Wasted retraining when nothing has changed (no-drift control)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ MLflow
def log_to_mlflow(runs, batches):
    try:
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
        import mlflow
        from mlflow.entities import Metric, Param
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("MLflow not installed - skipping experiment tracking.")
        return
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    exp = mlflow.set_experiment("driftwatch-benchmark")
    client = MlflowClient()
    print(f"Logging {len(runs)} runs to MLflow ...")
    for _, r in runs.iterrows():
        run = client.create_run(exp.experiment_id, run_name=f"{r.scenario}/{r.strategy}/seed{r.seed}",
                                tags={"scenario": r.scenario, "strategy": r.strategy})
        b = batches[(batches.scenario == r.scenario) & (batches.strategy == r.strategy) & (batches.seed == r.seed)]
        ts = int(time.time() * 1000)
        metrics = [Metric("accuracy", float(a), ts, int(t)) for t, a in zip(b.batch, b.accuracy)]
        metrics += [Metric("model_version", float(v), ts, int(t)) for t, v in zip(b.batch, b.model_version)]
        for k in ["detection_delay", "false_alarms", "retrains", "promotions", "acc_pre_drift",
                  "acc_post_drift", "acc_last_10", "f1_post_drift"]:
            if pd.notna(r[k]):
                metrics.append(Metric(k, float(r[k]), ts, 0))
        params = [Param("scenario", r.scenario), Param("strategy", r.strategy), Param("seed", str(r.seed)),
                  Param("window_batches", str(WINDOW_BATCHES)), Param("batch_size", str(BATCH_SIZE))]
        for i in range(0, len(metrics), 900):
            client.log_batch(run.info.run_id, metrics=metrics[i:i + 900], params=params if i == 0 else [])
        client.set_terminated(run.info.run_id)
    print("MLflow logging done. View with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


# ------------------------------------------------------------------ report
def _img(path):
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{path.stem}">'


def write_report(summary, seeds, path):
    t = summary.copy()
    t["scenario"] = t["scenario"].map(SCENARIO_LABELS)
    t["strategy"] = t["strategy"].map(STRATEGY_LABELS)
    t["detection_rate"] = (t["detection_rate"] * 100).round(0).astype(int).astype(str) + "%"
    t["mean_delay"] = t["mean_delay"].map(lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    for c in ["false_alarms_per_100", "retrains", "promotions"]:
        t[c] = t[c].map(lambda v: f"{v:.1f}")
    for c in ["acc_post_drift", "acc_last_10", "f1_post_drift"]:
        t[c] = t[c].map(lambda v: f"{v:.3f}")
    t["acc_post_drift"] = t["acc_post_drift"] + " ± " + summary["acc_post_drift_std"].map(lambda v: f"{v:.3f}").values
    t = t.drop(columns="acc_post_drift_std")
    t.columns = ["Scenario", "Strategy", "Detected", "Delay (batches)", "False alarms /100 batches",
                 "Retrains", "Promotions", "Accuracy after drift", "Accuracy last 10", "F1 after drift"]
    table = t.to_html(index=False, border=0, classes="tbl")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>DriftWatch Results</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}}
h1{{margin-bottom:4px}} .sub{{color:#52514e;margin-top:0}} img{{max-width:100%;border:1px solid #e6e5e1;border-radius:8px;margin:10px 0 24px;background:#fff}}
.tbl{{border-collapse:collapse;font-size:13px;width:100%}} .tbl th,.tbl td{{padding:6px 8px;border-bottom:1px solid #e6e5e1;text-align:left}}
.tbl th{{background:#f0efec}} .wrap{{overflow-x:auto}}
</style></head><body>
<h1>DriftWatch: drift detection &amp; automated retraining benchmark</h1>
<p class="sub">{len(SCENARIOS)} scenarios × {len(STRATEGIES)} strategies × {len(seeds)} seeds = {len(SCENARIOS)*len(STRATEGIES)*len(seeds)} simulated
production streams of {N_BATCHES} batches × {BATCH_SIZE} loan applications. Drift starts at batch {DRIFT_START}.</p>
<h2>1. Model accuracy over time</h2>{_img(RESULTS / 'accuracy_over_time.png')}
<h2>2. How fast does each detector notice?</h2>{_img(RESULTS / 'detection_delay.png')}
<h2>3. Accuracy after drift, per strategy</h2>{_img(RESULTS / 'accuracy_after_drift.png')}
<h2>4. The cost side: unnecessary retraining</h2>{_img(RESULTS / 'retraining_cost.png')}
<h2>5. Full results table</h2><div class="wrap">{table}</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run the DriftWatch benchmark")
    ap.add_argument("--seeds", type=int, default=5, help="random seeds per configuration (default 5)")
    ap.add_argument("--quick", action="store_true", help="2 seeds, for a fast test run")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    ap.add_argument("--jobs", type=int, default=2,
                    help="parallel worker processes (default 2; use 1 on low-RAM laptops, -1 for all cores)")
    args = ap.parse_args()
    seeds = list(range(2 if args.quick else args.seeds))

    RESULTS.mkdir(exist_ok=True)
    runs, batches = run_all(seeds, n_jobs=args.jobs)
    summary = summarise(runs)
    runs.to_csv(RESULTS / "runs.csv", index=False)
    batches.to_csv(RESULTS / "batches.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)

    chart_accuracy_over_time(batches, RESULTS / "accuracy_over_time.png")
    chart_detection(summary, runs, RESULTS / "detection_delay.png")
    chart_accuracy_table(summary, RESULTS / "accuracy_after_drift.png")
    chart_cost(summary, RESULTS / "retraining_cost.png")
    write_report(summary, seeds, RESULTS / "report.html")
    if not args.no_mlflow:
        log_to_mlflow(runs, batches)

    pd.set_option("display.width", 200)
    print("\n" + summary[["scenario", "strategy", "detection_rate", "mean_delay", "false_alarms_per_100",
                          "retrains", "acc_post_drift"]].round(3).to_string(index=False))
    print(f"\nReport written to {RESULTS / 'report.html'}")


if __name__ == "__main__":
    main()
