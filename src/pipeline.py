"""
The MLOps loop: serve -> monitor -> detect drift -> retrain -> validate -> promote.

For every production batch:
  1. The current CHAMPION model predicts on the batch.
  2. True labels arrive (assumption: labels come back, e.g. loans are later repaid or not).
  3. The monitoring strategy decides whether to retrain:
        none      - never retrain (the "deploy and forget" baseline)
        periodic  - retrain every 10 batches, drift or not (common industry practice)
        PSI/KS/JS/MMD/DDM - retrain only when that drift detector raises an alarm
  4. On retrain, a CHALLENGER is trained on the most recent labelled window. Both models are
     scored on a held-out 30% of the NEWEST batch, and the challenger is PROMOTED only if it
     is at least as good (champion/challenger quality gate).
  5. Follow-up retrain: when an alarm fires, the window still contains pre-drift batches.
     So one more retrain is scheduled WINDOW_BATCHES-1 batches later, when the window
     holds only post-alarm data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .data import DRIFT_START, FEATURES, initial_training_data, stream
from .detectors import DETECTORS, make_detector

STRATEGIES = ["none", "periodic"] + DETECTORS
WINDOW_BATCHES = 3        # retrain on the last 3 batches (1,500 labelled rows)
PERIODIC_EVERY = 10       # periodic strategy retrains at batches 5, 15, 25, ...
MAX_DELAY = 10            # an alarm more than 10 batches after drift starts counts as a miss


def new_model(seed):
    return RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=1, random_state=seed)


def _retrain(champion, window_X, window_y, X_new, y_new, seed):
    """Train a challenger on the window; validate both on 30% of the newest batch."""
    Xw = pd.concat(window_X, ignore_index=True)
    yw = np.concatenate(window_y)
    newest = np.arange(len(Xw) - len(X_new), len(Xw))
    _, val_idx = train_test_split(newest, test_size=0.3, random_state=seed, stratify=y_new)
    train_mask = np.ones(len(Xw), bool)
    train_mask[val_idx] = False

    challenger = new_model(seed).fit(Xw[train_mask], yw[train_mask])
    X_val, y_val = Xw.iloc[val_idx], yw[val_idx]
    champ_acc = accuracy_score(y_val, champion.predict(X_val))
    chall_acc = accuracy_score(y_val, challenger.predict(X_val))
    return challenger, champ_acc, chall_acc, Xw


def run_stream(scenario, strategy, seed, log=None, on_promote=None):
    """Run one full stream. Returns (per_batch_rows, summary_dict, final_champion_model).

    `log` is an optional callback log(message) for live, human-readable output.
    `on_promote(model, version, batch, X_train)` is called whenever a new model goes live.
    """
    say = log or (lambda msg: None)
    X0, y0 = initial_training_data(seed)
    champion = new_model(seed).fit(X0, y0)
    version = 1

    detector = None if strategy in ("none", "periodic") else make_detector(strategy, seed)
    if detector:
        detector.set_reference(X0)

    window_X, window_y = [], []
    rows, alarms = [], []
    retrains = promotions = 0
    follow_up_at = None

    for t, X, y in stream(scenario, seed):
        pred = champion.predict(X)
        acc, f1 = accuracy_score(y, pred), f1_score(y, pred, zero_division=0)

        # ---- 1. monitoring: does this batch raise an alarm? ----
        alarm, score, detail = False, np.nan, ""
        if detector:
            alarm, score, detail = detector.check(X, y_true=y, y_pred=pred)
            if detail.startswith("feature #"):
                detail = FEATURES[int(detail[9:])]
        elif strategy == "periodic":
            alarm, detail = t % PERIODIC_EVERY == PERIODIC_EVERY // 2, "schedule"

        window_X = (window_X + [X])[-WINDOW_BATCHES:]
        window_y = (window_y + [y])[-WINDOW_BATCHES:]

        # ---- 2. retraining: on alarm, or as the scheduled follow-up ----
        follow_up = follow_up_at == t and not alarm
        promoted = False
        if alarm or follow_up:
            retrains += 1
            challenger, champ_acc, chall_acc, Xw = _retrain(champion, window_X, window_y, X, y, seed + t)
            promoted = chall_acc >= champ_acc
            if promoted:
                champion, version, promotions = challenger, version + 1, promotions + 1
                if on_promote:
                    on_promote(champion, version, t, Xw)
            if detector:
                detector.set_reference(Xw)          # drift acknowledged: new baseline
            if alarm:
                alarms.append(t)
                if detector:                         # periodic is a fixed schedule: no follow-ups
                    follow_up_at = t + WINDOW_BATCHES - 1
            why = f"ALARM ({strategy}: {detail})" if alarm else "follow-up retrain"
            say(f"batch {t:2d} | acc {acc:.3f} | {why} -> challenger {chall_acc:.3f} vs champion "
                f"{champ_acc:.3f} -> {'PROMOTED to v' + str(version) if promoted else 'kept champion'}")
        else:
            say(f"batch {t:2d} | acc {acc:.3f} | ok")

        rows.append({"scenario": scenario, "strategy": strategy, "seed": seed, "batch": t,
                     "accuracy": acc, "f1": f1, "alarm": alarm, "retrained": alarm or follow_up,
                     "promoted": promoted, "score": score, "model_version": version})

    acc = np.array([r["accuracy"] for r in rows])
    if scenario == "no_drift":
        false_alarms, clean_batches, delay = len(alarms), len(rows), np.nan
    else:
        false_alarms, clean_batches = sum(a < DRIFT_START for a in alarms), DRIFT_START
        hits = [a - DRIFT_START for a in alarms if DRIFT_START <= a <= DRIFT_START + MAX_DELAY]
        delay = hits[0] if hits else np.nan
    summary = {
        "scenario": scenario, "strategy": strategy, "seed": seed,
        "detected": not np.isnan(delay),
        "detection_delay": delay,
        "false_alarms": false_alarms,
        "false_alarms_per_100": 100 * false_alarms / clean_batches,
        "alarms": len(alarms),
        "retrains": retrains,
        "promotions": promotions,
        "acc_pre_drift": acc[:DRIFT_START].mean(),
        "acc_post_drift": acc[DRIFT_START:].mean(),
        "acc_last_10": acc[-10:].mean(),
        "f1_post_drift": float(np.mean([r["f1"] for r in rows[DRIFT_START:]])),
    }
    return rows, summary, champion
