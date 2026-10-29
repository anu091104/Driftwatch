"""Fast sanity tests, run automatically by GitHub Actions on every push."""
import numpy as np
import pandas as pd

from src.data import DRIFT_START, make_batch, drift_state
from src.detectors import DETECTORS, make_detector
from src.pipeline import run_stream


def _ref_and_shifted():
    rng = np.random.default_rng(0)
    ref, _ = make_batch(rng, 2000)
    same, _ = make_batch(rng, 500)
    shifted, _ = make_batch(rng, 500, shock=1.0)
    return ref, same, shifted


def test_feature_detectors_flag_covariate_shift():
    ref, _, shifted = _ref_and_shifted()
    for name in ["PSI", "KS", "JS", "MMD"]:
        det = make_detector(name)
        det.set_reference(ref)
        assert det.check(shifted)[0], f"{name} missed an obvious shift"


def test_feature_detectors_quiet_without_shift():
    ref, same, _ = _ref_and_shifted()
    for name in ["PSI", "JS"]:            # threshold detectors: should never fire on identical distributions
        det = make_detector(name)
        det.set_reference(ref)
        assert not det.check(same)[0], f"{name} false alarm"


def test_ddm_flags_error_increase():
    det = make_detector("DDM")
    det.set_reference(pd.DataFrame(np.zeros((10, 6))))
    y = np.zeros(500, int)
    for _ in range(5):                    # 5 batches at ~10% error
        pred = y.copy(); pred[:50] = 1
        assert not det.check(None, y, pred)[0]
    pred = y.copy(); pred[:250] = 1       # error jumps to 50%
    assert det.check(None, y, pred)[0]


def test_drift_state_schedule():
    assert drift_state("concept", DRIFT_START - 1) == (0.0, "old")
    assert drift_state("concept", DRIFT_START) == (0.0, "new")
    assert drift_state("gradual_covariate", DRIFT_START)[0] < 1.0


def test_pipeline_recovers_from_concept_drift():
    _, s_none, _ = run_stream("concept", "none", 0)
    _, s_ddm, _ = run_stream("concept", "DDM", 0)
    assert s_ddm["detected"]
    assert s_ddm["acc_last_10"] > s_none["acc_last_10"] + 0.1


def test_all_detectors_registered():
    assert set(DETECTORS) == {"PSI", "KS", "JS", "MMD", "DDM"}
