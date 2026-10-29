"""
Five drift detectors behind one interface.

Data-distribution detectors (compare the new batch's FEATURES with a reference set,
no labels needed):
    PSI  - Population Stability Index, the credit-risk industry standard (alarm if > 0.2)
    KS   - Kolmogorov-Smirnov two-sample test per feature (Bonferroni-corrected p < 0.05)
    JS   - Jensen-Shannon distance between binned distributions (alarm if > 0.1, Evidently default)
    MMD  - Maximum Mean Discrepancy, a multivariate kernel test with a permutation p-value
           (Gretton et al., 2012); looks at all features jointly

Performance detector (needs true labels):
    DDM  - Drift Detection Method (Gama et al., 2004): watches the model's error rate and
           alarms when it rises 3 standard deviations above its best observed level
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


class Detector:
    name = "base"
    needs_labels = False

    def set_reference(self, X_ref):
        """Called at start and whenever a new model is trained (the reference = its training data)."""
        self.X_ref = np.asarray(X_ref, dtype=float)

    def check(self, X, y_true=None, y_pred=None):
        """Return (drift: bool, score: float, detail: str)."""
        raise NotImplementedError


def _bins(ref_col, n_bins=10):
    edges = np.unique(np.quantile(ref_col, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _hist(col, edges):
    counts = np.histogram(col, edges)[0].astype(float)
    return (counts + 1e-4) / (counts.sum() + 1e-4 * len(counts))


class PSIDetector(Detector):
    name = "PSI"

    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def check(self, X, y_true=None, y_pred=None):
        X = np.asarray(X, dtype=float)
        scores = []
        for j in range(X.shape[1]):
            edges = _bins(self.X_ref[:, j])
            p, q = _hist(self.X_ref[:, j], edges), _hist(X[:, j], edges)
            scores.append(np.sum((q - p) * np.log(q / p)))
        j = int(np.argmax(scores))
        return scores[j] > self.threshold, float(scores[j]), f"feature #{j}"


class KSDetector(Detector):
    name = "KS"

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def check(self, X, y_true=None, y_pred=None):
        X = np.asarray(X, dtype=float)
        pvals = [ks_2samp(self.X_ref[:, j], X[:, j]).pvalue for j in range(X.shape[1])]
        j = int(np.argmin(pvals))
        corrected_alpha = self.alpha / X.shape[1]          # Bonferroni: we run one test per feature
        return pvals[j] < corrected_alpha, float(pvals[j]), f"feature #{j}"


class JSDetector(Detector):
    name = "JS"

    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def check(self, X, y_true=None, y_pred=None):
        X = np.asarray(X, dtype=float)
        scores = []
        for j in range(X.shape[1]):
            edges = _bins(self.X_ref[:, j])
            scores.append(jensenshannon(_hist(self.X_ref[:, j], edges), _hist(X[:, j], edges)))
        j = int(np.argmax(scores))
        return scores[j] > self.threshold, float(scores[j]), f"feature #{j}"


class MMDDetector(Detector):
    name = "MMD"

    def __init__(self, alpha=0.05, n_perm=50, max_samples=250, seed=0):
        self.alpha, self.n_perm, self.max_samples = alpha, n_perm, max_samples
        self.rng = np.random.default_rng(seed)

    def set_reference(self, X_ref):
        super().set_reference(X_ref)
        self.mu, self.sd = self.X_ref.mean(0), self.X_ref.std(0) + 1e-9   # standardise with ref stats

    def check(self, X, y_true=None, y_pred=None):
        X = np.asarray(X, dtype=float)
        a = self.X_ref[self.rng.choice(len(self.X_ref), min(self.max_samples, len(self.X_ref)), replace=False)]
        b = X[self.rng.choice(len(X), min(self.max_samples, len(X)), replace=False)]
        Z = (np.vstack([a, b]) - self.mu) / self.sd
        sq = np.sum(Z ** 2, 1)
        D = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
        gamma = 1.0 / np.median(D[D > 0])                   # median heuristic bandwidth
        K = np.exp(-gamma * D)
        n = len(a)

        def mmd(idx):
            i, k = idx[:n], idx[n:]
            return K[np.ix_(i, i)].mean() + K[np.ix_(k, k)].mean() - 2 * K[np.ix_(i, k)].mean()

        observed = mmd(np.arange(len(Z)))
        null = [mmd(self.rng.permutation(len(Z))) for _ in range(self.n_perm)]
        p = (1 + np.sum(np.array(null) >= observed)) / (1 + self.n_perm)
        return p < self.alpha, float(p), "joint distribution"


class DDMDetector(Detector):
    name = "DDM"
    needs_labels = True

    def __init__(self, warn=2.0, drift=3.0, min_samples=500):
        self.k_drift, self.min_samples = drift, min_samples
        self.reset()

    def reset(self):
        self.n, self.errors = 0, 0
        self.p_min, self.s_min = np.inf, np.inf

    def set_reference(self, X_ref):
        super().set_reference(X_ref)
        self.reset()                                         # new model -> new error baseline

    def check(self, X, y_true=None, y_pred=None):
        self.errors += int(np.sum(np.asarray(y_true) != np.asarray(y_pred)))
        self.n += len(y_true)
        p = self.errors / self.n
        s = np.sqrt(p * (1 - p) / self.n)
        if self.n >= self.min_samples and p + s < self.p_min + self.s_min:
            self.p_min, self.s_min = p, s
        drift = self.n >= 2 * self.min_samples and p + s > self.p_min + self.k_drift * self.s_min
        return bool(drift), float(p), "error rate"


def make_detector(name, seed=0):
    return {
        "PSI": lambda: PSIDetector(),
        "KS": lambda: KSDetector(),
        "JS": lambda: JSDetector(),
        "MMD": lambda: MMDDetector(seed=seed),
        "DDM": lambda: DDMDetector(),
    }[name]()


DETECTORS = ["PSI", "KS", "JS", "MMD", "DDM"]
