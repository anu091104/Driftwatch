"""
Synthetic loan-default data stream with controlled, labelled drift.

Why synthetic?  To measure *detection delay* and *false alarms* we must know
exactly when drift starts. Real datasets don't tell you that, so drift research
(Gama et al., 2014; Lu et al., 2018) uses generators with injected drift.

Features (an applicant asking for a loan):
    age, income, loan_amount, credit_score, debt_ratio, employment_years
Target:
    default (1 = the loan will not be repaid)

Drift scenarios (drift begins at batch DRIFT_START):
    no_drift          - stationary stream; any alarm is a FALSE alarm
    sudden_covariate  - economic shock: incomes fall, debt rises, credit scores drop (P(X) changes at once)
    gradual_covariate - the same shock, phased in linearly over GRADUAL_SPAN batches
    concept           - a policy change: inputs look identical, but the rule linking inputs to
                        default changes (P(y|X) changes, P(X) does not)
"""
import numpy as np
import pandas as pd

FEATURES = ["age", "income", "loan_amount", "credit_score", "debt_ratio", "employment_years"]
SCENARIOS = ["no_drift", "sudden_covariate", "gradual_covariate", "concept"]

N_BATCHES = 60          # batches in the production stream
BATCH_SIZE = 500        # applications per batch (think: one day of traffic)
DRIFT_START = 30        # first drifted batch
GRADUAL_SPAN = 20       # gradual drift ramps from batch 30 to 49


def _features(rng, n, shock=0.0):
    """Sample applicants. `shock` in [0, 1] = how far the economic shock has progressed."""
    age = np.clip(rng.normal(40, 10, n), 21, 70)
    income = np.exp(rng.normal(10.8 - 0.25 * shock, 0.4, n))               # median ~49k -> ~38k
    loan_amount = income * rng.uniform(0.1, 0.6, n) * (1 + 0.2 * shock)    # people borrow more
    credit_score = np.clip(rng.normal(680 - 40 * shock, 60, n), 300, 850)
    debt_ratio = rng.beta(2 + 1.5 * shock, 5, n)                           # mean 0.29 -> 0.41
    employment_years = np.minimum(age - 21, rng.exponential(6, n))
    return pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "credit_score": credit_score, "debt_ratio": debt_ratio,
        "employment_years": employment_years,
    })


def _default_prob(X, concept="old"):
    """True (hidden) rule that decides who defaults."""
    lti = X["loan_amount"] / X["income"]                      # loan-to-income
    if concept == "old":
        z = (-1.0
             + 6.0 * (X["debt_ratio"] - 0.3)
             + 3.0 * (lti - 0.35)
             - 0.02 * (X["credit_score"] - 680)
             + 1.5 * (X["credit_score"] < 600))
    else:
        # New lending policy / new customer segment: credit score barely matters,
        # young applicants with short employment are now the main risk.
        z = (-1.0
             + 2.0 * (X["debt_ratio"] - 0.3)
             - 0.004 * (X["credit_score"] - 680)
             + 2.5 * (X["age"] < 32)
             - 0.35 * (X["employment_years"] - 4)
             + 4.0 * (lti - 0.35))
    return 1 / (1 + np.exp(-2.0 * z))


def make_batch(rng, n, shock=0.0, concept="old"):
    X = _features(rng, n, shock)
    y = (rng.random(n) < _default_prob(X, concept)).astype(int)
    return X, y


def drift_state(scenario, t):
    """Return (shock, concept) for batch index t of a scenario."""
    if scenario == "no_drift" or t < DRIFT_START:
        return 0.0, "old"
    if scenario == "sudden_covariate":
        return 1.0, "old"
    if scenario == "gradual_covariate":
        return min(1.0, (t - DRIFT_START + 1) / GRADUAL_SPAN), "old"
    if scenario == "concept":
        return 0.0, "new"
    raise ValueError(f"unknown scenario {scenario}")


def initial_training_data(seed, n=3000):
    """Historical data the first model is trained on (always pre-drift)."""
    return make_batch(np.random.default_rng(seed + 10_000), n)


def stream(scenario, seed):
    """Yield (t, X, y) for each production batch."""
    rng = np.random.default_rng(seed)
    for t in range(N_BATCHES):
        shock, concept = drift_state(scenario, t)
        X, y = make_batch(rng, BATCH_SIZE, shock, concept)
        yield t, X, y
