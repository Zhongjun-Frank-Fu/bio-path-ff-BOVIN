"""Shared log-rank helper used by tools/ scripts (can't import from tools/)."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def logrank_2group(time_a, event_a, time_b, event_b) -> dict:
    ta = np.asarray(time_a, dtype=float)
    ea = np.asarray(event_a, dtype=int)
    tb = np.asarray(time_b, dtype=float)
    eb = np.asarray(event_b, dtype=int)
    all_events = np.sort(np.unique(np.concatenate([ta[ea == 1], tb[eb == 1]])))

    obs_a = exp_a = var_a = 0.0
    for t in all_events:
        at_risk_a = (ta >= t).sum()
        at_risk_b = (tb >= t).sum()
        at_risk = at_risk_a + at_risk_b
        if at_risk == 0:
            continue
        d_a = ((ta == t) & (ea == 1)).sum()
        d_b = ((tb == t) & (eb == 1)).sum()
        d = d_a + d_b
        if d == 0:
            continue
        e_a = at_risk_a * d / at_risk
        obs_a += d_a
        exp_a += e_a
        if at_risk > 1:
            var_a += (at_risk_a * at_risk_b * d * (at_risk - d)) / (at_risk ** 2 * (at_risk - 1))

    if var_a == 0:
        return {"chi2": 0.0, "p": 1.0, "obs_a": obs_a, "exp_a": exp_a}
    z2 = (obs_a - exp_a) ** 2 / var_a
    return {"chi2": z2, "p": float(chi2.sf(z2, 1)), "obs_a": obs_a, "exp_a": exp_a}
