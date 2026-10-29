"""data.gravity

Poisson pseudo-maximum-likelihood (PPML) with origin and destination fixed
effects: the standard estimator for gravity models of migration and trade
(Santos Silva & Tenreyro, 2006; Beine, Bertoli & Fernández-Huertas Moraga, 2016).

Unlike OLS on log stock it keeps the zero corridors and is consistent under
heteroskedasticity. The two sets of fixed effects absorb everything specific to
an origin or a destination (size, income, policy, "multilateral resistance"),
so the coefficients measure the pure effect of pair attributes such as distance
or a shared language.

The fixed effects are concentrated out: given β, each set has a closed-form
update, and β takes Newton steps on the profiled likelihood. This avoids
building a design matrix with hundreds of dummy columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PPMLResult:
    coef: pd.Series
    se: pd.Series
    n_obs: int
    n_dropped: int
    pseudo_r2: float
    iterations: int
    fitted: pd.Series | None = None   # E[y] in original units, indexed like the input rows

    def table(self) -> pd.DataFrame:
        z = self.coef / self.se
        return pd.DataFrame({
            "term": self.coef.index,
            "coefficient": self.coef.values,
            "std_error": self.se.values,
            "p_value": 2 * stats.norm.sf(np.abs(z.values)),
            "ci_lower_95": (self.coef - 1.96 * self.se).values,
            "ci_upper_95": (self.coef + 1.96 * self.se).values,
        })


def _demean(X: np.ndarray, w: np.ndarray, g1: np.ndarray, g2: np.ndarray,
            tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Weighted alternating projections: remove both sets of group means from X."""
    X = X.copy()
    w1 = np.bincount(g1, weights=w)
    w2 = np.bincount(g2, weights=w)
    for _ in range(max_iter):
        prev = X.copy()
        for g, wg in ((g1, w1), (g2, w2)):
            means = np.column_stack([np.bincount(g, weights=w * X[:, k]) for k in range(X.shape[1])])
            X -= (means / wg[:, None])[g]
        if np.max(np.abs(X - prev)) < tol:
            break
    return X


def _drop_all_zero_groups(y, g1, g2):
    """Groups whose outcome is all zero have an FE of -inf; drop them (iteratively)."""
    keep = np.ones(len(y), dtype=bool)
    while True:
        s1 = np.bincount(g1[keep], weights=y[keep], minlength=g1.max() + 1)
        s2 = np.bincount(g2[keep], weights=y[keep], minlength=g2.max() + 1)
        new = keep & (s1[g1] > 0) & (s2[g2] > 0)
        if new.sum() == keep.sum():
            return keep
        keep = new


def ppml_fe(
    df: pd.DataFrame,
    y: str,
    x: list[str],
    fe: tuple[str, ...] = ("origin_iso3", "dest_iso3"),
    cluster: str | None = None,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> PPMLResult:
    """
    Fit E[y] = exp(Xβ + α_origin + γ_dest) by PPML.

    Pass a single column in `fe` for one set of fixed effects (e.g. origin ×
    year, so the model explains how each origin's total splits across
    destinations). Standard errors are heteroskedasticity-robust, clustered by
    `cluster` when given (e.g. an unordered country-pair id, so i→j and j→i
    share a cluster).
    """
    cols = list(dict.fromkeys([y, *x, *fe] + ([cluster] if cluster else [])))
    data = df[cols].dropna()

    def codes(frame):
        g1 = pd.factorize(frame[fe[0]])[0]
        g2 = pd.factorize(frame[fe[1]])[0] if len(fe) > 1 else np.zeros(len(frame), dtype=int)
        return g1, g2

    yv = data[y].to_numpy(float)
    g1, g2 = codes(data)
    keep = _drop_all_zero_groups(yv, g1, g2)
    n_dropped = int((~keep).sum())
    data = data[keep]
    yv = yv[keep]
    g1, g2 = codes(data)
    X = data[x].to_numpy(float)

    # Scaling y leaves β unchanged (the FE absorb it) and keeps exp() well behaved.
    scale = yv.mean()
    yv = yv / scale
    sy1 = np.bincount(g1, weights=yv)
    sy2 = np.bincount(g2, weights=yv)

    beta = np.zeros(X.shape[1])
    a = np.zeros(g1.max() + 1)
    b = np.zeros(g2.max() + 1)

    def solve_fe(xb, a, b):
        for _ in range(500):
            a_new = np.log(sy1) - np.log(np.bincount(g1, weights=np.exp(xb + b[g2])))
            b_new = np.log(sy2) - np.log(np.bincount(g2, weights=np.exp(xb + a_new[g1])))
            done = max(np.max(np.abs(a_new - a)), np.max(np.abs(b_new - b))) < 1e-12
            a, b = a_new, b_new
            if done:
                break
        return a, b

    def loglik(beta, a, b):
        eta = X @ beta + a[g1] + b[g2]
        return float(np.sum(yv * eta - np.exp(eta)))

    a, b = solve_fe(X @ beta, a, b)
    ll = loglik(beta, a, b)
    it = 0
    for it in range(1, max_iter + 1):
        mu = np.exp(X @ beta + a[g1] + b[g2])
        Xt = _demean(X, mu, g1, g2)
        score = Xt.T @ (yv - mu)
        H = Xt.T @ (Xt * mu[:, None])
        step = np.linalg.solve(H, score)
        # Step-halving keeps the likelihood monotone
        for _ in range(30):
            beta_new = beta + step
            a_new, b_new = solve_fe(X @ beta_new, a, b)
            ll_new = loglik(beta_new, a_new, b_new)
            if ll_new >= ll - 1e-12:
                break
            step /= 2
        beta, a, b, ll = beta_new, a_new, b_new, ll_new
        if np.max(np.abs(step)) < tol:
            break

    mu = np.exp(X @ beta + a[g1] + b[g2])
    Xt = _demean(X, mu, g1, g2)
    H_inv = np.linalg.inv(Xt.T @ (Xt * mu[:, None]))
    scores = Xt * (yv - mu)[:, None]
    if cluster:
        cl = pd.factorize(data[cluster])[0]
        G = cl.max() + 1
        S = np.column_stack([np.bincount(cl, weights=scores[:, k], minlength=G) for k in range(X.shape[1])])
        meat = S.T @ S * G / (G - 1)
    else:
        n = len(yv)
        meat = scores.T @ scores * n / (n - 1)
    V = H_inv @ meat @ H_inv

    pseudo_r2 = float(np.corrcoef(yv, mu)[0, 1] ** 2)
    return PPMLResult(
        coef=pd.Series(beta, index=x),
        se=pd.Series(np.sqrt(np.diag(V)), index=x),
        n_obs=len(yv),
        n_dropped=n_dropped,
        pseudo_r2=pseudo_r2,
        iterations=it,
        fitted=pd.Series(mu * scale, index=data.index),
    )


def ols_fe(df: pd.DataFrame, y: str, x: list[str], fe=("origin_iso3", "dest_iso3"),
           cluster: str | None = None) -> PPMLResult:
    """
    OLS on log(y) with the same fixed effects, dropping zero corridors: the
    traditional log-linear gravity estimator, kept for comparison with PPML.
    """
    data = df[df[y] > 0][[y, *x, *fe] + ([cluster] if cluster else [])].dropna()
    g1 = pd.factorize(data[fe[0]])[0]
    g2 = pd.factorize(data[fe[1]])[0]
    w = np.ones(len(data))
    Z = _demean(np.column_stack([np.log(data[y].to_numpy(float)), data[x].to_numpy(float)]), w, g1, g2)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX_inv = np.linalg.inv(Xt.T @ Xt)
    beta = XtX_inv @ Xt.T @ yt
    resid = yt - Xt @ beta
    scores = Xt * resid[:, None]
    if cluster:
        cl = pd.factorize(data[cluster])[0]
        G = cl.max() + 1
        S = np.column_stack([np.bincount(cl, weights=scores[:, k], minlength=G) for k in range(Xt.shape[1])])
        meat = S.T @ S * G / (G - 1)
    else:
        meat = scores.T @ scores
    V = XtX_inv @ meat @ XtX_inv
    r2_within = 1 - resid @ resid / (yt @ yt)
    return PPMLResult(
        coef=pd.Series(beta, index=x), se=pd.Series(np.sqrt(np.diag(V)), index=x),
        n_obs=len(data), n_dropped=int((df[y] <= 0).sum()), pseudo_r2=float(r2_within), iterations=1,
    )
