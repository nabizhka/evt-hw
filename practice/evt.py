"""MF731 HW 10 (10_extreme_practice): VaR of a $1M S&P 500 position by four methods.

Linearized loss L = -V X. Fits are run on losses in percent, which is only a change of
units (mu, sigma and beta scale, xi does not) and keeps scipy's optimizer well conditioned.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import genextreme, genpareto, norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
V, LAM, THETA, INIT = 1_000_000.0, 0.97, 0.97, 100
END, EWMA_DATE = "2021-12-28", "2021-11-01"
BLOCKS = [(73, 35), (35, 73)]
U_LEVEL = 0.95
ALPHAS = np.linspace(0.99, 0.9999, 100)
TOP = 0.9999
B, SEED = 1000, 731


def load():
    p = pd.read_csv(os.path.join(ROOT, "data", "gspc_close.csv"), index_col=0, parse_dates=True)["GSPC"]
    p = p.loc[:END]
    x = np.log(p).diff().dropna()
    assert len(p) == 2556 and len(x) == 2555, (len(p), len(x))
    return p, x


def var_empirical(loss, alpha):
    """inf{l : F_N(l) >= alpha} = order statistic ceil(N alpha)."""
    s = np.sort(loss)
    k = np.ceil(len(s) * np.asarray(alpha) - 1e-9).astype(int)
    return s[k - 1]


def ewma(x, stop=None):
    """Sample mean / variance of the first 100 returns, then the course recursion.

    Returns mu, s2 for the day after the last return used.
    """
    r = x.loc[:stop].values if stop else x.values
    mu, s2 = r[:INIT].mean(), r[:INIT].var(ddof=1)
    for xs in r[INIT:]:
        d = xs - mu
        mu = LAM * mu + (1 - LAM) * xs
        s2 = THETA * s2 + (1 - THETA) * d**2
    return mu, s2


def var_ewma(mu, s2, alpha):
    return V * (-mu + np.sqrt(s2) * norm.ppf(alpha))


def block_maxima(loss_pct, n, m):
    assert n * m == len(loss_pct)
    return loss_pct[: n * m].reshape(m, n).max(axis=1)


def fit_gev(maxima):
    """scipy's shape c is -xi."""
    c, mu, sig = genextreme.fit(maxima)
    return -c, mu, sig


def var_gev(xi, mu, sig, n, alpha):
    """Percent units. VaR = mu - (sigma/xi)(1 - (-n log alpha)^(-xi))."""
    a = -n * np.log(alpha)
    if abs(xi) < 1e-9:
        return mu - sig * np.log(a)
    return mu - sig / xi * (1 - a ** (-xi))


def fit_gp(exc):
    c, _, beta = genpareto.fit(exc, floc=0)
    return c, beta


def var_gp(xi, beta, u, alpha, fu=U_LEVEL):
    """Percent units. VaR = u + (beta/xi)(((1-alpha)/(1-F(u)))^(-xi) - 1)."""
    r = (1 - alpha) / (1 - fu)
    if abs(xi) < 1e-9:
        return u - beta * np.log(r)
    return u + beta / xi * (r ** (-xi) - 1)


# ------------------------------------------------------- hand-written likelihoods (checks)
def nll_gev(p, m):
    xi, mu, sig = p
    if sig <= 0:
        return 1e12
    z = (m - mu) / sig
    if abs(xi) < 1e-9:
        return len(m) * np.log(sig) + np.sum(z + np.exp(-z))
    w = 1 + xi * z
    if np.any(w <= 0):
        return 1e12
    return len(m) * np.log(sig) + (1 + 1 / xi) * np.sum(np.log(w)) + np.sum(w ** (-1 / xi))


def nll_gp(p, y):
    xi, beta = p
    if beta <= 0:
        return 1e12
    if abs(xi) < 1e-9:
        return len(y) * np.log(beta) + y.sum() / beta
    w = 1 + xi * y / beta
    if np.any(w <= 0):
        return 1e12
    return len(y) * np.log(beta) + (1 + 1 / xi) * np.sum(np.log(w))


def refit(nll, starts, data):
    best = None
    for s in starts:
        r = minimize(nll, s, args=(data,), method="Nelder-Mead",
                     options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 20000, "maxfev": 20000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def bootstrap(lp):
    """Resample each fit's own input: the block maxima, or the excesses over a fixed u.

    Resampling single days instead would break up volatility clusters and move crash days
    into other blocks, which biases the GEV fits upward relative to the data actually used.
    """
    rng = np.random.default_rng(SEED)
    rows = {}
    for n, m in BLOCKS:
        mx = block_maxima(lp, n, m)
        v = []
        for _ in range(B):
            xi, mu, sig = fit_gev(rng.choice(mx, m, replace=True))
            v.append(var_gev(xi, mu, sig, n, TOP))
        rows[f"gev{n}"] = v
    u = var_empirical(lp, U_LEVEL)
    y = lp[lp > u] - u
    v = []
    for _ in range(B):
        xi, beta = fit_gp(rng.choice(y, len(y), replace=True))
        v.append(var_gp(xi, beta, u, TOP))
    rows["gp"] = v
    return pd.DataFrame(rows)


def run(do_boot=True):
    p, x = load()
    lp = -100 * x.values                       # loss in percent of V
    usd = V / 100
    out = {"p": p, "x": x, "lp": lp, "alphas": ALPHAS, "curves": {}, "checks": {}}

    # (1) empirical
    out["curves"]["emp"] = var_empirical(lp, ALPHAS) * usd

    # (2) EWMA normal
    mu, s2 = ewma(x, EWMA_DATE)
    out["ewma"] = (mu, s2)
    out["curves"]["ewma"] = var_ewma(mu, s2, ALPHAS)
    mu_l, s2_l = ewma(x)
    out["ewma_last"] = (mu_l, s2_l, var_ewma(mu_l, s2_l, TOP))

    # (3) GEV
    out["gev"] = {}
    for n, m in BLOCKS:
        mx = block_maxima(lp, n, m)
        xi, mu_g, sig = fit_gev(mx)
        out["gev"][n] = dict(xi=xi, mu=mu_g, sig=sig, m=m, nll=nll_gev((xi, mu_g, sig), mx), maxima=mx)
        out["curves"][f"gev{n}"] = var_gev(xi, mu_g, sig, n, ALPHAS) * usd
        starts = [(xi, mu_g, sig), (0.1, mx.mean(), mx.std()), (0.4, np.median(mx), mx.std() / 2)]
        r = refit(nll_gev, starts, mx)
        a = TOP
        v = var_gev(xi, mu_g, sig, n, a)
        inv = brentq(lambda q: genextreme.cdf(q, -xi, mu_g, sig) ** (1 / n) - a, mx.min(), 1e4)
        out["checks"][f"gev{n}"] = dict(xi2=r.x[0], mu2=r.x[1], sig2=r.x[2], dnll=out["gev"][n]["nll"] - r.fun,
                                        var2=var_gev(*r.x, n, a) * usd, var=v * usd, inv=inv * usd)

    # (4) GP
    u = var_empirical(lp, U_LEVEL)
    y = lp[lp > u] - u
    xi, beta = fit_gp(y)
    out["gp"] = dict(xi=xi, beta=beta, u=u, nu=len(y), nll=nll_gp((xi, beta), y), y=y)
    out["curves"]["gp"] = var_gp(xi, beta, u, ALPHAS) * usd
    r = refit(nll_gp, [(xi, beta), (0.1, y.mean()), (0.4, y.mean() / 2)], y)
    v = var_gp(xi, beta, u, TOP)
    inv = brentq(lambda q: U_LEVEL + (1 - U_LEVEL) * genpareto.cdf(q - u, xi, 0, beta) - TOP, u, 1e4)
    out["checks"]["gp"] = dict(xi2=r.x[0], beta2=r.x[1], dnll=out["gp"]["nll"] - r.fun,
                               var2=var_gp(*r.x, u, TOP) * usd, var=v * usd, inv=inv * usd,
                               at_u=var_gp(xi, beta, u, U_LEVEL))

    # GP sensitivity to the threshold, with F(u) the level that defines u
    sens = {}
    for lev in (0.90, 0.95, 0.975):
        uu = var_empirical(lp, lev)
        yy = lp[lp > uu] - uu
        a, b = fit_gp(yy)
        sens[lev] = dict(u=uu * usd, nu=len(yy), xi=a, beta=b * usd / 1, var=var_gp(a, b, uu, TOP, lev) * usd)
    out["gp_sens"] = sens

    # fitting on raw dollars, as a naive script would
    n, m = BLOCKS[0]
    c, l, s = genextreme.fit(block_maxima(lp, n, m) * usd)
    out["gev_usd"] = dict(xi=-c, var=var_gev(-c, l, s, n, TOP))
    c, l, s = genextreme.fit(block_maxima(lp, n, m) / 100)
    out["gev_dec"] = dict(xi=-c, var=var_gev(-c, l, s, n, TOP) * V)

    if do_boot:
        out["boot"] = bootstrap(lp) * usd
    return out
