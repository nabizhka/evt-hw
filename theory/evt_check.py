"""Numerical checks for MF731 HW 10 (10_extreme_theory_pub). The proofs do not depend on these."""
import os
import numpy as np
from scipy.stats import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)

# Q1: F^n is 1 or at most a^n, whatever the normalizing constants
p = np.array([0.2, 0.5, 0.3]); K = len(p); a = 1 - p[-1]
F = lambda y: np.where(np.floor(y) >= K, 1.0, np.where(np.floor(y) < 1, 0.0, np.cumsum(p)[np.clip(np.floor(y).astype(int), 1, K) - 1]))
rng = np.random.default_rng(731)
ok = True
for n in (10, 50, 200):
    c = 10 ** rng.uniform(-3, 3, 5000); d = rng.uniform(-10, 10, 5000); x = rng.uniform(-5, 5, 5000)
    v = F(c * x + d) ** n
    ok &= bool(np.all((v == 1) | (v <= a ** n + 1e-15)))
assert ok

# Q2(a): exact identity
uu = np.array([0.5, 0.9, 0.99, 0.999])
gap_a = 0.0
for u in uu:
    x = np.linspace(0, 1 - u, 5001)
    gap_a = max(gap_a, np.max(np.abs(x / (1 - u) - x / (1 - u))))
    assert np.all(np.isclose(((u + x) - u) / (1 - u), x / (1 - u)))

# Q2(b): sup over x >= delta of |F_u - G_{0,1/u}| against the proof's bound
def sup_gap(u, delta):
    x = delta + np.concatenate(([0], np.logspace(-6, 2.5, 400000)))
    R = np.exp(norm.logsf(u + x) - norm.logsf(u))
    return np.max(np.abs(R - np.exp(-u * x)))

rows = []
for u in (2, 5, 10, 20, 40):
    row = [u]
    for delta in (0.1, 0.5):
        s = sup_gap(u, delta); b = (1 + 1 / u**2) * np.exp(-u * delta)
        assert s <= b + 1e-14, (u, delta, s, b)
        row += [s, b]
    # the [0, inf) version the sheet mentions but does not ask for
    row.append(sup_gap(u, 0.0))
    rows.append(row)

# check the Mills inequalities on a grid
g = np.linspace(0.05, 30, 20000)
sf = np.exp(norm.logsf(g)); pdf = np.exp(-g**2 / 2) / np.sqrt(2 * np.pi)
assert np.all(g / (1 + g**2) * pdf <= sf) and np.all(sf <= pdf / g)

with open(os.path.join(ROOT, "generated", "table.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\\toprule $u$ & sup, $\\delta=.1$ & bound & sup, $\\delta=.5$ & bound & sup on $[0,\\infty)$\\\\\\midrule\n")
    fm = lambda v: f"{v:.3e}".replace("e-0", "e-").replace("e+00", "")
    for r in rows:
        f.write(f"{r[0]} & " + " & ".join(fm(v) for v in r[1:]) + "\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
for r in rows:
    print(r)
print("all checks pass")
