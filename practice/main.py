"""Run evt.py, assert the checks, write generated/values.tex, the table and the figure."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evt

ROOT = evt.ROOT
GEN, FIG = os.path.join(ROOT, "generated"), os.path.join(ROOT, "figures")
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
r = evt.run(do_boot="--no-boot" not in sys.argv)
C, x, lp = r["curves"], r["x"], r["lp"]
macros = {}


def put(name, value, fmt="{:,.0f}"):
    macros[name] = value if isinstance(value, str) else fmt.format(value)


LABEL = {"emp": "(1) Empirical", "ewma": "(2) EWMA normal", "gev73": "(3i) GEV, n = 73",
         "gev35": "(3ii) GEV, n = 35", "gp": "(4) GP"}
ORDER = ["emp", "ewma", "gev73", "gev35", "gp"]
top = {k: C[k][-1] for k in ORDER}
assert abs(r["alphas"][-1] - 0.9999) < 1e-12 and len(r["alphas"]) == 100

# ---- checks
ck = r["checks"]
for k in ("gev73", "gev35", "gp"):
    c = ck[k]
    assert abs(c["var"] - c["inv"]) < 1, k                       # formula = numerical inversion
    assert c["dnll"] < 1e-4, (k, c["dnll"])                      # scipy is at the optimum
    assert abs(c["var"] / c["var2"] - 1) < 2e-3, (k, c["var"], c["var2"])
assert abs(ck["gp"]["at_u"] - r["gp"]["u"]) < 1e-9
assert r["gp"]["nu"] == 2555 - int(np.ceil(0.95 * 2555))
assert top["emp"] == lp.max() * 1e4

# ---- macros
put("Nprices", len(r["p"]), "{:d}"); put("Nret", len(x), "{:d}")
put("FirstDate", r["p"].index[0].strftime("%-m/%-d/%Y")); put("LastDate", r["p"].index[-1].strftime("%-m/%-d/%Y"))
worst = x.idxmin()
put("WorstDate", worst.strftime("%-m/%-d/%Y")); put("WorstPct", -100 * x.min(), "{:.2f}")
mu, s2 = r["ewma"]
put("MuBp", 1e4 * mu, "{:.2f}"); put("SigPct", 100 * np.sqrt(s2), "{:.3f}")
ml, sl, vl = r["ewma_last"]
put("MuLastBp", 1e4 * ml, "{:.2f}"); put("SigLastPct", 100 * np.sqrt(sl), "{:.3f}"); put("EwmaLast", vl)
for n, tag in ((73, "A"), (35, "B")):
    g = r["gev"][n]
    put(f"Xi{tag}", g["xi"], "{:.4f}"); put(f"Mu{tag}", g["mu"], "{:.4f}"); put(f"Sig{tag}", g["sig"], "{:.4f}")
    put(f"NegM{tag}", int((g["maxima"] < 0).sum()), "{:d}")
g = r["gp"]
put("XiGP", g["xi"], "{:.4f}"); put("BetaGP", g["beta"], "{:.4f}"); put("Upct", g["u"], "{:.4f}")
put("Uusd", g["u"] * 1e4); put("Nu", g["nu"], "{:d}"); put("Fhat", 1 - g["nu"] / len(x), "{:.4f}")
for k, tag in zip(ORDER, ("Emp", "Ewma", "GevA", "GevB", "GP")):
    put(f"Top{tag}", top[k]); put(f"Low{tag}", C[k][0])
ratio = lambda a, b: top[a] / top[b]
put("RatioEmpEwma", ratio("emp", "ewma"), "{:.2f}")
put("GpMinusGevA", top["gp"] - top["gev73"]); put("GevBMinusGevA", top["gev35"] - top["gev73"])
put("GevBMinusGPabs", top["gp"] - top["gev35"])
for k, s in zip(("gev73", "gev35", "gp"), ("GevA", "GevB", "GP")):
    put(f"Dual{s}", ck[k]["var2"])
for lev, s in zip((0.90, 0.95, 0.975), ("Ninety", "NinetyFive", "NineSevenFive")):
    q = r["gp_sens"][lev]
    put(f"SensU{s}", q["u"]); put(f"SensN{s}", q["nu"], "{:d}"); put(f"SensXi{s}", q["xi"], "{:.3f}"); put(f"SensV{s}", q["var"])
put("UsdXi", r["gev_usd"]["xi"], "{:.4f}"); put("UsdVar", r["gev_usd"]["var"])
put("DecXi", r["gev_dec"]["xi"], "{:.4f}"); put("DecVar", r["gev_dec"]["var"])
put("BB", evt.B, "{:d}"); put("Seed", evt.SEED, "{:d}")

if "boot" in r:
    b = r["boot"]
    b = b.replace([np.inf, -np.inf], np.nan).dropna()
    put("BootOK", len(b), "{:d}")
    q = lambda s: (np.percentile(s, 2.5), np.percentile(s, 97.5))
    for k, s in zip(("gev73", "gev35", "gp"), ("GevA", "GevB", "GP")):
        lo, hi = q(b[k]); put(f"Lo{s}", lo); put(f"Hi{s}", hi)
    keys = ("gev73", "gev35", "gp")
    intervals = {k: q(b[k]) for k in keys}
    for k in keys:
        assert intervals[k][0] < top[k] < intervals[k][1], k     # each interval covers its own estimate
    inside = all(intervals[i][0] < top[j] < intervals[i][1] for i in keys for j in keys)
    put("AllInside", "yes" if inside else "no")
    print("intervals", intervals, "all estimates inside all intervals:", inside)

with open(os.path.join(GEN, "values.tex"), "w") as f:
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

ranked = sorted(ORDER, key=lambda k: -top[k])
with open(os.path.join(GEN, "table.tex"), "w") as f:
    f.write("\\begin{tabular}{lrr}\\toprule Method & $\\VaR_{.99}$ (\\$) & $\\VaR_{.9999}$ (\\$)\\\\\\midrule\n")
    for k in ranked:
        f.write(f"{LABEL[k].replace('n = ', '$n=').replace(', $n=73', ', $n=73$').replace(', $n=35', ', $n=35$')} & {C[k][0]:,.0f} & {top[k]:,.0f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")

# ---- figure
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, MUTED = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(7.2, 4.4))
tail = -np.log10(1 - r["alphas"])
for k in ORDER:
    kw = dict(drawstyle="steps-post") if k == "emp" else {}
    ax.plot(tail, C[k] / 1e3, color=COL[k], lw=2, label=LABEL[k], solid_capstyle="round", **kw)
ticks = [0.99, 0.995, 0.999, 0.9995, 0.9999]
ax.set_xticks([-np.log10(1 - t) for t in ticks]); ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.set_xlabel(r"$\alpha$ (log scale in $1-\alpha$)", color=INK); ax.set_ylabel("VaR ($ thousands)", color=INK)
ax.grid(axis="y", color="#e6e5e0", lw=0.7); ax.set_axisbelow(True)
ax.set_title(r"One-day VaR on \$1M of S&P 500", loc="left", color=INK)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "var_curves.pdf")); fig.savefig(os.path.join(FIG, "var_curves.png"), dpi=170)

print({k: round(v) for k, v in top.items()})
print({k: round(C[k][0]) for k in ORDER})
for k, v in macros.items():
    print(k, v)
