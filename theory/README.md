# EVT proofs (MF731 HW 10, sheet `10_extreme_theory_pub`)

Q1: no non-degenerate maxima limit for a finite discrete law. Q2(a): U(0,1), beta(u) = 1 - u, exact.
Q2(b): N(0,1), beta(u) = 1/u, sup over [delta, inf) at most (1 + u^-2) e^(-u delta).

```bash
pip install -r requirements.txt
python src/evt_check.py
latexmk -pdf main.tex
```
