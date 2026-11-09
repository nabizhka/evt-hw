# EVT VaR for $1M of S&P 500 (MF731 HW 10, sheet `10_extreme_practice`)

Closes 11/1/2011 to 12/28/2021 (2556 days, 2555 log returns), frozen in `data/`. Yahoo has 2557
closes through 12/29/2021, so the end date moved one day as the sheet instructs.

VaR.9999: GP 135,599 (highest), empirical 127,652, GEV n=35 106,039, GEV n=73 90,357, EWMA normal 25,667.

```bash
pip install -r requirements.txt
python src/main.py        # ~20 s with the bootstrap; --no-boot skips it
latexmk -pdf main.tex
```
`src/get_data.py` re-downloads the data; it is not needed to reproduce the report.
