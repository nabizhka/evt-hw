"""Download S&P 500 daily closes from Yahoo Finance once and freeze them in data/."""
import os
import yfinance as yf

ROOT = os.path.join(os.path.dirname(__file__), "..")
df = yf.download("^GSPC", start="2011-11-01", end="2021-12-30", auto_adjust=False, progress=False)
close = df["Close"].squeeze().rename("GSPC")
close.index.name = "Date"
close.to_csv(os.path.join(ROOT, "data", "gspc_close.csv"))
print(len(close), close.index[0].date(), close.index[-1].date())
