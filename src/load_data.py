"""
Project 8 real data: estimate p = Pr(daily TMAX > 90 F) in June-August at Boston Logan
(NOAA GHCN-Daily station USW00014739, 1973 onward). Y_i = 1{TMAX_i > 90 F}, p_hat = mean(Y).

    python src/load_data.py

Downloads the station CSV into data/ on the first run (not committed to git).
"""

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
STATION = "USW00014739"
URL = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{STATION}.csv"
RAW = REPO / "data" / f"{STATION}.csv"
FIGDIR = REPO / "figures"
THRESHOLD_F, MONTHS, START_YEAR = 90.0, (6, 7, 8), 1973
Z = 1.96


def load():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, RAW)
    df = pd.read_csv(RAW, usecols=["STATION", "NAME", "DATE", "TMAX", "TMIN", "PRCP"],
                     parse_dates=["DATE"], low_memory=False)
    # GHCN-Daily stores TMAX/TMIN in tenths of degrees C and PRCP in tenths of mm
    df[["TMAX", "TMIN", "PRCP"]] /= 10
    df["TMAX_F"] = df["TMAX"] * 9 / 5 + 32
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH"] = df["DATE"].dt.month
    return df[df["YEAR"] >= START_YEAR].sort_values("DATE").reset_index(drop=True)


def main():
    df = load()

    # Describe: rows, columns, types, missing values
    print(f"{df['NAME'].iat[0]}: {df.shape[0]:,} rows x {df.shape[1]} columns, "
          f"{df['DATE'].min().date()} to {df['DATE'].max().date()}; one row = one station-day\n")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "min": df.select_dtypes("number").min().round(2),
                        "max": df.select_dtypes("number").max().round(2)}).to_string())

    # Estimate p with a CLT interval (Q1, Q2, Q4)
    summer = df[df["MONTH"].isin(MONTHS)].dropna(subset=["TMAX_F"])
    y = (summer["TMAX_F"] > THRESHOLD_F).to_numpy(dtype=float)
    n, p_hat, s = y.size, y.mean(), y.std(ddof=1)
    se = s / np.sqrt(n)
    print(f"\nPr(TMAX > {THRESHOLD_F:g} F), months {MONTHS}:")
    print(f"  n = {n:,}, hot days = {int(y.sum()):,}")
    print(f"  p_hat = {p_hat:.4f}, s^2 = {s**2:.4f}, SE = {se:.4f}")
    print(f"  95% CI = [{p_hat - Z * se:.4f}, {p_hat + Z * se:.4f}]")

    FIGDIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(summer["TMAX_F"], bins=50, edgecolor="white")
    ax.axvline(THRESHOLD_F, color="#bc4b51", lw=2, label=f"threshold = {THRESHOLD_F:g} F")
    ax.set(xlabel="daily maximum temperature (F)", ylabel="number of station-days",
           title=f"Distribution of summer TMAX, n = {n:,}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "real_tmax_hist.png", dpi=200)

    grid = np.arange(1, n + 1)
    running = np.cumsum(y) / grid
    band = Z * s / np.sqrt(grid)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(grid, running, lw=1.0, label=r"running $\hat{p}_n$")
    ax.axhline(p_hat, color="#bc4b51", ls="--", label=f"full-sample estimate = {p_hat:.3f}")
    ax.fill_between(grid, p_hat - band, p_hat + band, color="grey", alpha=0.18,
                    label=r"$\pm 1.96\, s/\sqrt{n}$")
    ax.set(xscale="log", xlim=(10, n), ylim=(max(0, p_hat - 0.25), p_hat + 0.25),
           xlabel="number of station-days used $n$", ylabel=r"$\hat{p}_n$",
           title=f"Running estimate of Pr(TMAX > {THRESHOLD_F:g} F)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "real_running_estimate.png", dpi=200)

    print(f"\nFigures written to {FIGDIR}")


if __name__ == "__main__":
    main()
