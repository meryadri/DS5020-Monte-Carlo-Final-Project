"""
Project 8 simulation check: estimate pi with Y = 4 * 1{U1^2 + U2^2 <= 1}, U1, U2 ~ Unif(0,1),
and compare against the closed forms from Q1-Q5 (mu = pi, sigma^2 = pi(4 - pi)).

    python src/simulate_pi.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MU = np.pi                  # Q1
SIGMA2 = np.pi * (4 - np.pi)  # Q5
Z = 1.96                    # Q4
FIGDIR = Path(__file__).resolve().parent.parent / "figures"


def estimate(n, rng):
    U = rng.random((n, 2))
    Y = 4.0 * (U[:, 0] ** 2 + U[:, 1] ** 2 <= 1)
    return Y.mean(), Y.std(ddof=1)


def replicate(n, R, rng):
    return np.array([estimate(n, rng) for _ in range(R)])


def main():
    rng = np.random.default_rng(0)
    n, R, eps = 1000, 2000, 0.1
    ns = np.array([100, 1_000, 10_000, 100_000])

    # Single runs at increasing n: the error shrinks (Q3)
    print(f"{'n':>8} {'mu_hat':>9} {'|error|':>9} {'CI half':>9}")
    singles = {}
    for m in ns:
        mu_hat, s = estimate(m, rng)
        singles[m] = s
        print(f"{m:>8,} {mu_hat:>9.5f} {abs(mu_hat - MU):>9.5f} {Z * s / np.sqrt(m):>9.5f}")

    # R repetitions at n = 1000
    res = replicate(n, R, rng)
    mu_hats, s = res[:, 0], res[:, 1]

    # Q2: RMSE vs n on log-log axes, slope should be -1/2
    rmse = np.array([np.sqrt(np.mean((replicate(m, 500, rng)[:, 0] - MU) ** 2)) for m in ns])
    slope, intercept = np.polyfit(np.log(ns), np.log(rmse), 1)

    half = Z * s / np.sqrt(n)
    coverage = np.mean((mu_hats - half <= MU) & (MU <= mu_hats + half))

    table = pd.DataFrame([
        ("Q1", "E[mu_hat_n]", MU, mu_hats.mean()),
        ("Q2", "Var(mu_hat_n)", SIGMA2 / n, mu_hats.var(ddof=1)),
        ("Q2", "log-log slope", -0.5, slope),
        ("Q3", f"Pr(|mu_hat_n - pi| >= {eps}) vs Chebyshev bound",
         SIGMA2 / (n * eps ** 2), np.mean(np.abs(mu_hats - MU) >= eps)),
        ("Q4", "CI coverage", 0.95, coverage),
        ("Q5", "s^2 (one sample, n = 1000)", SIGMA2, singles[1000] ** 2),
        ("Q5", "mean of s^2 over R runs", SIGMA2, np.mean(s ** 2)),
    ], columns=["Q", "quantity", "true", "estimate"])
    print(f"\nSimulation check (n = {n}, R = {R}):\n")
    print(table.to_string(index=False, float_format="{:.6f}".format))

    FIGDIR.mkdir(exist_ok=True)

    # Error decay
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(ns, rmse, "o", label="simulated RMSE")
    ax.loglog(ns, np.sqrt(SIGMA2 / ns), "-", label=r"theory $\sigma/\sqrt{n}$")
    ax.loglog(ns, np.exp(intercept) * ns ** slope, "--", label=f"fit, slope = {slope:.3f}")
    ax.set(xlabel="number of trials $n$", ylabel=r"RMSE of $\hat{\mu}_n$",
           title=r"Monte Carlo error decays like $n^{-1/2}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "loglog_error.png", dpi=200)

    print(f"\nFigure written to {FIGDIR / 'loglog_error.png'}")


if __name__ == "__main__":
    main()
