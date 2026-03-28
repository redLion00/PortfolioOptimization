# QUBO Portfolio Optimization with Simulated Annealing

Binary portfolio selection using Quadratic Unconstrained Binary Optimization (QUBO),
solved via Simulated Annealing (SA), with convex weight optimization and backtesting
on Indian equities (NSE).

---

## What it does

1. Downloads historical price data for a set of NSE tickers and a benchmark (NIFTY 50)
2. Computes annualized returns and covariance (with optional Ledoit-Wolf shrinkage or EWM)
3. Builds a QUBO matrix encoding both portfolio risk/return and a cardinality constraint
4. Solves the QUBO with Simulated Annealing using multi-seed consensus and warm restarts
5. Optimizes continuous weights for the selected assets using convex optimization (CVXPY)
6. Backtests the portfolio on held-out test data, with and without quarterly rebalancing
7. Produces performance plots, a rolling Sharpe chart, an asset holdings heatmap, and a
   final report with annualized returns, drawdowns, and transaction costs

---

## Project structure

```
qubo.ipynb          Main notebook — all code lives here
plots/              Output charts (created on run)
logs/               Run logs (created on run)
```

---

## Requirements

Python 3.12+

```
numpy
pandas
yfinance
cvxpy
dimod
neal
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy pandas yfinance cvxpy dimod neal matplotlib scikit-learn
```

---

## Quick start

Open `qubo.ipynb` and run all cells. To change tickers, dates, or capital, edit only
the `main()` function at the bottom:

```python
def main():
    TICKERS = [
        'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'LT.NS', 'RELIANCE.NS', 'BHARTIARTL.NS'
    ]
    BENCHMARK       = '^NSEI'
    START_DATE      = datetime(2023, 1, 1)
    END_DATE        = datetime(2026, 1, 1)
    INITIAL_CAPITAL = 1600000
    Q_RISK_AVERSION = 0.5
    B_ASSETS        = 5
```

Everything else is handled by `run_pipeline()`.

---

## How the QUBO is built

The optimization problem is:

```
min  q * x^T C x  -  mu^T x  +  lambda * (sum(x) - B)^2
```

where `x` is a binary vector (1 = asset selected), `C` is the covariance matrix,
`mu` is expected returns, `q` is risk aversion, `B` is the target number of assets,
and `lambda` is the cardinality penalty.

Expanding the penalty and mapping to matrix form:

```
Q_ii  =  q * C_ii  -  mu_i  +  lambda * (1 - 2B)
Q_ij  =  2 * lambda                                (i != j)
```

Both `mu` and `C` are normalized by their max absolute value before building `Q`
so neither term dominates the other.

---

## Lambda scaling

Lambda is auto-computed each rebalance period from the current rolling window:

```
lambda = max(5 * max|mu| * B,  2 * max|cov|)
```

This ensures the cardinality penalty is always large enough relative to the return
and risk terms. An energy check `E(B) < E(B+2)` is printed after each QUBO build.
If this check fails, lambda is too small for that period.

---

## Simulated Annealing

- `num_reads = 1000`, `num_sweeps = 1000` per seed
- `beta_range` is scaled automatically to the magnitude of Q entries
- Multi-seed consensus: 5 seeds are run, assets selected by majority vote (>= 0.5)
- Warm restart: the previous rebalance solution is passed as the initial state for the next run
- If SA returns the wrong cardinality, hard enforcement fires using `-diag(Q)` as asset scores

---

## Covariance estimation

Controlled by flags in `compute_statistics()`:

| Flag | Method |
|------|--------|
| `use_shrinkage=True` (default) | Ledoit-Wolf shrinkage — reduces estimation error for small samples |
| `use_ewm=True` | Exponentially weighted covariance — more weight on recent observations |
| neither | Sample covariance |

---

## Weight optimization

After asset selection, continuous weights are found by solving:

```
minimize    w^T Sigma_S w
subject to  mu_S^T w = 1
            w >= 0
```

This is equivalent to Sharpe ratio maximization. Weights are then normalized to sum to 1.
If the solver fails, equal weights are used as fallback.

---

## Train / test split

Data is split 67% / 33% to avoid look-ahead bias:

- Training window: used to compute `mu`, `cov`, build QUBO, select assets, optimize weights
- Test window: used for all backtesting and rebalancing evaluation

Rebalancing during the test period uses a rolling lookback of 252 trading days,
recomputing `mu`, `cov`, and `lambda` fresh each quarter from data available at
that point — no future data leaks in.

---

## Rebalancing

- Frequency: every 3 months (63 trading days)
- Each rebalance recomputes statistics, lambda, QUBO, and weights from scratch
- Transaction costs applied on turnover: `cost = 0.2% * |new_value - old_value|` per asset
- Total transaction costs across all rebalances are reported at the end

---

## Outputs

**plots/portfolio_results.png** — 6-panel figure:
1. Portfolio vs benchmark (absolute value)
2. Cumulative returns
3. With vs without rebalancing
4. Initial asset weights (horizontal bar)
5. Rolling 90-day Sharpe ratio (rebalanced strategy)
6. Asset holdings heatmap across rebalance periods

**logs/run.log** — timestamped log of all pipeline steps, lambda values, energy
checks, selected assets, and capital at each rebalance.

**Console** — full final report including total return, annualized return, max
drawdown, Sharpe ratio, diversification ratio, outperformance vs benchmark, and
total transaction costs.

---

## Key parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `B_ASSETS` | `main()` | 5 | Number of assets to select |
| `Q_RISK_AVERSION` | `main()` | 0.5 | Risk aversion coefficient `q` |
| `INITIAL_CAPITAL` | `main()` | 1,600,000 | Starting capital in INR |
| `alpha` | `compute_lambda()` | 5.0 | Lambda safety multiplier |
| `n_seeds` | `solve_qubo_sa()` | 5 | Number of SA seeds for consensus |
| `num_reads` | `solve_qubo_sa()` | 1000 | SA reads per seed |
| `num_sweeps` | `solve_qubo_sa()` | 1000 | SA sweeps per read |
| `rebalance_months` | `rebalance_portfolio()` | 3 | Rebalancing frequency |
| `lookback_days` | `rebalance_portfolio()` | 252 | Rolling window for statistics |
| `transaction_cost` | `rebalance_portfolio()` | 0.002 | Cost as fraction of turnover |

---

## Debugging

If SA consistently violates the cardinality constraint:

1. Check the energy output: `E(B) < E(B+2)` must hold. If not, increase `alpha` in `compute_lambda()`
2. Increase `n_seeds` or `num_reads` in `solve_qubo_sa()`
3. Check that `beta_range` printed by `build_qubo()` is reasonable — very small values
   indicate Q entries are large and SA is not exploring enough

If weight optimization fails:
- Check that selected assets have positive expected returns in the training window
- The fallback is equal weights, which will still run the backtest correctly

---

## Limitations

- Uses `yfinance` for data — suitable for research, not production
- SA is a heuristic; results can vary across runs even with the same seed when `n_seeds > 1`
- Covariance estimation degrades with fewer than ~60 observations per asset
- No slippage model beyond flat transaction costs
- Assumes full capital deployment with no cash buffer