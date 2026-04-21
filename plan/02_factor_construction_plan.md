# Factor Construction Plan — Baltic Factor Model

Notebook: `notebooks/02_factor_construction.ipynb`
Inputs: `data/baltic_prices.csv` — monthly_returns and daily_returns from EDA

---

## Context from EDA

Key decisions that shape this phase:
- **Non-normal returns** across all 15 tickers → use rank-based (quintile) normalisation, not z-scores
- **8/15 tickers have >20% stale days** → use monthly returns as the factor signal input, not daily
- **DGR1R.RG has 17.1% missing** → exclude from rankings before its first valid date
- **Autocorrelation leans momentum** — 10/15 tickers show positive lag-1 autocorr
- **PACF significance concentrated at lag-1 to lag-3** → short lookback windows appropriate

---

## 1. Setup & Data Preparation

**Goal:** Reconstruct the clean monthly return series and define the universe for each rebalance date.

- Load `daily_returns` and `monthly_returns` from EDA (or recompute from CSV)
- Define the **valid universe** per month: tickers with a non-NaN return in that month
- Build a `universe_mask` DataFrame (bool, same shape as monthly_returns) — True if ticker is tradeable in that month
- Flag DGR1R.RG as unavailable before its first valid date

```python
universe_mask = monthly_returns.notna()
```

---

## 2. Momentum Factor

**Goal:** Capture the tendency of recent winners to keep winning.

### 2a. Signal construction
Compute cumulative return over the lookback window, skipping the most recent month (standard momentum skips lag-1 to avoid short-term reversal):

```python
# 3-month momentum, skip last month (t-1)
mom_3 = monthly_returns.shift(1).rolling(3).apply(lambda x: (1 + x).prod() - 1)

# 6-month momentum, skip last month
mom_6 = monthly_returns.shift(1).rolling(6).apply(lambda x: (1 + x).prod() - 1)

# 1-month momentum (no skip — test reversal vs continuation)
mom_1 = monthly_returns.shift(1)
```

### 2b. Cross-sectional ranking
At each month-end, rank tickers by signal value within the valid universe:

```python
# Rank 1 = lowest signal, 5 = highest signal (within valid universe only)
def rank_cross_sectional(signal_row, mask_row):
    valid = signal_row[mask_row]
    return valid.rank(pct=True)  # percentile rank 0–1
```

### 2c. Quintile assignment
Assign each ticker to quintile Q1 (lowest) through Q5 (highest) at each rebalance date.

### 2d. Signal diagnostics
- Plot the distribution of momentum scores over time (are scores stable or volatile?)
- Check how often tickers switch quintiles month-to-month (high turnover = noisy signal)
- Heatmap: quintile membership per ticker per month

---

## 3. Mean Reversion Factor

**Goal:** Capture the tendency of recent losers to bounce back.

### 3a. Signal construction
1-month lagged return (simple reversal signal — last month's loser is this month's long):

```python
# Short-term reversal: negative of last month's return
# (low score = strong reversal candidate = should go long)
rev_1 = -monthly_returns.shift(1)
```

Also test 3-month reversal for comparison:
```python
rev_3 = -monthly_returns.shift(1).rolling(3).apply(lambda x: (1 + x).prod() - 1)
```

### 3b. Cross-sectional ranking
Same procedure as momentum — rank within valid universe, assign quintiles.

### 3c. Signal diagnostics
- Same diagnostics as 2d
- Compare reversal signal quintile assignments to momentum quintile assignments — are they anti-correlated? (They should be for short windows)

---

## 4. Volatility Factor

**Goal:** Capture the low-volatility anomaly — lower-risk stocks tend to outperform on a risk-adjusted basis.

### 4a. Signal construction
Realised volatility over the past month using daily returns:

```python
# 21-day (1-month) realised vol as of each month-end
vol_1m = daily_returns.rolling(21).std() * np.sqrt(252)
vol_signal = vol_1m.resample("ME").last()  # snapshot at month-end

# Also compute 3-month average vol (more stable, reduces vol-of-vol noise)
vol_3m = daily_returns.rolling(63).std() * np.sqrt(252)
vol_signal_3m = vol_3m.resample("ME").last()
```

Use `vol_3m` as the primary signal given EDA finding that PKG1T.TL and HAE1T.TL have high vol-of-vol.

### 4b. Cross-sectional ranking
Rank within valid universe. Q1 = lowest vol (long candidates), Q5 = highest vol (short candidates).

### 4c. Signal diagnostics
- Quintile persistence: does a low-vol stock stay low-vol month to month? (expect high persistence)
- Compare 1-month vs 3-month vol signal — does averaging improve quintile stability?

---

## 5. Factor Correlation & Orthogonality

**Goal:** Understand how much the three factors overlap — if momentum and mean reversion are nearly identical (anti-correlated), using both adds no value.

### 5a. Signal-level correlation
- Compute Spearman rank correlation between the three factor signals at each month-end
- Average across time: `corr(mom_3, rev_1)`, `corr(mom_3, vol)`, `corr(rev_1, vol)`

### 5b. Quintile overlap matrix
- For each pair of factors, count how often a ticker is in Q5 of both simultaneously
- High overlap → factors are redundant

### 5c. Decision
- If `|corr(mom, rev)| > 0.7` → they're too similar; use only one or blend at construction stage
- Volatility factor should be orthogonal to both (different information source)

---

## 6. Factor Scores Summary

**Goal:** Produce the final factor score DataFrames that feed directly into Phase 3 (factor analysis).

Create three clean DataFrames, each indexed by month-end date with 15 ticker columns:

| DataFrame | Content |
|---|---|
| `factor_momentum` | Cross-sectional percentile rank of momentum signal (0–1) |
| `factor_reversion` | Cross-sectional percentile rank of reversal signal (0–1) |
| `factor_volatility` | Cross-sectional percentile rank of vol signal (0–1) |

Each cell: NaN if ticker not in valid universe that month, else rank in [0, 1].

Save all three to `data/` as CSV for use in Phase 3:
```python
factor_momentum.to_csv("../data/factor_momentum.csv")
factor_reversion.to_csv("../data/factor_reversion.csv")
factor_volatility.to_csv("../data/factor_volatility.csv")
```

---

## 7. Key Questions to Answer

By the end of this notebook, document:

1. **Which momentum window is most stable?** — 1m, 3m, or 6m lookback
2. **Does reversal complement momentum or replicate it?** — correlation check
3. **Is vol-of-vol a problem for the volatility factor?** — does 3m smoothing help?
4. **How many tickers are in the valid universe each month?** — drops below 10 are a concern for quintile construction
5. **What is average monthly turnover per factor?** — high turnover = higher transaction costs in backtest

---

## Execution Order

```
Section 1 (setup)
    ↓
Section 2 (momentum) → Section 3 (mean reversion) → Section 4 (volatility)
    ↓
Section 5 (orthogonality check)
    ↓
Section 6 (save factor scores)
    ↓
Section 7 (document findings)
```

Sections 2–4 are independent and can be built in any order.

---

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
```
