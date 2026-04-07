# EDA Plan — Baltic Factor Model

Notebook: `notebooks/01_eda.ipynb`
Data: `data/baltic_prices.csv` (15 tickers, daily close, 2021-present)

---

## 1. Data Loading & Basic Inspection

**Goal:** Confirm data loaded correctly before any analysis.

- Load CSV with `index_col="Date", parse_dates=True`
- Check shape: expect ~1300 rows × 15 columns
- Print `df.head()`, `df.tail()` — spot-check dates and values
- Print `df.dtypes` — all columns should be float64
- Print `df.index.min()` / `df.index.max()` — confirm 2021-01-04 to present

---

## 2. Data Quality

**Goal:** Know exactly where and how much data is missing before computing anything.

### 2a. Missing values
- `df.isnull().sum()` — count NaNs per ticker
- `df.isnull().sum() / len(df) * 100` — NaN % per ticker
- Flag tickers with >5% missing (currently `DGR1R.RG` has ~17% missing)

### 2b. Missing value pattern
- Heatmap of nulls: `sns.heatmap(df.isnull(), cbar=False, yticklabels=False)`
- Are NaNs clustered at the start (late listing) or scattered (illiquidity/halts)?

### 2c. Trading calendar gaps
- Check if weekend/holiday gaps are consistent across all tickers
- Compute `df.index.to_series().diff().dt.days.value_counts()` — most gaps should be 1 or 3 (weekends)

### 2d. Stale prices (flat lines)
- For each ticker, count consecutive identical prices: signals illiquid days
- `(df.diff() == 0).sum()` — days with zero price change per ticker

---

## 3. Price & Return Series

**Goal:** Understand the raw price levels and compute returns that all subsequent analysis will use.

### 3a. Compute returns
```python
daily_returns = df.pct_change().dropna(how="all")
monthly_prices = df.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna(how="all")
```
Save both as variables — monthly returns are the primary input for factor analysis.

### 3b. Normalized price chart
- Divide each price series by its first non-NaN value → base 100
- Plot all 15 tickers on one chart; use separate y-axis scale or log scale
- Identify top/bottom performers over the full period

### 3c. Individual price series
- Small multiples (3×5 grid) of each ticker's price history
- Helps spot regime changes, crashes, data anomalies

---

## 4. Return Distributions

**Goal:** Understand the statistical properties of returns — critical for choosing factor normalisation later.

### 4a. Summary statistics (daily returns)
- `daily_returns.describe()` — mean, std, min, max, quartiles
- Annualised return: `daily_returns.mean() * 252`
- Annualised vol: `daily_returns.std() * np.sqrt(252)`

### 4b. Distribution shape
- Histogram + KDE for each ticker (small multiples)
- Compute skewness and excess kurtosis per ticker: `daily_returns.skew()`, `daily_returns.kurt()`
- Flag tickers with |skew| > 1 or kurt > 3 (fat tails matter for factor ranking)

### 4c. Normality test
- Jarque-Bera test per ticker: `scipy.stats.jarque_bera(returns.dropna())`
- Most equity return series will fail normality — document which ones deviate most

### 4d. Monthly return distributions
- Repeat 4a–4b for monthly returns (used in factor analysis)
- Box plot of monthly returns per ticker — visual comparison of vol and outliers

---

## 5. Correlation Analysis

**Goal:** Understand co-movement — relevant for portfolio diversification and factor orthogonality.

### 5a. Correlation matrix (monthly returns)
- `monthly_returns.corr()`
- Heatmap with annotation; cluster by exchange (TL / VS / RG)

### 5b. Cross-exchange vs within-exchange correlation
- Average within-TL, within-VS, within-RG, and cross-exchange correlations
- Are Tallinn stocks more correlated with each other than with Riga/Vilnius?

### 5c. Rolling correlation
- Pick 2–3 stock pairs; plot 3-month rolling correlation over time
- Identify periods of correlation breakdown (crisis, local market stress)

---

## 6. Return Autocorrelation

**Goal:** Detect mean reversion or momentum in individual stock returns — directly informs factor design.

### 6a. Autocorrelation of monthly returns
- `pd.plotting.autocorrelation_plot(monthly_returns[ticker])` for a few tickers
- Or compute `monthly_returns.apply(lambda x: x.autocorr(lag=1))` for all tickers
- Positive lag-1 autocorr → momentum; negative → mean reversion

### 6b. Partial autocorrelation
- PACF plot for 2–3 tickers (lags 1–12)
- Guides choice of lookback windows for momentum and mean reversion factors

---

## 7. Volatility Analysis

**Goal:** Understand volatility dynamics — needed for the volatility factor construction.

### 7a. Rolling volatility
- 21-day (1-month) rolling std of daily returns for each ticker
- Plot to identify volatility clustering and regime shifts

### 7b. Volatility ranking
- Rank tickers by average realised vol
- Bar chart: annualised vol per ticker, coloured by exchange

### 7c. Vol of vol
- How stable is each stock's volatility? Compute std of rolling vol
- High vol-of-vol stocks may be harder to rank reliably

---

## 8. Summary Table

**Goal:** Single reference table for all key stats — use as a sanity check before moving to factor construction.

Produce a DataFrame with one row per ticker containing:

| Ticker | Exchange | # Obs | NaN % | Ann. Return | Ann. Vol | Sharpe | Skew | Kurt | Lag-1 Autocorr |
|--------|----------|-------|-------|-------------|----------|--------|------|------|----------------|

---

## 9. Key Findings to Document

At the end of the notebook, write a markdown cell summarising:

1. **Missing data:** which tickers have gaps and why (late listing, halts)
2. **Return characteristics:** are returns fat-tailed? Any outlier stocks?
3. **Correlation structure:** are exchanges segmented?
4. **Autocorrelation signal:** early evidence for momentum vs mean reversion
5. **Data issues to handle in factor construction:** stale prices, missing data treatment

---

## Execution Order

```
Section 1  → Section 2 → Section 3 (compute returns here)
         ↓
Section 4 → Section 5 → Section 6 → Section 7 → Section 8 → Section 9
```

All sections after 3 depend on `daily_returns` and `monthly_returns` being defined.

---

## Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```
