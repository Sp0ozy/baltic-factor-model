# Baltic Factor Model

## Project Goal
Build a factor-based investing model on Baltic equities (Nasdaq Riga, Tallinn, Vilnius).
Research whether momentum, volatility, and mean reversion factors predict returns.
Backtest a long/short portfolio strategy and compare to buy-and-hold benchmark.

## Universe
15 Baltic stocks from Nasdaq Baltic Main List.
Data: daily close prices 2021-present via yfinance.
Tickers: CPA1T.TL, ARC1T.TL, EFT1T.TL, EEG1T.TL, MRK1T.TL, NCN1T.TL, HAE1T.TL, 
LHV1T.TL, PKG1T.TL, APG1L.VS, AKO1L.VS, GRG1L.VS, IGN1L.VS, NTU1L.VS, DGR1R.RG

## Project Structure
data/           - raw price data (csv)
notebooks/      - EDA and research notebooks
src/            - production code
  data_ingestion.py  - downloads and saves price data

## Development Phases
1. EDA — return distributions, correlation matrix, missing data
2. Factor construction — momentum, volatility, mean reversion
3. Factor analysis — do factors predict next month returns?
4. Portfolio construction — rank and rebalance monthly
5. Backtesting — Sharpe ratio, max drawdown, vs benchmark
6. Visualization — equity curve, factor attribution

## Key Decisions
- Monthly rebalancing
- Evaluate factors using quintile analysis
- Primary metric: Sharpe ratio
- Benchmark: equal weight buy-and-hold of all 15 stocks

## Context
This is a portfolio project by Ilja Molcanovs, a Fintech student targeting 
quant/data science internships in Baltic/Nordic finance. 
The Baltic focus is intentional — unique angle for local companies like INDEXO, AOX Trade.