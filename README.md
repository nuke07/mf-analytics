# 📊 MF Quantitative Analytics Platform

An institutional-grade quantitative analytics dashboard for Indian mutual funds, built with Python and Streamlit.

> **Disclaimer:** This platform provides quantitative analytics only. It does not provide investment advice, recommendations, or fund ratings. Predictive Analytics (GARCH, Monte Carlo) models risk and scenario ranges from historical data — it does not predict future returns. All metrics computed within a fund's own category; cross-category comparisons are not supported by design.

---

## ✨ Features

- **64 quantitative metrics** per fund, computed against true benchmark data (see below)
- **Total Return Index (TRI) benchmarking** — 11 validated NSE indices sourced directly from niftyindices.com, not price-return proxies. Automatic fallback to index-fund NAV proxies if TRI data is unavailable for a category
- **6-Factor Attribution model** — Market, SMB (Size), HML (Value), WML (Momentum), QMJ (Quality), BAB (Low Volatility), with standardised betas for cross-fund comparison, rolling factor exposures, return attribution, and regime-conditional betas (Bull/Sideways/Bear)
- **Fama-French 4-Factor model** on every fund's individual analytics page, alongside the dedicated 6-factor model
- **Predictive Analytics** — GARCH(1,1) conditional volatility forecasting, block-bootstrap Monte Carlo simulation (preserves fat tails and volatility clustering from actual historical return blocks), and derived drawdown risk — framed strictly as scenario/risk estimation, never return prediction
- **Portfolio Analytics** — build and compare two portfolios (A/B) side by side, with risk and allocation breakdowns
- **Rankings** — 11 tabs covering Performance, Risk-Adjusted, Risk, Consistency, Stability, Alpha, Absolute Returns, Momentum, Persistence, Factor Model, and Quartile View
- **Quartile system** — every metric ranked Q1–Q4 within its category
- **Data quality reporting** — NAV coverage and missing-data warnings per fund
- **CSV export** on every ranking and comparison table
- **Live data** from AMFI via mftool, with mfapi.in as automatic fallback — refreshes daily

---

## 📂 Pages

| Page | Purpose |
|---|---|
| `app.py` | Home — fund counts, category cards, TRI data-staleness indicator |
| `pages/3_Fund_Analytics.py` | Single-fund deep dive — Charts, Alpha, Factor (4F), All Metrics, Data Quality tabs |
| `pages/4_Fund_Comparison.py` | Compare 2–5 funds, trailing returns with benchmark overlay |
| `pages/5_Rankings.py` | 11-tab ranking system across all metric families |
| `pages/6_Data_Quality.py` | NAV coverage scan and quality matrix |
| `pages/7_Portfolio_Analytics.py` | Dual portfolio (A/B) builder and comparison |
| `pages/8_Predictive_Analytics.py` | GARCH volatility forecasting, Monte Carlo, Drawdown Risk |
| `pages/9_Factor_Attribution.py` | Dedicated 6-factor model — loadings, rolling exposures, attribution, regimes |

---

## 📂 Project Structure

mf_analytics/
│
├── app.py                        Home page + global sidebar + TRI staleness note
├── requirements.txt
│
├── .streamlit/
│   └── config.toml               Pitch-black theme, bundled webfonts, static serving.
│                                 Repeats four colour tokens from utils/theme.py by
│                                 necessity — Streamlit reads this before Python runs.
│                                 test_design_system.py asserts they stay in step.
│
├── static/fonts/                 Archivo + IBM Plex Mono woff2, served locally.
│                                 Bundled, not CDN: the app must not need the network
│                                 for its typeface.
│
├── utils/
│   ├── theme.py                  Single source of truth for colour and type
│   ├── ui.py                     Shared controls: rf_control, kpi, kpi_row, chart,
│   │                             card, export_button, swatch
│   ├── constants.py              Categories, metric keys, labels, help, ANALYTICS_VERSION
│   ├── formatters.py             Display formatting helpers
│   ├── validators.py             NAV series checks, coverage reports
│   └── session.py                Versioned session-state key builders
│
├── data/
│   ├── fund_loader.py            mftool + direct API fallback, parallel NAV loading
│   ├── category_mapper.py        Keyword-based category detection
│   ├── nav_processor.py          NAV cleaning, returns computation
│   ├── benchmark_loader.py       TRI-first benchmark resolution, proxy fallback
│   ├── tri_loader.py             Sole TRI bridge (data/tri/*.csv → NAV contract)
│   ├── factor_loader.py          6-factor return series (Mkt/SMB/HML/WML/QMJ/BAB)
│   └── tri/                      13 validated NSE TRI CSVs
│
├── indices/                      NSE TRI ingestion package
│   ├── config/                   Index registry, metadata, endpoints
│   ├── data_ingestion/           Downloader, session/cookie handling, validators, cache
│   └── utils/                    Logging
│
├── scripts/
│   └── update_indices.py         Refresh TRI: python -m scripts.update_indices
│
├── analytics/                    Metrics engine — 78 metric keys
│   ├── engine.py                 Master orchestrator
│   ├── performance.py / volatility.py / risk.py / risk_adjusted.py
│   ├── consistency.py / distribution.py / stability.py / persistence.py
│   ├── alpha.py / momentum.py / alpha_persistence.py
│   ├── factor_model.py           6-factor: standardised betas, rolling alpha, regimes
│   ├── uncertainty.py            Lo (2002) Sharpe standard errors, Newey-West serial-
│   │                             correlation correction, indistinguishable_bands()
│   ├── garch_model.py            GARCH(1,1) volatility forecasting
│   ├── monte_carlo.py            Block-bootstrap scenario simulation
│   └── quartile.py
│
├── visualizations/               Plotly chart builders on the shared theme
│
├── pages/                        7 pages, numbered 1–7
│
└── tests/                        9 suites + runner
    ├── run_all.py                python tests/run_all.py --fast  (skips the browser suite)
    ├── test_analytics_regressions.py   Numeric correctness. The ONLY coverage of P99,
    │                             DaR95, QMJ/BAB loadings, the D-06 intercept fix and
    │                             calc_max_drawdown's _dd_series. Do not delete.
    ├── test_uncertainty.py       Sharpe SE against closed form, ACF factor vs AR(1) theory
    ├── test_design_system.py     Palette, contrast, type, emoji sweep
    ├── test_pages_with_data.py   Every page driven WITH data, offline
    ├── test_pages_render.py      Every page renders without a Streamlit exception
    ├── test_rf_navigation.py     Real browser — AppTest cannot navigate between pages
    ├── test_rf_control.py / test_parallel_load.py / test_visual_polish.py
    ├── test_amfi_parse.py        AMFI schema fix, both field layouts
    └── sitecustomize.py          Synthetic-fund stubs, gated on MF_STUB_DATA=1.
                                  MUST stay here. Python auto-imports any module named
                                  sitecustomize at interpreter startup, so at the repo
                                  root one stray env var would have the real app
                                  computing returns on synthetic funds.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mf_analytics.git
cd mf_analytics
```

### 2. Create and activate a conda environment

```bash
conda create -n mf_env python=3.11
conda activate mf_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check connectivity (run this before the app)

```bash
python debug_connection.py
```

This verifies your machine can reach AMFI and mfapi.in, and confirms your installed library versions and mftool API surface. If it fails, see Troubleshooting below.

### 5. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## 📦 Core Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Frontend UI |
| `mftool` (v3.3 API) | Mutual fund NAV data via AMFI |
| `pandas` (3.x) | Data processing |
| `numpy` | Numerical computations |
| `scipy` | Skewness, kurtosis, OLS regression |
| `arch` | GARCH(1,1) volatility forecasting |
| `plotly` | Interactive charts |
| `requests` | AMFI/mfapi fallback, NSE TRI downloader |

See `requirements.txt` for pinned versions.

---

## 📊 Supported Fund Categories

| Category | Benchmark |
|---|---|
| Large Cap | Nifty 100 TRI |
| Mid Cap | Nifty Midcap 150 TRI |
| Small Cap | Nifty Smallcap 250 TRI |
| Flexi Cap / Multi Cap / ELSS / Value / Contra / Focused | Nifty 500 TRI |
| Aggressive Hybrid / Balanced Advantage | Nifty 50 TRI |
| Index Funds | Tracked index (no separate benchmark) |

All benchmarks resolve TRI-first, with automatic silent fallback to an index-fund NAV proxy if TRI data is temporarily unavailable for a category.

---

## 🔧 Troubleshooting

### "No schemes returned" error
Run `python debug_connection.py` — it identifies which URL is blocked.

**Common fixes:**
- Disable VPN
- Switch to personal WiFi / mobile hotspot
- Run `pip install --upgrade certifi requests`
- Add Python to Windows Defender / antivirus exceptions

### Pandas version errors (`applymap`, `infer_datetime_format`)
This project targets **pandas 3.x**, which removed `applymap` (use `.map()` on Styler objects) and `infer_datetime_format`. Upgrade if you see these:
```bash
pip install --upgrade pandas
```

### Slow first load
The first run fetches NAV history for every fund in a category. Results are cached — subsequent loads are much faster.

---

## 🏗️ Architecture

```
mftool / mfapi.in ─────────────────┐
niftyindices.com (TRI) ──indices/──┤
                                    ↓
                    data/fund_loader.py, tri_loader.py, benchmark_loader.py
                                    ↓
                    data/nav_processor.py   (NAV cleaning, returns)
                                    ↓
                    analytics/engine.py     (64-metric orchestrator)
                    analytics/factor_model.py, garch_model.py, monte_carlo.py
                                    ↓
                    visualizations/*.py     (Plotly chart builders)
                                    ↓
                    pages/*.py              (Streamlit UI)
```

---

## 📈 Quantitative Methodology (selected)

### CAGR
```
CAGR = (End NAV / Start NAV) ^ (1 / actual_years) - 1
```

### Sharpe Ratio
```
Sharpe = mean(daily_return - rf_daily) / std(daily_return - rf_daily) × √252
```

### Sortino Ratio
```
Sortino = annualised_excess_return / (std(returns below MAR) × √252)
```

### Maximum Drawdown
```
MDD = min((NAV_t - max(NAV_0..NAV_t)) / max(NAV_0..NAV_t))
```

### 6-Factor Model
Market, SMB, HML, WML, QMJ, BAB constructed from TRI index differences (e.g. SMB = Smallcap250 TRI − Nifty100 TRI). Betas are standardised (zero mean, unit variance, full-sample) for cross-fund and cross-factor comparability; raw betas are used separately for return attribution.

### Predictive Analytics
GARCH(1,1) models conditional volatility and produces 30/60/90-day forecasts, VaR/CVaR, and volatility persistence — a risk estimate, not a return forecast. Monte Carlo simulation uses block bootstrap (21-day blocks) on actual historical returns rather than a parametric distribution, preserving fat tails, skew, and volatility clustering.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Data Sources

- **AMFI India** — [amfiindia.com](https://www.amfiindia.com) — Official NAV data
- **mfapi.in** — Free open API for Indian mutual fund data (fallback)
- **mftool** — Python library for AMFI data access
- **niftyindices.com** — Total Return Index (TRI) data for benchmarking and factor construction
