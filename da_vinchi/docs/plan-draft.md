
# Stage 0 — Data hygiene (non-negotiable)

**Why:** avoid leakage, misalignment, and garbage-in.

**Do:**

* Use **adjusted prices** (splits/dividends). Align all instruments to a common trading calendar; forward-fill only **truly stale** features (e.g., last earnings date), never returns.
* Winsorize features at e.g. 1st/99th pct; optional robust z-scores.
* Use **as-of times**: if a feature isn’t available by T, don’t use it to predict T+1.

**Carry forward:** clean OHLCV per instrument (O,H,L,C,AdjC,V), timestamps, corporate actions.

---

# Stage 1 — Baseline OHLCV features (per instrument)

Start here. These alone can carry a lot of signal.

### 1.1 Returns & volatility

* Simple return: $r_t = \frac{C_t - C_{t-1}}{C_{t-1}}$
* Log return: $\ell_t = \ln(C_t/C_{t-1})$
* Rolling returns: $R_{t}^{(n)} = \sum_{k=1}^{n} \ell_{t-k+1}$
* Realized vol (close-to-close): $\sigma_{t}^{(n)} = \sqrt{\frac{1}{n-1}\sum_{k=1}^{n}( \ell_{t-k+1} - \bar{\ell})^2}$

**High–low estimators (better with gaps):**

* Parkinson: $\sigma_{HL} = \sqrt{\frac{1}{4n\ln 2}\sum_{k=1}^{n} \ln^2(H_{t-k+1}/L_{t-k+1})}$
* Garman–Klass (uses O,C): $\hat{\sigma}^2 = \frac{1}{n}\sum\left[0.5\ln^2(H/L) - (2\ln2 - 1)\ln^2(C/O)\right]$

### 1.2 Trend/momentum

* **SMA/EMA:** $\text{EMA}_t^{(n)} = \alpha C_t + (1-\alpha)\text{EMA}_{t-1}^{(n)}, \ \alpha=\frac{2}{n+1}$
* **MACD:** $\text{EMA}^{12} - \text{EMA}^{26}$; Signal = EMA$^9$(MACD)
* **RSI (n=14):** Wilder smoothing
  Gains\_t = max(ΔC, 0); Losses\_t = max(-ΔC, 0)
  $\text{RS}_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}$; $\text{RSI}_t = 100 - \frac{100}{1+\text{RS}_t}$
* **Stoch %K:** $100\cdot\frac{C_t - \min(L)_{n}}{\max(H)_{n}-\min(L)_{n}}$, %D = SMA$_3$(%K)
* **ADX (trend strength):** compute +DI, −DI from true range; ADX = EMA of |+DI − −DI|/(+DI+−DI)

### 1.3 Range, bands, volatility-of-price

* **ATR (n=14):** TR = max(H−L, |H−C$_{t-1}$|, |L−C$_{t-1}$|); ATR = Wilder EMA(TR)
* **Bollinger:** Middle=SMA$_n$, Upper=Middle+2·Std$_n$, Lower=Middle−2·Std$_n$
  Width = (Upper−Lower)/Middle; %B = (C−Lower)/(Upper−Lower)

### 1.4 Liquidity & microstructure

* Dollar volume: $\text{DV} = C \cdot V$
* Turnover: $V/\text{FreeFloat}$ (if available)
* **Amihud illiquidity:** $\frac{1}{n}\sum_{k=1}^{n}\frac{|r_{t-k+1}|}{\text{DV}_{t-k+1}}$
* **Roll spread (if no quotes):** $\text{Roll} = 2\sqrt{-\text{Cov}(\Delta C_t, \Delta C_{t-1})}$ (if covariance < 0)

**Outputs to carry:** a tidy per-instrument feature frame at daily (or chosen) frequency.

**Pseudocode (pandas-style):**

```python
# assume df: [date, instrument, open, high, low, close, adj_close, volume] sorted
g = df.groupby('instrument', group_keys=False)

df['log_ret'] = g['adj_close'].apply(lambda s: np.log(s/s.shift(1)))
df['vol_20']  = g['log_ret'].apply(lambda s: s.rolling(20).std() * np.sqrt(252))
df['ema_12']  = g['adj_close'].apply(lambda s: s.ewm(span=12, adjust=False).mean())
df['ema_26']  = g['adj_close'].apply(lambda s: s.ewm(span=26, adjust=False).mean())
df['macd']    = df['ema_12'] - df['ema_26']
df['macd_sig']= g['macd'].apply(lambda s: s.ewm(span=9, adjust=False).mean())

# RSI
def rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))
df['rsi_14'] = g['adj_close'].apply(rsi)
```

---

# Stage 2 — Cross-sectional risk & market context (per instrument, needs a benchmark)

Use a market/sector index to contextualize each instrument.

### 2.1 Rolling beta/alpha (vs benchmark $m$)

* Beta: $\beta_t^{(n)} = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)}$ over a window $n$
* Alpha (residual return): $\hat{\alpha}_t = r_{i,t} - \beta_t r_{m,t}$
* Idiosyncratic vol: std of regression residuals over window

### 2.2 Cross-sectional ranks (daily)

* Rank or z-score each feature **within the universe** per day (robust to scale)
* Examples: momentum rank (R$_{t}^{(126)}$), value rank (if fundamentals), liquidity rank

**Outputs to carry:** beta, residuals, idio vol, daily cross-sectional ranks/z-scores.

**Pseudocode:**

```python
# rolling beta
def rolling_beta(ret_i, ret_m, n=60):
    cov = ret_i.rolling(n).cov(ret_m)
    var = ret_m.rolling(n).var()
    return cov / (var + 1e-12)

bench = df[df.instrument=='^BENCH'][['date','log_ret']].rename(columns={'log_ret':'mret'})
df = df.merge(bench, on='date', how='left')
df['beta_60'] = g.apply(lambda x: rolling_beta(x['log_ret'], x['mret']))
df['alpha']   = df['log_ret'] - df['beta_60']*df['mret']
```

---

# Stage 3 — Regimes & seasonal structure (uses Stage 1–2)

### 3.1 Regime labels

* Make a compact regime feature using only **contemporaneous** features (no lookahead):

  * Inputs: recent vol (e.g., $\sigma_{20}$), trend (EMA spreads), liquidity z-scores
  * Cluster: k-means (k=2–4) or HMM → label {low-vol, high-vol, trending, choppy}
* Use a 3–5 day majority filter to de-noise labels.

### 3.2 Seasonality/time encodings

* Day-of-week/month as cyclical:
  $\sin(2\pi \cdot \text{dow}/7),\ \cos(2\pi \cdot \text{dow}/7)$
* Turn-of-month flag (last 3 trading days + first 3)
* Time-since-event counters (since last earnings/dividend if applicable)

**Outputs to carry:** `regime_id`, cyclical time features, event windows.

---

# Stage 4 — Instrument relationships: correlation, clustering, cointegration

This stage creates the **map** you’ll use to build cross-instrument features.

### 4.1 Rolling correlation network

* Compute rolling Spearman/Pearson correlations on log returns (window e.g. 90).
* Noise-reduce with shrinkage (e.g., Ledoit–Wolf) if you have many instruments.
* Distance: $d_{ij} = \sqrt{2(1-\rho_{ij})}$
* Cluster: hierarchical (Ward/average) → cluster id per instrument.
* **Peer list:** for each i, choose top-k peers with $\rho_{ij} \ge \tau$ (e.g., τ=0.4).

### 4.2 Lead–lag diagnostics

* Cross-correlation $\text{CCF}_{ij}(k) = \text{Corr}(r_i(t), r_j(t-k))$ for lags $k\in[1,5]$
  Keep pairs if |CCF| > threshold and significant (perm test or Bartlett’s approx).

### 4.3 Cointegration (pairs within cluster)

* Engle–Granger:

  1. OLS: $P_i = a + b P_j + \varepsilon$ on log‐prices
  2. ADF test on $\varepsilon$. If stationary → cointegrated.
* Build spread: $z = \frac{\varepsilon - \mu_\varepsilon}{\sigma_\varepsilon}$
* **Half-life** of mean reversion from AR(1):
  $z_t = \rho z_{t-1} + \eta_t \Rightarrow \text{HalfLife} = -\ln(2)/\ln|\rho|$

**Outputs to carry:** cluster\_id, peer lists, lead-lag lags, cointegrated pairs with $b$, z-score, half-life.

---

# Stage 5 — Cross-instrument enriched features (uses Stage 4 artifacts)

Now feed each instrument with information from its peers/leaders.

### 5.1 Peer aggregates

* Peer mean return: $\bar{r}^{\text{peer}}_{t} = \sum_{j\in \mathcal{P}(i)} w_{ij} r_{j,t}$, weights $w_{ij} \propto \rho_{ij}$
* Relative strength: $r_{i,t} - \bar{r}^{\text{peer}}_{t}$
* Peer volatility ratio: $\sigma_i/\overline{\sigma}_{\text{peer}}$
* Peer momentum gap: $\text{EMA\_fast}_i - \sum w_{ij}\text{EMA\_fast}_j$

### 5.2 Lead–lag features

* Include **lagged** peer returns at discovered lags: $r_{j,t-k^\*}$ for top pairs
* Rolling sum of positive lead signals above a threshold (count features)

### 5.3 Cointegration spread features

* Current z-score of spread $z_{ij,t}$
* **Reversion pressure:** $-z_{ij,t}/\text{HalfLife}_{ij}$ (faster expected snapback)
* Spread momentum: Δz, rolling mean of z, time since last |z|>2 event

**Outputs to carry:** per-instrument feature frame augmented with peer aggregates, lagged peer signals, and spread stats.

**Pseudocode (sketch):**

```python
# given peer_map[i] = list of (j, weight_ij)
def peer_agg(df_feat, peer_map, col='log_ret'):
    out = []
    for i, grp in df_feat.groupby('instrument'):
        peers = peer_map.get(i, [])
        tmp = grp[['date', 'instrument', col]].rename(columns={col: f'{col}_self'})
        for j, w in peers:
            pj = df_feat[df_feat.instrument==j][['date', col]].rename(columns={col: f'{col}_{j}'})
            tmp = tmp.merge(pj, on='date', how='left')
        # weighted mean across peer columns
        peer_cols = [c for c in tmp.columns if c.startswith(f'{col}_') and c not in (f'{col}_self',)]
        wts = np.array([w for (_, w) in peers])
        tmp['peer_ret'] = (tmp[peer_cols].values * wts).sum(axis=1) / (wts.sum()+1e-12)
        tmp['rel_str']  = tmp[f'{col}_self'] - tmp['peer_ret']
        out.append(tmp[['date','instrument','peer_ret','rel_str']])
    return pd.concat(out)
```

---

# Stage 6 — Alternative data (news, social, macro) → sentiment & event features

Ingest and aggregate **by instrument–day** with careful timestamps.

### 6.1 News sentiment

* Score each article with a finance-tuned model (e.g., finBERT). For instrument $i$ on day $t$:

  * Mean sentiment $S_{i,t} = \text{mean}(\text{score}_a)$
  * **Volume of news:** count$_{i,t}$; shock = z-score of count vs 60-day history
  * **Recency-weighted sentiment:** $\sum_a w_a \cdot \text{score}_a$, $w_a \propto e^{-\lambda \Delta t_a}$
  * Polarity split: pos, neg, neu proportions
* **Deltas:** $\Delta S_{i,t} = S_{i,t} - \text{EMA}_{10}(S_{i})_t$

### 6.2 Social/buzz (if available)

* Mentions$_{i,t}$, unique authors, engagement per post
* Social sentiment mean, std, and change; buzz z-score

### 6.3 Macro & fundamentals

* Map a small set of **global factors** to each instrument (depends on asset class):

  * Equities: yield-curve slope (10y−2y), credit spread, VIX, oil (for energy), USD index
  * FX: interest diff, commodity proxies; Rates: term premium; Commodities: inventories/spreads
* Use **surprise** transforms where applicable: $\text{Surprise} = (Actual - Consensus)/\text{Std}$
* Company fundamentals (if available): earnings surprise, YoY revenue growth, valuation ratios (then cross-sectional rank).

**Outputs to carry:** daily sentiment/buzz aggregates, event flags (earnings, guidance, product), macro factor panel merged by date.

---

# Stage 7 — Targets & labeling (what you ultimately predict)

Create labels **after** all features are dated properly.

### 7.1 Forward returns (regression/classification)

* Horizon $h$ days: $y^{(h)}_t = \frac{C_{t+h} - C_t}{C_t}$
  (Create for h ∈ {1, 5, 10, 21})
* Classification labels: sign(y), or quantiles (top 30% = 1, bottom 30% = −1, else 0)

### 7.2 Volatility targets

* Realized vol ahead: $\sigma^{\text{fwd}}_{t} = \sqrt{\sum_{k=1}^{h}\ell_{t+k}^2}$
* Regime class (low/med/high) by quantiles of $\sigma^{\text{fwd}}$

### 7.3 “Tradeability”/profitability proxies

* **n-day max adverse excursion (MAE) & max favorable excursion (MFE):** within \[t+1, t+h]
* Breakout continuation: label=1 if C$_t$ > N-day high **and** C$_{t+h}$ > C$_t$

**Pseudocode:**

```python
for h in [1,5,10,21]:
    df[f'y_ret_{h}d'] = g['adj_close'].apply(lambda s: s.shift(-h)/s - 1.0)
df['y_cls_5d'] = pd.qcut(df['y_ret_5d'], q=[0, .3, .7, 1], labels=[-1,0,1])
```

---

# Stage 8 — Model-ready dataset assembly

**Do:**

* **Time splits:** expanding-window CV (e.g., train 2016–2021, val 2022, test 2023, roll forward).
* **Scaling:** fit scalers on train only. For tree models, scaling optional; for linear/NN, use robust scaling.
* **Leakage audit:** every column’s timestamp ≤ label’s origin time. Remove any that peek ahead.
* **Collinearity:** drop one of any pair |corr| > 0.95; optionally VIF screening.
* **Missingness:** impute per feature with train-period medians or industry medians; never forward-fill returns.

**Sanity checks (quick wins):**

* Monotonic bins: does higher RSI or momentum rank associate with higher forward return? (plot mean y by decile)
* Simple baselines: predict with 3–5 strong features; make sure you beat them before going fancy.

---

# Feature shortlist (to compute first, in this order)

1. **Stage 1 core:** log\_ret, vol\_20, EMA(12/26), MACD & signal, RSI(14), ATR(14), Bollinger %B, DollarVol, Amihud.
2. **Stage 2:** beta\_60 vs benchmark, alpha (residual), idio\_vol\_60, cross-sectional ranks of momentum(126d), vol\_20, DollarVol.
3. **Stage 3:** regime\_id (k-means on \[vol\_20, MACD, DollarVol\_z]), DoW/MoY sin/cos, turn-of-month.
4. **Stage 4:** corr\_90 matrix → clusters, peer\_map (top-k by ρ), lead-lag lags, cointegration pairs with spread z & half-life.
5. **Stage 5:** peer\_ret (weighted), rel\_strength, lagged\_peer\_ret at discovered lags, spread\_z, reversion\_pressure.
6. **Stage 6:** news sentiment mean, sentiment Δ vs EMA10, news\_count\_z, event flags (earnings ±3d), macro factors (VIX, curve slope) + deltas.
7. **Stage 7:** labels y\_ret\_{1,5,21}, vol\_fwd\_21, cls labels (quantiles).

---

# Common pitfalls (so you don’t step on them)

* **Lookahead bias:** any rolling statistic must use data ≤ t. No re-fit on the whole history per day.
* **Survivorship bias:** use historical constituents; delisted names matter.
* **Multiple testing:** thousands of features will “work” in-sample. Use honest OOS and penalize complexity.
* **Non-stationarity:** re-tune/re-select features periodically; compare by regime.

---

If you want, I can generate a **ready-to-run pandas notebook** that computes Stages 1–3 from a simple `[date, instrument, open, high, low, close, adj_close, volume]` CSV and outputs a feature table you can plug into the later stages.
