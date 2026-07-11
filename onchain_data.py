"""
onchain_data.py — data layer + anomaly detection
=================================================
Shared functions used by both the CLI analysis script and the Streamlit dashboard.

Data sources (all public, no key, reachable without a proxy):
  - DefiLlama          https://api.llama.fi
  - DefiLlama stables  https://stablecoins.llama.fi

Signature method: rolling mean ± k·σ anomaly detection on daily changes.
This is the same statistical approach used in my quant-research automation project
(funding-rate / order-flow anomaly extraction), applied here to on-chain aggregates.
"""

import time
import requests
import pandas as pd
import numpy as np

LLAMA = "https://api.llama.fi"
STABLES = "https://stablecoins.llama.fi"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


def _get(url, retries=3, backoff=1.5):
    """GET JSON with retries + exponential backoff — robust to transient
    SSL/connection drops (common when routed through a proxy)."""
    last = None
    for attempt in range(retries):
        try:
            r = _SESSION.get(url, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    raise last


# ----------------------------------------------------------------------
# Raw data
# ----------------------------------------------------------------------
def chain_tvl():
    """Current DeFi TVL per chain, sorted desc. -> DataFrame[name, tvl, share_%]"""
    df = pd.DataFrame(_get(f"{LLAMA}/v2/chains"))[["name", "tvl"]].dropna()
    df = df[df["tvl"] > 0].sort_values("tvl", ascending=False).reset_index(drop=True)
    total = df["tvl"].sum()
    df["share_%"] = (df["tvl"] / total * 100).round(2)
    return df


def stablecoin_supply():
    """Current circulating supply per stablecoin. -> DataFrame[symbol, name, circulating, share_%]"""
    assets = _get(f"{STABLES}/stablecoins?includePrices=false")["peggedAssets"]
    rows = []
    for a in assets:
        usd = (a.get("circulating") or {}).get("peggedUSD", 0) or 0
        if usd > 0:
            rows.append({"symbol": a["symbol"], "name": a["name"], "circulating": usd})
    df = pd.DataFrame(rows).sort_values("circulating", ascending=False).reset_index(drop=True)
    total = df["circulating"].sum()
    df["share_%"] = (df["circulating"] / total * 100).round(2)
    return df


def stablecoin_history():
    """Total stablecoin circulating supply over time. -> DataFrame[date, total]"""
    hist = pd.DataFrame(_get(f"{STABLES}/stablecoincharts/all"))
    hist["date"] = pd.to_datetime(hist["date"].astype(int), unit="s")
    hist["total"] = hist["totalCirculatingUSD"].apply(
        lambda d: d.get("peggedUSD", 0) if isinstance(d, dict) else 0
    )
    return hist[hist["total"] > 0][["date", "total"]].reset_index(drop=True)


def chain_tvl_history(chain="Ethereum"):
    """Historical TVL for one chain. -> DataFrame[date, tvl]"""
    data = _get(f"{LLAMA}/v2/historicalChainTvl/{chain}")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    return df[["date", "tvl"]].reset_index(drop=True)


# ----------------------------------------------------------------------
# Protocol-level data (for chain drill-down)
# ----------------------------------------------------------------------
# The /v2/chains endpoint and the /protocols chainTvls keys don't always use the
# same chain name (e.g. chains says "BSC", protocols say "Binance"). Map display
# name -> chainTvls key here.
_CHAIN_KEY_ALIAS = {
    "BSC": "Binance",
    "OP Mainnet": "Optimism",
    "Zksync Era": "zkSync Era",
}


def _chain_key(chain, protocols):
    """Resolve the chainTvls key that matches a display chain name."""
    # exact match present in any protocol?
    for p in protocols:
        if chain in (p.get("chainTvls") or {}):
            return chain
    return _CHAIN_KEY_ALIAS.get(chain, chain)


def all_protocols():
    """Full protocol list with per-chain TVL breakdown. Cached-friendly raw pull."""
    return _get(f"{LLAMA}/protocols")


def chain_names(min_tvl=2e8, protocols=None):
    """List of chain names worth exploring (current TVL >= min_tvl), largest first."""
    df = chain_tvl()
    return df[df["tvl"] >= min_tvl]["name"].tolist()


def chain_protocols(chain, protocols=None, exclude_cex=True, top=15):
    """
    Top protocols on a given chain, ranked by that chain's TVL slice.
    -> DataFrame[name, category, tvl_on_chain, share_%]
    """
    protocols = protocols or all_protocols()
    key = _chain_key(chain, protocols)
    rows = []
    for p in protocols:
        tvl = (p.get("chainTvls") or {}).get(key)
        if not isinstance(tvl, (int, float)) or tvl <= 0:
            continue
        cat = p.get("category") or "Other"
        if exclude_cex and cat in ("CEX", "Chain"):
            continue
        rows.append({"name": p["name"], "category": cat, "tvl_on_chain": tvl})
    if not rows:
        return pd.DataFrame(columns=["name", "category", "tvl_on_chain", "share_%"])
    df = pd.DataFrame(rows).sort_values("tvl_on_chain", ascending=False).reset_index(drop=True)
    total = df["tvl_on_chain"].sum()
    df["share_%"] = (df["tvl_on_chain"] / total * 100).round(2) if total else 0
    return df.head(top)


def chain_category_breakdown(chain, protocols=None, exclude_cex=True):
    """TVL grouped by protocol category on a chain. -> DataFrame[category, tvl]"""
    protocols = protocols or all_protocols()
    key = _chain_key(chain, protocols)
    cats = {}
    for p in protocols:
        tvl = (p.get("chainTvls") or {}).get(key)
        if not isinstance(tvl, (int, float)) or tvl <= 0:
            continue
        cat = p.get("category") or "Other"
        if exclude_cex and cat in ("CEX", "Chain"):
            continue
        cats[cat] = cats.get(cat, 0) + tvl
    return pd.DataFrame(sorted(cats.items(), key=lambda i: -i[1]),
                        columns=["category", "tvl"]).reset_index(drop=True)


def chain_dex_history(chain):
    """Daily DEX volume for a chain. -> DataFrame[date, volume] (empty if unavailable)."""
    try:
        d = _get(f"{LLAMA}/overview/dexs/{chain.lower()}?excludeTotalDataChartBreakdown=true")
    except Exception:
        return pd.DataFrame(columns=["date", "volume"])
    chart = d.get("totalDataChart") or []
    if not chart:
        return pd.DataFrame(columns=["date", "volume"])
    df = pd.DataFrame(chart, columns=["date", "volume"])
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    return df


def stablecoins_by_chain():
    """Stablecoin circulating supply per chain. -> DataFrame[name, total]"""
    data = _get(f"{STABLES}/stablecoinchains")
    rows = []
    for c in data:
        usd = (c.get("totalCirculatingUSD") or {}).get("peggedUSD", 0) or 0
        if usd > 0:
            rows.append({"name": c["name"], "total": usd})
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Signature: rolling mean ± k·σ anomaly detection
# ----------------------------------------------------------------------
def detect_anomalies(df, value_col, date_col="date", window=30, k=2.0):
    """
    Flag days whose day-over-day RETURN (percent change) deviates more than k
    standard deviations from a trailing `window`-day rolling mean of returns.

    Detection runs on percent change (scale-invariant), not absolute change,
    so z-scores stay calibrated even when the series drifts a lot.

    Returns a copy of df with added columns:
      change      : day-over-day ABSOLUTE change in value_col (kept for display)
      ret         : day-over-day percent change (what detection runs on)
      roll_mean / roll_std / z : rolling stats of ret
      anomaly     : bool, |z| > k
      direction   : 'spike' / 'drop' / '' for flagged rows
    """
    out = df.sort_values(date_col).copy().reset_index(drop=True)
    out["change"] = out[value_col].diff()          # absolute, for USD display
    out["ret"] = out[value_col].pct_change()       # relative, for detection
    out["roll_mean"] = out["ret"].rolling(window, min_periods=window // 2).mean()
    out["roll_std"] = out["ret"].rolling(window, min_periods=window // 2).std()
    out["z"] = (out["ret"] - out["roll_mean"]) / out["roll_std"]
    out["anomaly"] = out["z"].abs() > k
    out["direction"] = ""
    out.loc[out["anomaly"] & (out["z"] > 0), "direction"] = "spike"
    out.loc[out["anomaly"] & (out["z"] < 0), "direction"] = "drop"
    return out


def anomaly_table(df_flagged, value_col):
    """Extract only the flagged rows as a clean summary table, most recent first."""
    a = df_flagged[df_flagged["anomaly"]].copy()
    a = a[["date", value_col, "change", "z", "direction"]].sort_values("date", ascending=False)
    a["z"] = a["z"].round(2)
    return a.reset_index(drop=True)


# ----------------------------------------------------------------------
# Quant analytics: multi-chain history, correlation, risk, concentration
# ----------------------------------------------------------------------
def multi_chain_history(chains, days=365):
    """
    Wide DataFrame of daily TVL for several chains, aligned on common dates.
    -> DataFrame indexed by date, one column per chain.
    """
    series = {}
    for c in chains:
        try:
            h = chain_tvl_history(c).set_index("date")["tvl"]
            series[c] = h
        except Exception:
            continue
    df = pd.DataFrame(series).dropna()
    if len(df):
        df = df.last(f"{days}D")
    return df


def correlation_matrix(wide_df):
    """Pearson correlation of daily TVL % changes across chains."""
    return wide_df.pct_change().dropna().corr()


def risk_metrics(wide_df):
    """
    Per-chain risk table from daily TVL:
      ann_vol_%     : annualized volatility of daily returns
      max_drawdown_%: worst peak-to-trough decline over the window
      cur_drawdown_%: how far below the running peak right now
      ret_30d_%     : trailing 30-day TVL change
    -> DataFrame indexed by chain.
    """
    ret = wide_df.pct_change().dropna()
    rows = {}
    for c in wide_df.columns:
        s = wide_df[c]
        dd = s / s.cummax() - 1
        rows[c] = {
            "ann_vol_%": round(ret[c].std() * np.sqrt(365) * 100, 1),
            "max_drawdown_%": round(dd.min() * 100, 1),
            "cur_drawdown_%": round(dd.iloc[-1] * 100, 1),
            "ret_30d_%": round((s.iloc[-1] / s.iloc[-31] - 1) * 100, 1) if len(s) > 31 else np.nan,
        }
    return pd.DataFrame(rows).T


def _wide_ffill(chains, days):
    """
    Like multi_chain_history but aligns on a common daily index and forward-fills
    (TVL is a stock, so carrying the last value forward is valid). Chains that
    simply didn't exist early on start as NaN and are dropped from those early dates.
    """
    series = {}
    for c in chains:
        try:
            series[c] = chain_tvl_history(c).set_index("date")["tvl"]
        except Exception:
            continue
    if not series:
        return pd.DataFrame()
    df = pd.DataFrame(series).sort_index()
    df = df.last(f"{days}D").ffill()
    return df


def hhi_over_time(chains=None, days=365, top_n=25):
    """
    Market concentration (Herfindahl index) of DeFi TVL over time.
    Uses the top_n chains by current TVL. -> DataFrame[date, hhi].
    """
    if chains is None:
        chains = chain_tvl().head(top_n)["name"].tolist()
    wide = _wide_ffill(chains, days).dropna(how="all")
    if wide.empty:
        return pd.DataFrame(columns=["date", "hhi"])
    wide = wide.fillna(0.0)
    shares = wide.div(wide.sum(axis=1), axis=0)
    hhi = (shares ** 2).sum(axis=1)
    return pd.DataFrame({"date": hhi.index, "hhi": hhi.values})


def capital_rotation(chains=None, days=90, top_n=15):
    """
    Which chains gained / lost market share over the window.
    -> DataFrame[chain, share_now_%, share_then_%, delta_pp] sorted by delta.
    """
    if chains is None:
        chains = chain_tvl().head(top_n)["name"].tolist()
    wide = multi_chain_history(chains, days=days + 5)
    if wide.empty or len(wide) < days:
        return pd.DataFrame(columns=["chain", "share_now_%", "share_then_%", "delta_pp"])
    shares = wide.div(wide.sum(axis=1), axis=0) * 100
    now, then = shares.iloc[-1], shares.iloc[-days]
    out = pd.DataFrame({
        "chain": shares.columns,
        "share_now_%": now.round(2).values,
        "share_then_%": then.round(2).values,
    })
    out["delta_pp"] = (out["share_now_%"] - out["share_then_%"]).round(2)
    return out.sort_values("delta_pp", ascending=False).reset_index(drop=True)


def rebased_trajectories(wide_df):
    """Rebase each chain's TVL to 100 at the window start for shape comparison."""
    return wide_df.div(wide_df.iloc[0]) * 100


# ----------------------------------------------------------------------
# Stablecoin de-peg monitoring (risk view)
# ----------------------------------------------------------------------
# DefiLlama price keys differ from ticker symbols; map the ones we care about.
_PEG_KEYS = {
    "USDT": "tether", "USDC": "usd-coin", "DAI": "dai",
    "USDe": "ethena-usde", "FDUSD": "first-digital-usd",
    "TUSD": "true-usd", "USDS": "usds", "PYUSD": "paypal-usd",
}


def stablecoin_prices(symbols=("USDT", "USDC", "DAI", "USDe"), days=365):
    """
    Daily price history for selected stablecoins.
    -> DataFrame indexed by date, one column per symbol (price in USD).
    """
    raw = _get(f"{STABLES}/stablecoinprices")
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    out = {"date": df["date"]}
    for sym in symbols:
        key = _PEG_KEYS.get(sym)
        out[sym] = df["prices"].apply(lambda p: p.get(key) if isinstance(p, dict) else None)
    res = pd.DataFrame(out).set_index("date").dropna(how="all")
    return res.last(f"{days}D")


def depeg_events(price_df, threshold=0.005):
    """
    Flag days a stablecoin deviated from $1 by more than `threshold` (e.g. 0.5%).
    -> DataFrame[date, symbol, price, deviation_%, direction] sorted by |deviation|.
    """
    rows = []
    for sym in price_df.columns:
        s = price_df[sym].dropna()
        dev = s - 1.0
        hit = dev[dev.abs() > threshold]
        for dt, d in hit.items():
            rows.append({
                "date": dt.date(), "symbol": sym, "price": round(s[dt], 4),
                "deviation_%": round(d * 100, 3),
                "direction": "premium" if d > 0 else "discount",
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.reindex(df["deviation_%"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# TVL forecast: OLS linear trend + prediction interval
# ----------------------------------------------------------------------
def forecast_tvl(hist_df, value_col="tvl", horizon=30, fit_days=120, ci=0.95):
    """
    Fit an ordinary-least-squares linear trend on the last `fit_days` of data and
    project `horizon` days forward with a prediction interval.

    Returns (fit_df, future_df):
      fit_df    : the data used for fitting [date, actual, fitted]
      future_df : [date, forecast, lower, upper]
    Method note: prediction interval uses the standard OLS formula
      ŷ ± t · s · sqrt(1 + 1/n + (x0 - x̄)² / Σ(x - x̄)²)
    """
    d = hist_df.dropna(subset=[value_col]).tail(fit_days).reset_index(drop=True)
    n = len(d)
    x = np.arange(n, dtype=float)
    y = d[value_col].to_numpy(dtype=float)

    # OLS slope/intercept
    xbar, ybar = x.mean(), y.mean()
    sxx = ((x - xbar) ** 2).sum()
    slope = ((x - xbar) * (y - ybar)).sum() / sxx
    intercept = ybar - slope * xbar
    yhat = intercept + slope * x

    # residual standard error
    dof = max(n - 2, 1)
    s = np.sqrt(((y - yhat) ** 2).sum() / dof)
    # t multiplier (normal approx; ~1.96 at 95%) — avoids a scipy dependency
    from math import erf, sqrt
    # invert standard normal for common CIs
    t_mult = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(round(ci, 2), 1.960)

    last_date = d["date"].iloc[-1]
    fx = np.arange(n, n + horizon, dtype=float)
    fdates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    fcast = intercept + slope * fx
    se_pred = s * np.sqrt(1 + 1 / n + (fx - xbar) ** 2 / sxx)
    lower = fcast - t_mult * se_pred
    upper = fcast + t_mult * se_pred

    fit_df = pd.DataFrame({"date": d["date"], "actual": y, "fitted": yhat})
    future_df = pd.DataFrame({"date": fdates, "forecast": fcast, "lower": lower, "upper": upper})
    return fit_df, future_df


# ----------------------------------------------------------------------
# Protocol-level deep dive
# ----------------------------------------------------------------------
def protocol_list(min_tvl=5e8, protocols=None):
    """Protocol slugs worth exploring (current TVL >= min_tvl). -> DataFrame[name, slug, category, tvl]."""
    protocols = protocols or all_protocols()
    rows = []
    for p in protocols:
        if (p.get("tvl") or 0) >= min_tvl and p.get("slug"):
            rows.append({"name": p["name"], "slug": p["slug"],
                         "category": p.get("category", "?"), "tvl": p["tvl"]})
    return pd.DataFrame(rows).sort_values("tvl", ascending=False).reset_index(drop=True)


def chain_protocol_detail(chain, slugs, horizons=(1, 5, 7, 15, 30)):
    """
    For a set of protocol slugs, pull chain-specific history and return:
      name, first_seen (date it first appeared on this chain), tvl_now,
      and one % change column per horizon ('1d','5d',...).
    One API call per protocol (cache upstream).
    """
    rows = []
    for slug in slugs:
        try:
            d = _get(f"{LLAMA}/protocol/{slug}")
        except Exception:
            continue
        ctvls = d.get("chainTvls") or {}
        key = chain if chain in ctvls else _CHAIN_KEY_ALIAS.get(chain, chain)
        series = (ctvls.get(key) or {}).get("tvl") or []
        if not series:
            continue
        s = pd.DataFrame(series)
        s["date"] = pd.to_datetime(s["date"].astype(int), unit="s")
        s = s.rename(columns={"totalLiquidityUSD": "tvl"})
        v = s["tvl"].to_numpy()
        now = v[-1]
        rec = {
            "name": d.get("name", slug),
            "first_seen": s["date"].iloc[0].date(),
            "tvl_now": now,
        }
        for h in horizons:
            rec[f"{h}d"] = round((now / v[-1 - h] - 1) * 100, 1) if len(v) > h else float("nan")
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("tvl_now", ascending=False).reset_index(drop=True)
    return df


def market_posture(chg30, lead_category, latest_vol, avg_vol):
    """
    Combine 3 signals (30d TVL flow direction, dominant category, DEX volume vs
    its average) into a one-line market-posture read. -> (label, explanation).
    """
    yield_cats = {"Liquid Staking", "Lending", "Staking Pool", "Restaking",
                  "Liquid Restaking", "CDP", "Yield", "Farm"}
    inflow = chg30 >= 2
    outflow = chg30 <= -2
    hot = (avg_vol is not None) and (latest_vol is not None) and (latest_vol > avg_vol)
    yield_led = lead_category in yield_cats

    if outflow and hot:
        return ("⚠️ De-risking — capital leaving",
                "TVL is falling while trading stays elevated: money is exiting and being sold. "
                "A risk-off tape worth watching closely.")
    if outflow:
        return ("😴 Quiet contraction",
                "TVL is drifting lower on subdued volume — a slow bleed rather than a panic. "
                "Often just consolidation, but confirm the trend.")
    if inflow and yield_led and not hot:
        return ("🛡️ Risk-off / yield-seeking",
                "Capital is flowing in but parking in staking/lending for yield while trading cools. "
                "Money wants income, not speculation — a defensive, income-driven posture.")
    if inflow and not yield_led and hot:
        return ("🔥 Risk-on / speculative",
                "Inflows plus rising DEX volume with trading-heavy categories leading — "
                "active, speculative capital. Momentum-friendly but higher risk.")
    if inflow and hot:
        return ("📈 Expanding & active",
                "Both capital and trading activity are rising — a broadly healthy, growing chain.")
    if inflow:
        return ("📥 Accumulating",
                "Capital is entering but trading is quiet — money positioning without much churn yet.")
    return ("➡️ Range-bound",
            "No strong flow or volume signal in either direction — a balanced, wait-and-see tape.")


def protocol_tvl_history(slug):
    """Historical total TVL for one protocol. -> DataFrame[date, tvl]."""
    try:
        d = _get(f"{LLAMA}/protocol/{slug}")
    except Exception:
        return pd.DataFrame(columns=["date", "tvl"])
    tvl = d.get("tvl") or []
    df = pd.DataFrame(tvl)
    if df.empty or "totalLiquidityUSD" not in df.columns:
        return pd.DataFrame(columns=["date", "tvl"])
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    return df.rename(columns={"totalLiquidityUSD": "tvl"})[["date", "tvl"]]


def period_changes(wide_df, horizons=(1, 5, 7, 15, 30, 60)):
    """
    Percent change of each column over several trailing horizons.
    -> DataFrame indexed by entity, columns like '1d','5d',... in %.
    """
    rows = {}
    for col in wide_df.columns:
        s = wide_df[col].dropna()
        r = {}
        for h in horizons:
            if len(s) > h:
                r[f"{h}d"] = round((s.iloc[-1] / s.iloc[-1 - h] - 1) * 100, 1)
            else:
                r[f"{h}d"] = float("nan")
        rows[col] = r
    return pd.DataFrame(rows).T


def largest_chain_flows(chains=None, days=30, top=20):
    """
    Whale-sized flow monitor at the CHAIN level: the largest single-day net TVL
    moves (in USD) across the selected chains over the window.
    -> DataFrame[date, chain, flow_usd, pct, direction] ranked by |flow|.
    Note: this is aggregate net flow (a proxy for big-money movement), not
    per-wallet whale tracking, which needs an on-chain indexer.
    """
    if chains is None:
        chains = chain_tvl().head(15)["name"].tolist()
    wide = multi_chain_history(chains, days=days + 5)
    if wide.empty:
        return pd.DataFrame(columns=["date", "chain", "flow_usd", "pct", "direction"])
    diffs = wide.diff().tail(days)
    rows = []
    for chain in wide.columns:
        prev = wide[chain].shift(1)
        for dt, val in diffs[chain].dropna().items():
            base = prev.get(dt, None)
            pct = (val / base * 100) if base else float("nan")
            rows.append({"date": dt.date(), "chain": chain, "flow_usd": val,
                         "pct": round(pct, 1),
                         "direction": "inflow" if val > 0 else "outflow"})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.reindex(df["flow_usd"].abs().sort_values(ascending=False).index).head(top).reset_index(drop=True)
    return df


# A chain's TVL is dominated by assets denominated in its major token; comparing
# TVL change to that token's price change separates real flow from price effect.
_CHAIN_COIN = {
    "Ethereum": "ethereum", "Solana": "solana", "BSC": "binancecoin",
    "Tron": "tron", "Bitcoin": "bitcoin", "Avalanche": "avalanche-2",
    "Base": "ethereum", "Arbitrum": "ethereum", "OP Mainnet": "ethereum",
    "Polygon": "matic-network",
}
CG = "https://api.coingecko.com/api/v3"


def coingecko_change(coin_id, days=30):
    """% price change of a CoinGecko asset over `days` (daily data)."""
    d = _get(f"{CG}/coins/{coin_id}/market_chart?vs_currency=usd&days={days + 1}&interval=daily")
    prices = [p[1] for p in d.get("prices", [])]
    if len(prices) < 2:
        return None
    return (prices[-1] / prices[0] - 1) * 100


def flow_vs_price(chain, days=30):
    """
    Decompose a chain's TVL change into price effect vs real capital flow:
      tvl_chg   = % change in chain TVL over `days`
      price_chg = % change in the chain's dominant token over `days`
      real_flow = tvl_chg - price_chg  (implied deposit/withdrawal component)
    -> dict or None if the chain has no token mapping / data.
    """
    coin = _CHAIN_COIN.get(chain)
    if not coin:
        return None
    h = chain_tvl_history(chain)
    if len(h) <= days:
        return None
    tvl_chg = (h["tvl"].iloc[-1] / h["tvl"].iloc[-1 - days] - 1) * 100
    price_chg = coingecko_change(coin, days)
    if price_chg is None:
        return None
    return {"chain": chain, "token": coin, "days": days,
            "tvl_chg": round(tvl_chg, 1), "price_chg": round(price_chg, 1),
            "real_flow": round(tvl_chg - price_chg, 1)}


def net_flows(days=30, chains=None):
    """
    True net capital flow over the SAME window for both signals (not a sum of
    extreme days), so they are directly comparable:
      chain_net  = change in combined TVL of the top chains over `days`
      stable_net = change in total stablecoin supply over `days` (pegged = clean cash)
    -> (chain_net, stable_net) in USD.
    """
    sh = stablecoin_history().sort_values("date")
    stable_net = (sh["total"].iloc[-1] - sh["total"].iloc[-1 - days]) if len(sh) > days else 0.0
    if chains is None:
        chains = chain_tvl().head(15)["name"].tolist()
    wide = multi_chain_history(chains, days=days + 5)
    chain_net = float((wide.iloc[-1] - wide.iloc[-1 - days]).sum()) if len(wide) > days else 0.0
    return chain_net, stable_net


def largest_stablecoin_flows(days=60, top=15, min_usd=3e8):
    """
    Whale/institutional flow monitor via stablecoins: the largest single-day
    mint/burn days in total stablecoin supply (pegged to 1 USD, so a clean flow).
    -> DataFrame[date, flow_usd, pct, direction] over min_usd, ranked by |flow|.
    """
    h = stablecoin_history()
    h = h.sort_values("date").copy()
    h["flow"] = h["total"].diff()
    h["pct"] = h["total"].pct_change() * 100
    win = h.tail(days).dropna(subset=["flow"])
    big = win[win["flow"].abs() >= min_usd].copy()
    big["direction"] = big["flow"].apply(lambda v: "mint (money in)" if v > 0 else "burn (money out)")
    big = big.reindex(big["flow"].abs().sort_values(ascending=False).index).head(top)
    return big.rename(columns={"flow": "flow_usd"})[["date", "flow_usd", "pct", "direction"]].reset_index(drop=True)


def protocol_fees_history(slug, data_type="dailyFees"):
    """
    Daily fees or revenue for a protocol.
    data_type: 'dailyFees' or 'dailyRevenue'. -> DataFrame[date, value] (empty if none).
    """
    try:
        d = _get(f"{LLAMA}/summary/fees/{slug}?dataType={data_type}")
    except Exception:
        return pd.DataFrame(columns=["date", "value"])
    chart = d.get("totalDataChart") or []
    if not chart:
        return pd.DataFrame(columns=["date", "value"])
    df = pd.DataFrame(chart, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"].astype(int), unit="s")
    return df


# ======================================================================
# Coin analysis (CoinGecko) — reuses the risk-metric methodology from my
# quant alpha-strategy project (Sharpe / Sortino / Calmar / drawdown / VaR).
# Binance funding-rate & order-flow data is geo-blocked here, so price/volume
# analytics run on CoinGecko, which is reachable without a proxy.
# ======================================================================
def coingecko_markets(n=30, vs="usd"):
    """Top n coins by market cap. -> DataFrame[id, symbol, name, price, market_cap, ...]."""
    d = _get(f"{CG}/coins/markets?vs_currency={vs}&order=market_cap_desc&per_page={n}&page=1"
             "&price_change_percentage=24h,7d,30d")
    df = pd.DataFrame(d)
    keep = ["id", "symbol", "name", "current_price", "market_cap", "total_volume",
            "price_change_percentage_24h_in_currency",
            "price_change_percentage_7d_in_currency",
            "price_change_percentage_30d_in_currency"]
    return df[[c for c in keep if c in df.columns]]


def coin_market_chart(coin_id, days=180, vs="usd"):
    """Daily price + volume history for a coin. -> DataFrame[date, price, volume]."""
    d = _get(f"{CG}/coins/{coin_id}/market_chart?vs_currency={vs}&days={days}&interval=daily")
    prices = pd.DataFrame(d.get("prices", []), columns=["ts", "price"])
    vols = pd.DataFrame(d.get("total_volumes", []), columns=["ts", "volume"])
    if prices.empty:
        return pd.DataFrame(columns=["date", "price", "volume"])
    df = prices.merge(vols, on="ts", how="left")
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    return df[["date", "price", "volume"]].reset_index(drop=True)


def coin_risk_metrics(returns, annual_factor=365, risk_free_rate=0.02):
    """
    Full risk-metric suite on a daily-return series (ported from my alpha-strategy
    project): Sharpe, Sortino, Calmar, max drawdown, annualized return/vol,
    win rate, profit factor, VaR(95%), CVaR(95%).
    """
    r = returns.dropna()
    if len(r) < 5:
        return {}
    mean_ret, std_ret = r.mean(), r.std()
    excess = mean_ret - risk_free_rate / annual_factor
    sharpe = excess / std_ret * np.sqrt(annual_factor) if std_ret > 0 else 0
    downside = r[r < 0].std()
    sortino = excess / downside * np.sqrt(annual_factor) if downside and downside > 0 else 0
    cum = (1 + r).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    annual_ret = (1 + mean_ret) ** annual_factor - 1
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0
    wins, losses = r[r > 0], r[r < 0]
    win_rate = len(wins) / len(r)
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    var_95 = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean() if len(r[r <= var_95]) else var_95
    return {
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2), "calmar": round(calmar, 2),
        "max_drawdown": round(max_dd * 100, 1), "annual_return": round(annual_ret * 100, 1),
        "annual_vol": round(std_ret * np.sqrt(annual_factor) * 100, 1),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "var_95": round(var_95 * 100, 2), "cvar_95": round(cvar_95 * 100, 2),
    }


def coin_beta_to(coin_df, benchmark_df):
    """Beta and correlation of a coin's daily returns vs a benchmark (e.g. BTC)."""
    a = coin_df.set_index("date")["price"].pct_change().dropna()
    b = benchmark_df.set_index("date")["price"].pct_change().dropna()
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 5:
        return None
    cov = j.cov().iloc[0, 1]
    var_b = j.iloc[:, 1].var()
    beta = cov / var_b if var_b else float("nan")
    corr = j.iloc[:, 0].corr(j.iloc[:, 1])
    return {"beta": round(beta, 2), "corr": round(corr, 2)}
