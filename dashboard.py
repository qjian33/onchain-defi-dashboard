"""
On-Chain DeFi Intelligence Dashboard  (enterprise edition)
==========================================================
Three modules over live DefiLlama data:
  1. Market Overview   — market structure, concentration, stablecoins
  2. Chain Explorer    — drill into a single chain: TVL trend + anomalies,
                          category mix, top protocols, DEX volume, stablecoins
  3. Stablecoin Monitor— supply growth + rolling mean ± kσ anomaly detection

Run locally:   streamlit run dashboard.py
Deploy free:   push to GitHub -> share.streamlit.io -> one-click deploy

Author: Qingqing Jian
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import onchain_data as od

st.set_page_config(page_title="On-Chain DeFi Intelligence", page_icon="📊", layout="wide")

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "font.size": 10,
    "figure.autolayout": True,
})
BLUE, RED, GREEN = "#2563eb", "#dc2626", "#059669"
PALETTE = ["#2563eb", "#0891b2", "#7c3aed", "#059669", "#d97706", "#db2777", "#64748b"]

# Streamlit markdown renders "$...$" as LaTeX, which mangles dollar amounts in prose.
# In narrative text use money() (no "$" symbol); charts/metrics/tables keep usd().
def money(x):
    ax = abs(x)
    if ax >= 1e9:  return f"{x/1e9:.1f}B"
    if ax >= 1e6:  return f"{x/1e6:.0f}M"
    if ax >= 1e3:  return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


def usd(x, _=None):
    ax = abs(x)
    if ax >= 1e9:  return f"${x/1e9:.1f}B"
    if ax >= 1e6:  return f"${x/1e6:.0f}M"
    if ax >= 1e3:  return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


# ---- cached data loaders (1h TTL) ------------------------------------
@st.cache_data(ttl=3600)
def L_chain_tvl():        return od.chain_tvl()
@st.cache_data(ttl=3600)
def L_stables():          return od.stablecoin_supply()
@st.cache_data(ttl=3600)
def L_stable_hist():      return od.stablecoin_history()
@st.cache_data(ttl=3600)
def L_protocols():        return od.all_protocols()
@st.cache_data(ttl=3600)
def L_chain_hist(c):      return od.chain_tvl_history(c)
@st.cache_data(ttl=3600)
def L_chain_dex(c):       return od.chain_dex_history(c)
@st.cache_data(ttl=3600)
def L_stable_chains():    return od.stablecoins_by_chain()
@st.cache_data(ttl=3600)
def L_multi_hist(names, days): return od.multi_chain_history(list(names), days=days)
@st.cache_data(ttl=3600)
def L_hhi(days):          return od.hhi_over_time(days=days)
@st.cache_data(ttl=3600)
def L_rotation(days):     return od.capital_rotation(days=days)
@st.cache_data(ttl=3600, show_spinner=False)
def L_prot_tvl(slug):     return od.protocol_tvl_history(slug)
@st.cache_data(ttl=3600, show_spinner=False)
def L_prot_fees(slug, dt): return od.protocol_fees_history(slug, dt)
@st.cache_data(ttl=3600)
def L_whale_chain(days):  return od.largest_chain_flows(days=days)
@st.cache_data(ttl=3600)
def L_whale_stable(days): return od.largest_stablecoin_flows(days=days)
@st.cache_data(ttl=3600)
def L_depeg_prices():     return od.stablecoin_prices(("USDT", "USDC", "DAI", "USDe"), days=30)
@st.cache_data(ttl=3600)
def L_net_flows(days):    return od.net_flows(days=days)
@st.cache_data(ttl=3600)
def L_flow_vs_price(chain, days): return od.flow_vs_price(chain, days)


# ======================================================================
# Header
# ======================================================================
st.title("📊 On-Chain DeFi Intelligence Dashboard")
st.caption("Live data from DefiLlama · multi-chain drill-down · rolling mean ± kσ anomaly detection · "
           "Python + Streamlit — by Qingqing Jian")

with st.sidebar:
    st.header("⚙️ Controls")
    window = st.slider("Anomaly rolling window (days)", 14, 90, 30)
    k = st.slider("Anomaly threshold (k · σ)", 1.5, 5.0, 3.0, step=0.1)
    exclude_cex = st.checkbox("Exclude CEX / custodial from chain view", value=True,
                              help="CEX wallets (Binance, OKX…) dominate raw TVL but aren't native DeFi. "
                                   "Excluding them gives a cleaner picture of on-chain DeFi activity.")
    st.markdown("---")
    st.markdown("**Anomaly method** — each day's return is scored against a strictly trailing "
                "baseline (the tested day never contaminates it) using **robust statistics**: median + MAD "
                "(×1.4826, σ-equivalent) instead of mean/std, so one large shock cannot inflate the scale and "
                "mask later anomalies. Days beyond **k·σ** are flagged. Evolved from the funding-rate / "
                "order-flow anomaly detection in my quant-research project.")

chains = L_chain_tvl()
total_tvl = chains["tvl"].sum()

tab_overview, tab_chain, tab_compare, tab_quant, tab_protocol, tab_whale, tab_stable = st.tabs(
    ["🌐  Market Overview", "🔗  Chain Explorer", "⚖️  Compare", "📈  Quant Analytics",
     "🔬  Protocol Deep Dive", "🐋  Whale Monitor", "💵  Stablecoin Monitor"]
)

# ======================================================================
# TAB 1 — Market Overview
# ======================================================================
with tab_overview:
    stables = L_stables()
    total_stable = stables["circulating"].sum()
    hhi = ((chains["tvl"] / total_tvl) ** 2).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total DeFi TVL", usd(total_tvl), f"{len(chains)} chains")
    c2.metric("Top-3 chain share", f"{chains['share_%'].head(3).sum():.0f}%")
    c3.metric("Stablecoin supply", usd(total_stable), f"{len(stables)} coins")
    c4.metric("Concentration (HHI)", f"{hhi:.3f}",
              "concentrated" if hhi > 0.25 else "competitive")

    lead = chains.iloc[0]
    top_stable = stables.iloc[0]
    st.info(
        f"**Summary.** DeFi holds **{money(total_tvl)} USD** across {len(chains)} chains, but it's a "
        f"concentrated market — **{lead['name']}** alone is {lead['share_%']:.0f}% and the top 3 chains "
        f"are {chains['share_%'].head(3).sum():.0f}% (HHI {hhi:.2f}, where >0.25 is 'concentrated'). "
        f"Stablecoins are a **{money(total_stable)} USD** market with **{top_stable['symbol']}** at "
        f"{top_stable['share_%']:.0f}% dominance — a single point of systemic concentration worth watching."
    )

    # ---- cross-tab executive summary: synthesize signals from several tabs ----
    with st.expander("📋 **Executive summary** — one read across every tab", expanded=True):
        bullets = []
        # 1) concentration (Overview / Quant)
        bullets.append(f"**Structure:** concentrated market (HHI {hhi:.2f}); {lead['name']} at "
                       f"{lead['share_%']:.0f}%, top-3 chains {chains['share_%'].head(3).sum():.0f}%.")
        # 2) stablecoin net flow = money entering/leaving crypto (Stablecoin / Whale)
        try:
            sh = L_stable_hist()
            net30 = (sh["total"].iloc[-1] - sh["total"].iloc[-31]) if len(sh) > 31 else 0
            flow_word = "entering crypto" if net30 >= 0 else "leaving crypto"
            bullets.append(f"**Capital flow:** stablecoin supply {('+' if net30>=0 else '')}{money(net30)} USD "
                           f"over 30 days — net money **{flow_word}** (clean, price-free signal).")
        except Exception:
            pass
        # 3) biggest whale move (Whale)
        try:
            cf = L_whale_chain(30)
            if len(cf):
                b = cf.iloc[0]
                bullets.append(f"**Whale activity:** largest single-day move was {money(abs(b['flow_usd']))} USD "
                               f"**{b['direction']}** on {b['chain']} ({b['date']}).")
        except Exception:
            pass
        # 4) capital rotation leader (Quant / Compare)
        try:
            rot = L_rotation(90)
            if len(rot):
                g, l = rot.iloc[0], rot.iloc[-1]
                bullets.append(f"**Rotation:** over 90 days **{g['chain']}** gained the most share "
                               f"({g['delta_pp']:+.1f}pp) and **{l['chain']}** lost the most ({l['delta_pp']:+.1f}pp).")
        except Exception:
            pass
        # 5) stablecoin peg health (Stablecoin risk)
        try:
            dp = od.depeg_events(L_depeg_prices(), threshold=0.005)
            if len(dp):
                w = dp.iloc[0]
                bullets.append(f"**Peg risk:** {len(dp)} de-peg day(s) in the last 30d; worst was "
                               f"{w['symbol']} at {w['deviation_%']:+.2f}% ({w['direction']}).")
            else:
                bullets.append("**Peg risk:** all monitored stablecoins held their peg (±0.5%) over 30 days — calm.")
        except Exception:
            pass
        for b in bullets:
            st.markdown("- " + b)
        st.caption("This box combines signals from the Chain, Compare, Quant, Whale and Stablecoin tabs into "
                   "a single market read — the kind of synthesis a desk wants first thing each morning.")

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.subheader("Top blockchains by DeFi TVL")
        top = chains.head(12)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top["name"][::-1], top["tvl"][::-1], color=BLUE)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(usd))
        st.pyplot(fig); plt.close(fig)
        st.caption(f"**Takeaway.** {top.iloc[0]['name']} leads with {money(top.iloc[0]['tvl'])} USD "
                   f"({top.iloc[0]['share_%']:.0f}%) — about {top.iloc[0]['tvl']/top.iloc[1]['tvl']:.1f}× the "
                   f"#2 chain {top.iloc[1]['name']}. These {len(top)} chains hold "
                   f"{top['share_%'].sum():.0f}% of all DeFi TVL.")
    with right:
        st.subheader("Stablecoin market share")
        top6 = stables.head(6)
        others = pd.DataFrame([{"symbol": "Others", "circulating": stables["circulating"][6:].sum()}])
        pie_df = pd.concat([top6[["symbol", "circulating"]], others], ignore_index=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(pie_df["circulating"], labels=pie_df["symbol"], autopct="%1.1f%%",
               colors=PALETTE, startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1})
        st.pyplot(fig); plt.close(fig)
        st.caption(f"**Takeaway.** {stables.iloc[0]['symbol']} dominates at {stables.iloc[0]['share_%']:.0f}%, "
                   f"then {stables.iloc[1]['symbol']} ({stables.iloc[1]['share_%']:.0f}%). The top two are "
                   f"{stables['share_%'].head(2).sum():.0f}% of the market — heavy reliance on a few issuers.")

    st.subheader("Stablecoin supply by chain (top 12)")
    sbc = L_stable_chains().head(12)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(sbc["name"], sbc["total"], color=GREEN)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig); plt.close(fig)
    st.caption(f"**Takeaway.** {sbc.iloc[0]['name']} hosts the most stablecoins "
               f"({money(sbc.iloc[0]['total'])} USD, {sbc.iloc[0]['total']/sbc['total'].sum()*100:.0f}% of the "
               f"top-12), about {sbc.iloc[0]['total']/sbc.iloc[1]['total']:.1f}× {sbc.iloc[1]['name']}. "
               "Where stablecoins sit is where trading and settlement actually happen.")

# ======================================================================
# TAB 2 — Chain Explorer  (single-chain drill-down)
# ======================================================================
with tab_chain:
    options = chains[chains["tvl"] >= 2e8]["name"].tolist()
    default_ix = options.index("Ethereum") if "Ethereum" in options else 0
    chain = st.selectbox("Select a chain to explore", options, index=default_ix)

    protocols = L_protocols()
    row = chains[chains["name"] == chain].iloc[0]
    hist = L_chain_hist(chain)
    dexh = L_chain_dex(chain)

    # headline metrics for this chain
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(f"{chain} TVL", usd(row["tvl"]), f"{row['share_%']:.1f}% of all DeFi")
    if len(hist) > 30:
        chg = (hist["tvl"].iloc[-1] / hist["tvl"].iloc[-31] - 1) * 100
        d2.metric("30-day TVL change", f"{chg:+.1f}%")
    if len(dexh):
        d3.metric("DEX volume (24h)", usd(dexh["volume"].iloc[-1]))
    sbc = L_stable_chains()
    sb_row = sbc[sbc["name"] == chain]
    if len(sb_row):
        d4.metric("Stablecoins on chain", usd(sb_row["total"].iloc[0]))

    # ---- auto summary + cross-signal market posture for this chain ----
    cat = od.chain_category_breakdown(chain, protocols, exclude_cex=exclude_cex)
    if len(hist) > 31 and len(cat):
        chg30 = (hist["tvl"].iloc[-1] / hist["tvl"].iloc[-31] - 1) * 100
        lead_cat = cat.iloc[0]
        cat_share = lead_cat["tvl"] / cat["tvl"].sum() * 100
        trend_word = "grown" if chg30 >= 0 else "contracted"
        st.info(
            f"**Summary.** {chain} holds **{money(row['tvl'])} USD** ({row['share_%']:.1f}% of all DeFi) and has "
            f"**{trend_word} {abs(chg30):.1f}%** over 30 days. Its DeFi activity is led by "
            f"**{lead_cat['category']}** ({cat_share:.0f}% of on-chain TVL), and the anomaly layer flags "
            "the specific days its TVL moved abnormally — useful for tying jumps to real events."
        )

        # cross-signal synthesis: flow direction + dominant category + volume vs avg
        latest_vol = dexh["volume"].iloc[-1] if len(dexh) else None
        avg_vol = dexh["volume"].tail(90).mean() if len(dexh) else None
        label, explanation = od.market_posture(chg30, lead_cat["category"], latest_vol, avg_vol)
        vol_txt = ""
        if latest_vol is not None:
            vs = "above" if latest_vol > avg_vol else "below"
            vol_txt = f" DEX volume is {vs} its 90-day average."
        st.success(
            f"### Market posture: {label}\n\n"
            f"{explanation}{vol_txt} "
            f"*(Read from 3 signals: 30-day flow **{chg30:+.1f}%**, "
            f"dominant category **{lead_cat['category']}**, and DEX volume vs. its average.)*"
        )

        # ---- real flow vs price decomposition (evidence, not inference) ----
        fvp = L_flow_vs_price(chain, 30)
        if fvp:
            st.markdown("#### Is the TVL move real deposits or just price?")
            e1, e2, e3 = st.columns(3)
            e1.metric("TVL change (30d)", f"{fvp['tvl_chg']:+.1f}%")
            e2.metric(f"{fvp['token'].upper()} price (30d)", f"{fvp['price_chg']:+.1f}%")
            e3.metric("Implied real flow", f"{fvp['real_flow']:+.1f}%",
                      help="TVL change minus price change = the deposit/withdrawal component")
            rf = fvp["real_flow"]
            if abs(rf) < 2:
                read = (f"**Almost entirely price.** {chain}'s TVL moved {fvp['tvl_chg']:+.1f}% while its main "
                        f"asset moved {fvp['price_chg']:+.1f}% — implied real flow is only {rf:+.1f}%. The TVL "
                        "change is repricing of already-locked assets, **not fresh capital**.")
            elif rf > 0:
                read = (f"**Genuine inflow.** TVL rose more than price ({fvp['tvl_chg']:+.1f}% vs "
                        f"{fvp['price_chg']:+.1f}%) — an implied **{rf:+.1f}% of real deposits** on top of price.")
            else:
                read = (f"**Hidden outflow.** Price rose {fvp['price_chg']:+.1f}% but TVL only "
                        f"{fvp['tvl_chg']:+.1f}% — implied **{rf:+.1f}% real capital left**, masked by the price gain. "
                        "Looking at TVL alone would have missed it.")
            st.info(read + " *Method: TVL% − dominant-token price% ≈ net deposit/withdrawal component "
                    "(a first-order approximation, since chains hold a mix of assets).*")

    st.markdown("---")

    # TVL history + anomaly detection for THIS chain
    st.subheader(f"{chain} — TVL trend with anomaly detection")
    if len(hist) > window:
        flagged = od.detect_anomalies(hist, "tvl", window=window, k=k)
        an = flagged[flagged["anomaly"]]
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(flagged["date"], flagged["tvl"], color=BLUE, linewidth=1.5, label="TVL")
        ax.fill_between(flagged["date"], flagged["tvl"], alpha=0.10, color=BLUE)
        ax.scatter(an["date"], an["tvl"], color=RED, s=26, zorder=5,
                   label=f"Anomaly (|z| > {k})")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
        ax.legend(loc="upper left")
        st.pyplot(fig); plt.close(fig)
        st.write(f"**{len(an)} anomalous days** flagged on {chain} "
                 f"(window={window}d, k={k}).")
        if len(an):
            last = an.sort_values("date").iloc[-1]
            chg = last["change"]
            prev = last["tvl"] - chg
            pct = chg / prev * 100 if prev else 0
            flow = "net inflow" if chg > 0 else "net outflow"
            arrow = "↑" if chg > 0 else "↓"
            st.caption(f"**Takeaway.** Most recent anomaly: {last['date'].date()} — TVL {arrow} "
                       f"**{money(chg)} USD ({pct:+.1f}%)** in a single day (z={last['z']:.1f}), a **{flow}**. "
                       "Caveat: a chain-level TVL move can be real deposits/withdrawals *or* just the USD price "
                       "of already-locked assets changing — netflow-specific data is needed to fully separate them.")
    else:
        st.info("Not enough history for anomaly detection on this chain.")

    # ---- 30-day TVL forecast (OLS trend + prediction interval) ----
    st.subheader(f"{chain} — 30-day TVL forecast (OLS trend + 95% band)")
    if len(hist) > 60:
        fit_df, fut_df = od.forecast_tvl(hist, "tvl", horizon=30, fit_days=120, ci=0.95)
        fig, ax = plt.subplots(figsize=(12, 4))
        recent = hist.tail(120)
        ax.plot(recent["date"], recent["tvl"], color=BLUE, linewidth=1.5, label="Actual")
        ax.plot(fit_df["date"], fit_df["fitted"], color="#94a3b8", linewidth=1, linestyle="--", label="Trend fit")
        ax.plot(fut_df["date"], fut_df["forecast"], color=GREEN, linewidth=1.8, label="Forecast")
        ax.fill_between(fut_df["date"], fut_df["lower"], fut_df["upper"],
                        color=GREEN, alpha=0.15, label="95% prediction interval")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
        ax.legend(loc="upper left", fontsize=8)
        st.pyplot(fig); plt.close(fig)
        now_v, f_v = fit_df["actual"].iloc[-1], fut_df["forecast"].iloc[-1]
        lo, hi = fut_df["lower"].iloc[-1], fut_df["upper"].iloc[-1]
        direction = "up" if f_v > now_v else "down"
        st.caption(f"Linear trend over the last 120 days extrapolated 30 days out. "
                   f"Point forecast **{money(f_v)} USD** ({direction} from {money(now_v)}), "
                   f"95% interval **{money(lo)} – {money(hi)}**. "
                   "A linear model is a deliberately simple, transparent baseline — the widening "
                   "band shows forecast uncertainty growing with horizon.")
    else:
        st.info("Not enough history to forecast this chain.")

    st.markdown("---")
    colA, colB = st.columns([1, 1])

    # category mix
    with colA:
        st.subheader("TVL by category")
        cat = od.chain_category_breakdown(chain, protocols, exclude_cex=exclude_cex)
        cat_top = cat.head(7)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(cat_top["tvl"], labels=cat_top["category"], autopct="%1.0f%%",
               colors=PALETTE, startangle=90, wedgeprops={"edgecolor": "white"})
        st.pyplot(fig); plt.close(fig)
        if len(cat) >= 2:
            tot = cat["tvl"].sum()
            st.caption(f"**Takeaway.** {cat.iloc[0]['category']} leads on {chain} "
                       f"({cat.iloc[0]['tvl']/tot*100:.0f}% of on-chain TVL), ahead of "
                       f"{cat.iloc[1]['category']} ({cat.iloc[1]['tvl']/tot*100:.0f}%). "
                       "The category mix shows what the chain is actually used for.")

    # DEX volume trend
    with colB:
        st.subheader("DEX volume (last 90d)")
        if len(dexh):
            recent = dexh.tail(90)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar(recent["date"], recent["volume"], color=PALETTE[1], width=1.0)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
            ax.tick_params(axis="x", rotation=30)
            st.pyplot(fig); plt.close(fig)
            avg90 = recent["volume"].mean()
            latest_v = dexh["volume"].iloc[-1]
            vs = "above" if latest_v > avg90 else "below"
            st.caption(f"**Takeaway.** Latest daily DEX volume {money(latest_v)} USD, {vs} the 90-day "
                       f"average of {money(avg90)} USD. DEX volume is a direct read on real trading activity.")
        else:
            st.info("No DEX volume data reported for this chain.")

    # top protocols table
    st.subheader(f"Top protocols on {chain}")
    cp_raw = od.chain_protocols(chain, protocols, exclude_cex=exclude_cex, top=15)
    if len(cp_raw) >= 2:
        st.caption(f"**Takeaway.** {cp_raw.iloc[0]['name']} is the largest DeFi protocol on {chain} "
                   f"({money(cp_raw.iloc[0]['tvl_on_chain'])} USD, {cp_raw.iloc[0]['share_%']:.0f}% of the "
                   f"top-15 shown), followed by {cp_raw.iloc[1]['name']}. Concentration at the top mirrors "
                   "the chain-level picture — a few protocols carry most of the TVL.")
    cp = cp_raw.copy()
    cp["tvl_on_chain"] = cp["tvl_on_chain"].apply(usd)
    cp.columns = ["Protocol", "Category", "TVL on chain", "Share %"]
    st.dataframe(cp, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download this table (CSV)",
                       od.chain_protocols(chain, protocols, exclude_cex=exclude_cex, top=50).to_csv(index=False),
                       file_name=f"{chain}_protocols.csv", mime="text/csv")

    # ---- protocol age + last-30-day momentum ----
    st.subheader(f"Protocol age & recent momentum on {chain} (top 10)")
    name_to_slug = {p["name"]: p.get("slug") for p in protocols if p.get("slug")}
    top_slugs = [name_to_slug[n] for n in cp_raw["name"].head(10) if name_to_slug.get(n)]
    with st.spinner("Pulling per-protocol chain history…"):
        det = od.chain_protocol_detail(chain, tuple(top_slugs))
    if len(det):
        horizon_cols = [c for c in ["1d", "5d", "7d", "15d", "30d"] if c in det.columns]
        show = det[["name", "first_seen", "tvl_now"] + horizon_cols].copy()
        show["tvl_now"] = show["tvl_now"].apply(usd)
        show = show.rename(columns={"name": "Protocol", "first_seen": "First seen on chain",
                                    "tvl_now": "TVL now"})
        st.dataframe(
            show.style.format({c: "{:+.1f}%" for c in horizon_cols})
                .background_gradient(cmap="RdYlGn", subset=horizon_cols, vmin=-30, vmax=30),
            use_container_width=True, hide_index=True,
        )
        gain = det.loc[det["30d"].idxmax()]
        loss = det.loc[det["30d"].idxmin()]
        newest = det.loc[det["first_seen"].idxmax()]
        oldest = det.loc[det["first_seen"].idxmin()]
        st.caption(
            f"**Takeaway.** Columns show each protocol's TVL % change over 1/5/7/15/30 days — reading "
            f"left→right shows whether momentum is building or fading. Over 30 days, **{gain['name']}** grew "
            f"the most ({gain['30d']:+.1f}%) and **{loss['name']}** fell the most ({loss['30d']:+.1f}%). "
            f"Oldest here is **{oldest['name']}** (since {oldest['first_seen']}), newest is "
            f"**{newest['name']}** (since {newest['first_seen']}) — a young protocol already near the top "
            "signals fast-rising, less-battle-tested capital worth extra scrutiny."
        )
    else:
        st.info("Per-protocol chain history unavailable for this chain.")

# ======================================================================
# TAB 3 — Compare  (multi-horizon momentum)
# ======================================================================
with tab_compare:
    st.subheader("Multi-horizon performance comparison")
    picks = st.multiselect("Chains to compare", chains.head(20)["name"].tolist(),
                           default=chains.head(8)["name"].tolist(), key="cmp_chains")
    horizons = (1, 5, 7, 15, 30, 60)

    if len(picks) >= 1:
        wide = L_multi_hist(tuple(picks), 90)
        if len(wide) > 60:
            pc = od.period_changes(wide, horizons)
            pc = pc.reindex(picks)  # keep user's order
            disp = pc.copy()
            disp.columns = [f"{h}" for h in ["1d", "5d", "7d", "15d", "30d", "60d"]]

            st.markdown("#### % change in TVL over each horizon")
            st.dataframe(
                disp.style.format("{:+.1f}%")
                    .background_gradient(cmap="RdYlGn", vmin=-25, vmax=25),
                use_container_width=True,
            )
            st.caption("Green = TVL grew over that lookback, red = shrank. Reading left→right for one "
                       "chain shows whether momentum is building (short-term > long-term) or fading.")

            # ---- auto insight ----
            latest = pc["30d"].dropna()
            if len(latest):
                best = latest.idxmax(); worst = latest.idxmin()
                # momentum: 7d annualized-ish vs 30d
                accel = [c for c in pc.index
                         if pd.notna(pc.loc[c, "7d"]) and pd.notna(pc.loc[c, "30d"])
                         and pc.loc[c, "7d"] > 0 and pc.loc[c, "30d"] > 0
                         and pc.loc[c, "7d"] * (30 / 7) > pc.loc[c, "30d"]]
                msg = (f"**Summary.** Over the last 30 days, **{best}** leads at "
                       f"{pc.loc[best,'30d']:+.1f}% while **{worst}** lags at {pc.loc[worst,'30d']:+.1f}%. ")
                if accel:
                    msg += (f"Momentum is **accelerating** for {', '.join(accel[:4])} "
                            "(recent 7-day pace outruns the 30-day trend). ")
                pos = (latest > 0).sum()
                msg += f"{pos}/{len(latest)} selected chains grew over 30 days — " + \
                       ("broadly risk-on." if pos > len(latest) / 2 else "a defensive tape.")
                st.info(msg)

            # ---- bar of a chosen horizon ----
            h_pick = st.select_slider("Highlight one horizon as a bar chart",
                                      options=["1d", "5d", "7d", "15d", "30d", "60d"], value="30d")
            col = disp[h_pick].dropna().sort_values()
            fig, ax = plt.subplots(figsize=(11, 3.8))
            ax.barh(col.index, col.values, color=[GREEN if v >= 0 else RED for v in col.values])
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlabel(f"TVL change over {h_pick} (%)")
            st.pyplot(fig); plt.close(fig)
            gainers = (col > 0).sum()
            st.caption(f"**Takeaway.** Over {h_pick}, **{col.index[-1]}** performed best "
                       f"({col.iloc[-1]:+.1f}%) and **{col.index[0]}** worst ({col.iloc[0]:+.1f}%); "
                       f"{gainers}/{len(col)} of the selected chains were positive.")
        else:
            st.info("Not enough history for the selected chains.")
    else:
        st.info("Select at least one chain.")

# ======================================================================
# TAB 4 — Quant Analytics  (cross-chain statistics)
# ======================================================================
with tab_quant:
    st.subheader("Cross-chain quantitative analysis")
    universe = chains.head(10)["name"].tolist()
    picks = st.multiselect("Chains to compare", chains.head(20)["name"].tolist(),
                           default=universe[:6])
    lookback = st.select_slider("Lookback window", options=[90, 180, 365, 730], value=365)

    if len(picks) >= 2:
        wide = L_multi_hist(tuple(picks), lookback)
        if len(wide) > 30:
            ret = wide.pct_change().dropna()

            # ---- Risk metrics table ----
            st.markdown("#### Risk & return profile")
            risk = od.risk_metrics(wide)
            risk_disp = risk.copy()
            risk_disp.columns = ["Ann. volatility %", "Max drawdown %", "Current drawdown %", "30d return %"]
            st.dataframe(risk_disp.style.format("{:.1f}")
                         .background_gradient(cmap="RdYlGn_r", subset=["Ann. volatility %", "Max drawdown %"]),
                         use_container_width=True)
            st.caption("Annualized volatility of daily TVL returns, worst peak-to-trough drawdown, "
                       "and how far each chain sits below its all-time peak right now.")

            colL, colR = st.columns(2)

            # ---- Correlation heatmap ----
            with colL:
                st.markdown("#### TVL correlation matrix")
                corr = ret.corr()
                fig, ax = plt.subplots(figsize=(5.5, 4.8))
                im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
                for i in range(len(corr)):
                    for j in range(len(corr)):
                        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black", fontsize=8)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ax.grid(False)
                st.pyplot(fig); plt.close(fig)
                st.caption("Do chains move together? High correlation ⇒ shared market beta, "
                           "little diversification across the space.")

            # ---- Rebased trajectories ----
            with colR:
                st.markdown("#### Relative performance (rebased to 100)")
                reb = od.rebased_trajectories(wide)
                fig, ax = plt.subplots(figsize=(5.5, 4.8))
                for i, c in enumerate(reb.columns):
                    ax.plot(reb.index, reb[c], label=c, color=PALETTE[i % len(PALETTE)], linewidth=1.5)
                ax.axhline(100, color="#999", linestyle="--", linewidth=0.8)
                ax.legend(fontsize=8, loc="upper left")
                ax.set_ylabel("Index (start = 100)")
                st.pyplot(fig); plt.close(fig)
                st.caption("Each chain's TVL rebased to 100 at window start — compares growth "
                           "shape regardless of absolute size.")

            st.markdown("---")

            # ---- Market concentration over time ----
            st.markdown("#### Market concentration over time (HHI)")
            hhi_ts = L_hhi(lookback)
            if len(hhi_ts):
                fig, ax = plt.subplots(figsize=(12, 3.4))
                ax.plot(hhi_ts["date"], hhi_ts["hhi"], color="#7c3aed", linewidth=1.8)
                ax.fill_between(hhi_ts["date"], hhi_ts["hhi"], alpha=0.10, color="#7c3aed")
                ax.axhline(0.25, color=RED, linestyle="--", linewidth=0.8, label="0.25 = 'concentrated' threshold")
                ax.legend(fontsize=8)
                ax.set_ylabel("HHI")
                st.pyplot(fig); plt.close(fig)
                trend = "consolidating ↑" if hhi_ts["hhi"].iloc[-1] > hhi_ts["hhi"].iloc[0] else "fragmenting ↓"
                st.caption(f"Herfindahl index across the top chains. Trend over window: **{trend}**. "
                           "Rising HHI ⇒ TVL concentrating into fewer chains.")

            # ---- Capital rotation ----
            st.markdown("#### Capital rotation — who's gaining / losing share")
            rot = L_rotation(min(lookback, 90))
            if len(rot):
                rot_show = pd.concat([rot.head(6), rot.tail(6)]).drop_duplicates("chain")
                fig, ax = plt.subplots(figsize=(12, 3.8))
                colors = [GREEN if v >= 0 else RED for v in rot_show["delta_pp"]]
                ax.bar(rot_show["chain"], rot_show["delta_pp"], color=colors)
                ax.axhline(0, color="#333", linewidth=0.8)
                ax.set_ylabel("Share change (pp)")
                ax.tick_params(axis="x", rotation=35)
                st.pyplot(fig); plt.close(fig)
                st.caption(f"Change in each chain's share of combined TVL over the last "
                           f"{min(lookback, 90)} days. Green = capital flowing in, red = flowing out.")

            # ---- cross-chart quant verdict ----
            corr = ret.corr()
            n = len(corr)
            # mean of off-diagonal correlations (exclude the n ones on the diagonal)
            avg_corr = (corr.values.sum() - n) / (n * n - n) if n > 1 else 0.0
            risk = od.risk_metrics(wide)
            riskiest = risk["ann_vol_%"].idxmax()
            deepest_dd = risk["max_drawdown_%"].idxmin()
            hhi_trend = ("consolidating" if len(hhi_ts) and hhi_ts["hhi"].iloc[-1] > hhi_ts["hhi"].iloc[0]
                         else "fragmenting")
            div_word = ("move almost as one — little diversification benefit across chains"
                        if avg_corr >= 0.7 else
                        "are moderately correlated — some diversification available" if avg_corr >= 0.4 else
                        "move fairly independently — real diversification across chains")
            rot_line = ""
            if len(rot):
                rot_line = (f" Capital is rotating toward **{rot.iloc[0]['chain']}** "
                            f"({rot.iloc[0]['delta_pp']:+.1f}pp) and out of **{rot.iloc[-1]['chain']}** "
                            f"({rot.iloc[-1]['delta_pp']:+.1f}pp).")
            st.success(
                f"### 📈 Quant read\n\n"
                f"Selected chains {div_word} (avg correlation **{avg_corr:.2f}**). "
                f"**{riskiest}** is the most volatile (ann. vol {risk.loc[riskiest,'ann_vol_%']:.0f}%) and "
                f"**{deepest_dd}** has the deepest drawdown ({risk.loc[deepest_dd,'max_drawdown_%']:.0f}%). "
                f"The market is **{hhi_trend}** over this window.{rot_line} "
                "*High correlation + concentration means chain-picking adds little — the whole space rises "
                "and falls together, so risk is mostly directional (crypto beta), not chain-specific.*"
            )
        else:
            st.info("Not enough overlapping history for the selected chains/window.")
    else:
        st.info("Select at least 2 chains to compare.")

# ======================================================================
# TAB 4 — Protocol Deep Dive
# ======================================================================
with tab_protocol:
    st.subheader("Single-protocol deep dive")
    plist = od.protocol_list(min_tvl=5e8, protocols=L_protocols())
    plist = plist[~plist["category"].isin(["CEX", "Chain"])].reset_index(drop=True)
    labels = [f"{r['name']}  ·  {r['category']}  ·  {usd(r['tvl'])}" for _, r in plist.iterrows()]
    default_ix = next((i for i, r in plist.iterrows() if r["name"] == "Aave V3"), 0)
    sel = st.selectbox("Select a protocol", range(len(plist)),
                       format_func=lambda i: labels[i], index=int(default_ix))
    slug = plist.iloc[sel]["slug"]
    pname = plist.iloc[sel]["name"]

    with st.spinner(f"Loading {pname} data…"):
        tvl_h = L_prot_tvl(slug)
        fees_h = L_prot_fees(slug, "dailyFees")
        rev_h = L_prot_fees(slug, "dailyRevenue")

    if not len(tvl_h):
        st.warning(f"No TVL history reported for {pname} on DefiLlama.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    if len(tvl_h):
        m1.metric(f"{pname} TVL", usd(tvl_h["tvl"].iloc[-1]))
        if len(tvl_h) > 31:
            chg = (tvl_h["tvl"].iloc[-1] / tvl_h["tvl"].iloc[-31] - 1) * 100
            m2.metric("30d TVL change", f"{chg:+.1f}%")
    if len(fees_h):
        m3.metric("Avg daily fees (30d)", usd(fees_h["value"].tail(30).mean()))
    if len(rev_h):
        m4.metric("Avg daily revenue (30d)", usd(rev_h["value"].tail(30).mean()))

    st.markdown("---")

    # TVL history with anomaly detection
    if len(tvl_h) > window:
        st.subheader(f"{pname} — TVL history with anomaly detection")
        flg = od.detect_anomalies(tvl_h, "tvl", window=window, k=k)
        an = flg[flg["anomaly"]]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(flg["date"], flg["tvl"], color=BLUE, linewidth=1.5, label="TVL")
        ax.fill_between(flg["date"], flg["tvl"], alpha=0.10, color=BLUE)
        ax.scatter(an["date"], an["tvl"], color=RED, s=24, zorder=5, label=f"Anomaly (|z|>{k})")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
        ax.legend(loc="upper left", fontsize=8)
        st.pyplot(fig); plt.close(fig)
        peak = tvl_h["tvl"].max()
        cur = tvl_h["tvl"].iloc[-1]
        dd = (cur / peak - 1) * 100
        cap = f"**Takeaway.** {pname} TVL is {money(cur)} USD, {abs(dd):.0f}% " \
              f"{'below' if dd < 0 else 'at'} its all-time peak of {money(peak)} USD; " \
              f"{len(an)} abnormal days flagged."
        if len(an):
            cap += f" Most recent: {an.sort_values('date').iloc[-1]['date'].date()}."
        st.caption(cap)

    # Fees vs revenue
    if len(fees_h):
        st.subheader(f"{pname} — daily fees vs. protocol revenue")
        fig, ax = plt.subplots(figsize=(12, 3.8))
        ff, rr = fees_h.tail(180), rev_h.tail(180)
        ax.plot(ff["date"], ff["value"], color=PALETTE[4], linewidth=1.3, label="Fees (to users/LPs)")
        if len(rr):
            ax.plot(rr["date"], rr["value"], color=PALETTE[2], linewidth=1.3, label="Revenue (to protocol)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
        ax.legend(loc="upper left", fontsize=8)
        st.pyplot(fig); plt.close(fig)
        # ---- deeper business-quality analysis ----
        if len(tvl_h) and len(rev_h) and len(fees_h):
            fee30 = fees_h["value"].tail(30).mean()
            fee_prev30 = fees_h["value"].tail(60).head(30).mean() if len(fees_h) > 60 else float("nan")
            rev30 = rev_h["value"].tail(30).mean()
            ann_rev = rev30 * 365
            tvl_now = tvl_h["tvl"].iloc[-1]
            capture = (rev30 / fee30 * 100) if fee30 else float("nan")   # % of fees kept by protocol
            rev_tvl = ann_rev / tvl_now * 100 if tvl_now else float("nan")
            fee_trend = (fee30 / fee_prev30 - 1) * 100 if fee_prev30 and fee_prev30 == fee_prev30 else float("nan")

            k1, k2, k3 = st.columns(3)
            k1.metric("Revenue capture", f"{capture:.0f}%", help="Share of fees the protocol keeps (vs paid to LPs)")
            k2.metric("Revenue / TVL (ann.)", f"{rev_tvl:.1f}%", help="How much income each dollar of TVL generates")
            if fee_trend == fee_trend:
                k3.metric("Fee trend (30d vs prior 30d)", f"{fee_trend:+.0f}%")

            # synthesis verdict
            parts = []
            if rev_tvl >= 3: parts.append("**high capital efficiency**")
            elif rev_tvl >= 1: parts.append("moderate capital efficiency")
            else: parts.append("**low capital efficiency** (lots of TVL, little revenue)")
            if capture == capture:
                parts.append(f"the protocol keeps **{capture:.0f}%** of fees as revenue"
                             + (" — a strong take rate" if capture >= 30 else " — most value flows to LPs"))
            if fee_trend == fee_trend:
                parts.append("fees are **growing**" if fee_trend > 5 else
                             "fees are **shrinking**" if fee_trend < -5 else "fees are roughly flat")
            st.success(f"**Business read.** {pname} shows " + ", ".join(parts) + ". "
                       "Revenue/TVL is the clearest quality signal — a protocol earning well on its locked "
                       "capital is more sustainable than one paying to attract idle TVL.")
            st.caption(f"Annualized revenue ≈ {money(ann_rev)} on {money(tvl_now)} USD TVL. "
                       "Fees = what users pay; revenue = the slice the protocol keeps; the gap goes to LPs.")
    else:
        st.info("This protocol does not report fee/revenue data on DefiLlama.")

# ======================================================================
# TAB 6 — Whale Monitor
# ======================================================================
with tab_whale:
    st.subheader("🐋 Whale-sized flow monitor")
    st.caption("Tracks the largest single-day **net capital movements** — a proxy for big-money "
               "(whale/institutional) activity. Note: this is aggregate net flow from TVL & stablecoin "
               "supply data, not per-wallet tracking (which needs an on-chain indexer like Etherscan/Nansen).")
    wdays = st.select_slider("Look-back window (days)", options=[14, 30, 60, 90], value=30)

    # ---- cross-signal whale verdict (same window; stablecoin signal prioritized) ----
    chain_net, stable_net = L_net_flows(wdays)
    if chain_net > 0 and stable_net > 0:
        vlabel = "🟢 Whales accumulating"
        vexp = ("Both chain TVL and stablecoin supply grew over the window — real cash is entering DeFi "
                "and staying. Historically a risk-on backdrop.")
    elif chain_net < 0 and stable_net < 0:
        vlabel = "🔴 Whales de-risking"
        vexp = ("Both chain TVL and stablecoin supply shrank — big money is stepping back and redeeming to "
                "cash. A clear risk-off signal.")
    elif stable_net < 0 <= chain_net:
        vlabel = "🟠 Cash leaving (TVL propped by price)"
        vexp = ("Stablecoins are being redeemed (real cash exiting) even though chain TVL rose — the TVL gain "
                "is likely price appreciation of locked assets, not fresh deposits. Leaning risk-off: the "
                "stablecoin signal is the cleaner cash tell, so treat the TVL rise with caution.")
    else:  # stable_net >= 0, chain_net < 0
        vlabel = "🟡 Cash entering, TVL soft"
        vexp = ("Fresh stablecoins are being minted (cash entering crypto) while chain TVL slipped — new money "
                "is arriving but sitting on the sidelines rather than deploying yet. Tentatively constructive.")
    st.success(
        f"### {vlabel}\n\n{vexp} "
        f"*(True net over the same {wdays}-day window: chain TVL "
        f"{('+' if chain_net>=0 else '')}{money(chain_net)} USD, stablecoin supply "
        f"{('+' if stable_net>=0 else '')}{money(stable_net)} USD. Stablecoin supply is pegged, so it has no "
        "price noise — it's the more reliable read on real cash in/out.)*"
    )

    # ---- chain-level whale flows ----
    st.markdown("#### Largest chain net flows")
    cf = L_whale_chain(wdays)
    if len(cf):
        inflow = cf[cf["flow_usd"] > 0].head(10)
        outflow = cf[cf["flow_usd"] < 0].head(10)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🟢 Biggest inflows**")
            t1 = inflow.copy()
            t1["flow_usd"] = t1["flow_usd"].apply(usd)
            t1 = t1[["date", "chain", "flow_usd", "pct"]]
            t1.columns = ["Date", "Chain", "Inflow", "%"]
            st.dataframe(t1.style.format({"%": "{:+.1f}%"}), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**🔴 Biggest outflows**")
            t2 = outflow.copy()
            t2["flow_usd"] = t2["flow_usd"].apply(usd)
            t2 = t2[["date", "chain", "flow_usd", "pct"]]
            t2.columns = ["Date", "Chain", "Outflow", "%"]
            st.dataframe(t2.style.format({"%": "{:+.1f}%"}), use_container_width=True, hide_index=True)

        biggest = cf.iloc[0]
        net = cf["flow_usd"].sum()
        st.caption(
            f"**Takeaway.** The single largest move was **{money(abs(biggest['flow_usd']))} USD "
            f"{biggest['direction']}** on {biggest['chain']} ({biggest['date']}). Across the top flows in "
            f"this window the net is **{money(net)} USD** — "
            + ("net inflows dominate (big money entering DeFi)." if net > 0
               else "net outflows dominate (big money leaving DeFi).")
        )
    else:
        st.info("No flow data available.")

    st.markdown("---")

    # ---- stablecoin whale flows (mints/burns) ----
    st.markdown("#### Largest stablecoin mints & burns (institutional flow)")
    sf = L_whale_stable(max(wdays, 60))
    if len(sf):
        fig, ax = plt.subplots(figsize=(12, 3.8))
        colors = [GREEN if v > 0 else RED for v in sf["flow_usd"]]
        ax.bar(sf["date"].astype(str), sf["flow_usd"], color=colors)
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig); plt.close(fig)

        mints = sf[sf["flow_usd"] > 0]["flow_usd"].sum()
        burns = sf[sf["flow_usd"] < 0]["flow_usd"].sum()
        st.caption(
            f"**Takeaway.** Among the biggest stablecoin-supply days, total mints "
            f"**{money(mints)} USD** vs burns **{money(burns)} USD**. Because stablecoins are pegged, "
            "these are clean money-in / money-out signals: sustained net minting = fresh capital entering "
            "crypto, sustained burning = capital exiting. Large single-day mints often precede risk-on moves."
        )
        show = sf.copy()
        show["flow_usd"] = show["flow_usd"].apply(usd)
        show.columns = ["Date", "Flow", "% of supply", "Direction"]
        st.dataframe(show.style.format({"% of supply": "{:+.2f}%"}),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No large stablecoin flow events in this window.")

# ======================================================================
# TAB 7 — Stablecoin Monitor
# ======================================================================
with tab_stable:
    hist = L_stable_hist()

    # ---- cross-signal stablecoin verdict ----
    _sup = hist["total"].iloc[-1]
    _chg30 = (hist["total"].iloc[-1] / hist["total"].iloc[-31] - 1) * 100 if len(hist) > 31 else 0
    _stb = L_stables()
    _dom = _stb.iloc[0]
    _dp = od.depeg_events(L_depeg_prices(), threshold=0.005)
    supply_word = ("expanding (capital entering crypto)" if _chg30 > 1 else
                   "contracting (net redemptions)" if _chg30 < -1 else "roughly flat")
    peg_word = (f"all pegs held (±0.5%)" if not len(_dp)
                else f"{len(_dp)} de-peg day(s), worst {_dp.iloc[0]['symbol']} {_dp.iloc[0]['deviation_%']:+.2f}%")
    health = "🟢 Healthy" if (not len(_dp) and _chg30 >= -1) else \
             "🔴 Stressed" if (len(_dp) and _chg30 < -1) else "🟡 Watch"
    st.success(
        f"### {health} stablecoin market\n\n"
        f"Total supply **{money(_sup)} USD**, **{_chg30:+.1f}%** over 30 days ({supply_word}); {peg_word}. "
        f"Concentration is high — **{_dom['symbol']}** alone is {_dom['share_%']:.0f}% of all stablecoins, "
        "so its peg and reserves are a systemic dependency for the whole market. "
        "*Supply is a clean money-in/out gauge (pegged assets have no price noise); peg deviations are the "
        "early-warning tripwire.*"
    )

    st.subheader("Total stablecoin supply — anomaly detection")
    flagged = od.detect_anomalies(hist, "total", window=window, k=k)
    an = flagged[flagged["anomaly"]]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(flagged["date"], flagged["total"], color=BLUE, linewidth=1.6, label="Total supply")
    ax.fill_between(flagged["date"], flagged["total"], alpha=0.10, color=BLUE)
    ax.scatter(an["date"], an["total"], color=RED, s=26, zorder=5, label=f"Anomaly (|z| > {k})")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(usd))
    ax.legend(loc="upper left")
    st.pyplot(fig); plt.close(fig)
    _cur = hist["total"].iloc[-1]
    _chg30 = (hist["total"].iloc[-1] / hist["total"].iloc[-31] - 1) * 100 if len(hist) > 31 else 0
    _flow = "capital entering crypto" if _chg30 >= 0 else "net redemptions"
    st.caption(f"**Takeaway.** Total stablecoin supply is {money(_cur)} USD, **{_chg30:+.1f}% over 30 days** "
               f"({_flow}). Unlike chain TVL, stablecoins are pegged to 1 USD — so supply changes are a **clean "
               "flow signal with no price effect**: a rise is genuine minting (money in), a fall is redemptions "
               "(money out). Flagged days mark the sharpest mint/burn events, often around market stress.")

    st.write(f"**{len(an)} anomalous days** flagged (window={window}d, k={k}). Most recent 12:")
    tbl = od.anomaly_table(flagged, "total").head(12).copy()
    tbl["total"] = tbl["total"].apply(usd)
    tbl["change"] = tbl["change"].apply(usd)
    tbl.columns = ["Date", "Supply", "Day change", "z-score", "Direction"]
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---- De-peg monitoring (risk view) ----
    st.subheader("🚨 De-peg monitor — deviation from $1.00")
    dp1, dp2 = st.columns([2, 1])
    with dp2:
        peg_thresh = st.slider("De-peg threshold (%)", 0.1, 2.0, 0.5, step=0.1) / 100
    prices = od.stablecoin_prices(("USDT", "USDC", "DAI", "USDe"), days=365)

    fig, ax = plt.subplots(figsize=(12, 4))
    for i, sym in enumerate(prices.columns):
        ax.plot(prices.index, prices[sym], label=sym, color=PALETTE[i], linewidth=1.2)
    ax.axhline(1.0, color="#333", linewidth=0.8)
    ax.axhline(1 + peg_thresh, color=RED, linestyle="--", linewidth=0.7)
    ax.axhline(1 - peg_thresh, color=RED, linestyle="--", linewidth=0.7,
               label=f"±{peg_thresh*100:.1f}% band")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="lower left", fontsize=8, ncol=5)
    st.pyplot(fig); plt.close(fig)

    events = od.depeg_events(prices, threshold=peg_thresh)
    if len(events):
        st.write(f"**{len(events)} de-peg days** where a coin left the ±{peg_thresh*100:.1f}% band "
                 "(largest deviations first):")
        ev = events.head(15).copy()
        ev.columns = ["Date", "Coin", "Price", "Deviation %", "Direction"]
        st.dataframe(ev, use_container_width=True, hide_index=True)
        st.caption("From a risk desk's view, these are the moments that matter: a stablecoin trading "
                   "away from $1 signals redemption stress, thin liquidity, or loss of confidence. "
                   "USDe (a synthetic dollar) shows the widest deviations here — expected, given its "
                   "delta-hedged design differs from fiat-backed USDT/USDC.")
    else:
        st.success(f"No stablecoin left the ±{peg_thresh*100:.1f}% band in the last year.")

st.caption("Data: DefiLlama public API (no key required). Cached 1h. Dashboard by Qingqing Jian.")
