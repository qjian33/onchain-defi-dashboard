"""
DeFi Chain TVL & Stablecoin Market Analysis
============================================
Data source: DefiLlama public API (https://defillama.com/docs/api) — no API key required.

What this script answers:
  1. Which blockchains hold the most DeFi Total Value Locked (TVL), and how concentrated is the market?
  2. How has the stablecoin market grown, and who dominates it?

Author: Qingqing Jian
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
LLAMA = "https://api.llama.fi"
STABLES = "https://stablecoins.llama.fi"
CHARTS_DIR = "charts"
TOP_N = 12

# Clean, readable chart style
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


def get(url):
    """GET JSON with a browser-like UA and a timeout."""
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (research script)"}, timeout=30)
    r.raise_for_status()
    return r.json()


def fmt_billions(x, _=None):
    return f"${x/1e9:.0f}B"


# ----------------------------------------------------------------------
# Analysis 1 — DeFi TVL by chain
# ----------------------------------------------------------------------
def analyze_chain_tvl():
    print("Fetching chain TVL ...")
    data = get(f"{LLAMA}/v2/chains")
    df = pd.DataFrame(data)[["name", "tvl"]].dropna()
    df = df.sort_values("tvl", ascending=False).reset_index(drop=True)

    total = df["tvl"].sum()
    top = df.head(TOP_N).copy()
    top["share_%"] = (top["tvl"] / total * 100).round(2)

    # Herfindahl index — market concentration (0=spread out, 1=monopoly)
    hhi = ((df["tvl"] / total) ** 2).sum()
    top3_share = top["share_%"].head(3).sum()

    print(f"  Total DeFi TVL across {len(df)} chains: ${total/1e9:.1f}B")
    print(f"  Top-3 chains hold {top3_share:.1f}% of all TVL")
    print(f"  Concentration (HHI): {hhi:.3f}")

    # Chart: horizontal bar of top chains by TVL
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["name"][::-1], top["tvl"][::-1], color="#3b82f6")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_billions))
    ax.set_title(f"Top {TOP_N} Blockchains by DeFi TVL\nTotal market ${total/1e9:.0f}B  |  Top-3 = {top3_share:.0f}%",
                 fontweight="bold", loc="left")
    ax.set_xlabel("Total Value Locked (USD)")
    fig.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/1_chain_tvl.png")
    plt.close(fig)
    print(f"  -> saved {CHARTS_DIR}/1_chain_tvl.png")

    return top[["name", "tvl", "share_%"]]


# ----------------------------------------------------------------------
# Analysis 2 — Stablecoin market: size, growth, dominance
# ----------------------------------------------------------------------
def analyze_stablecoins():
    print("Fetching stablecoin data ...")
    # Current circulating supply by stablecoin
    assets = get(f"{STABLES}/stablecoins?includePrices=false")["peggedAssets"]
    rows = []
    for a in assets:
        circ = a.get("circulating", {}) or {}
        usd = circ.get("peggedUSD", 0) or 0
        if usd > 0:
            rows.append({"symbol": a["symbol"], "name": a["name"], "circulating": usd})
    df = pd.DataFrame(rows).sort_values("circulating", ascending=False).reset_index(drop=True)

    total = df["circulating"].sum()
    df["share_%"] = (df["circulating"] / total * 100).round(2)
    top = df.head(TOP_N)

    print(f"  Total stablecoin supply: ${total/1e9:.1f}B across {len(df)} coins")
    print(f"  #1 {top.iloc[0]['symbol']} = {top.iloc[0]['share_%']:.1f}% dominance")

    # Chart A: pie of stablecoin market share (top 6 + others)
    top6 = df.head(6).copy()
    others = pd.DataFrame([{"symbol": "Others", "circulating": df["circulating"][6:].sum()}])
    pie_df = pd.concat([top6[["symbol", "circulating"]], others], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(pie_df["circulating"], labels=pie_df["symbol"], autopct="%1.1f%%",
           colors=plt.cm.Blues_r(range(0, 256, 36)), startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax.set_title(f"Stablecoin Market Share\nTotal ${total/1e9:.0f}B", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/2_stablecoin_share.png")
    plt.close(fig)
    print(f"  -> saved {CHARTS_DIR}/2_stablecoin_share.png")

    # Chart B: total stablecoin supply over time (historical)
    hist = get(f"{STABLES}/stablecoincharts/all")
    h = pd.DataFrame(hist)
    h["date"] = pd.to_datetime(h["date"].astype(int), unit="s")
    h["total"] = h["totalCirculatingUSD"].apply(lambda d: d.get("peggedUSD", 0) if isinstance(d, dict) else 0)
    h = h[h["total"] > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(h["date"], h["total"], color="#2563eb", linewidth=2)
    ax.fill_between(h["date"], h["total"], alpha=0.12, color="#2563eb")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_billions))
    ax.set_title("Total Stablecoin Supply Over Time", fontweight="bold", loc="left")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(f"{CHARTS_DIR}/3_stablecoin_growth.png")
    plt.close(fig)
    print(f"  -> saved {CHARTS_DIR}/3_stablecoin_growth.png")

    return top[["symbol", "circulating", "share_%"]]


def main():
    print("=" * 60)
    print("DeFi Chain TVL & Stablecoin Market Analysis")
    print("=" * 60)
    chains = analyze_chain_tvl()
    print()
    stables = analyze_stablecoins()

    # Save data tables as CSV (part of the portfolio deliverable)
    chains.to_csv("chain_tvl.csv", index=False)
    stables.to_csv("stablecoins.csv", index=False)
    print("\nDone. Charts in ./charts/, data tables as CSV.")


if __name__ == "__main__":
    main()
