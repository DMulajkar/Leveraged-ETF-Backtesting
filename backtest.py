import matplotlib
matplotlib.use("Agg")
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime

TICKERS = ["TQQQ", "SOXL", "SSO", "UPRO", "TECL"]
UNLEVERAGED_MAP = {
    "TQQQ": {"underlying_etf": "QQQ", "underlying_name": "NASDAQ-100", "leverage": 3},
    "SOXL": {"underlying_etf": "SOXX", "underlying_name": "Semiconductors", "leverage": 3},
    "SSO":  {"underlying_etf": "SPY", "underlying_name": "S&P 500", "leverage": 2},
    "UPRO": {"underlying_etf": "SPY", "underlying_name": "S&P 500", "leverage": 3},
    "TECL": {"underlying_etf": "XLK", "underlying_name": "Technology", "leverage": 3},
}

EXPENSE_RATIOS = {
    "TQQQ": 0.88,
    "SOXL": 0.75,
    "SSO": 0.89,
    "UPRO": 0.91,
    "TECL": 0.88,
}

START_DATE = "2006-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 10000
BENCHMARK = "SPY"

# ============================
# DATA DOWNLOAD
# ============================
print(f"Downloading data from {START_DATE} to {END_DATE}...")
all_tickers = TICKERS + list(set(v["underlying_etf"] for v in UNLEVERAGED_MAP.values() if v["underlying_etf"])) + [BENCHMARK]
all_tickers = list(set(all_tickers))
raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]

# ============================
# INDIVIDUAL ETF STATS
# ============================
all_stats = []
individual_curves = {}
for ticker in TICKERS:
    if ticker not in raw.columns:
        print(f"  {ticker}: no data found, skipping.")
        continue
    valid = raw[ticker].dropna()
    if valid.empty:
        continue
    start_idx, end_idx = valid.index[0], valid.index[-1]
    g = (valid / valid.iloc[0]) * INITIAL_CAPITAL
    individual_curves[ticker] = g

    daily_rets = g.pct_change().dropna()
    years = (end_idx - start_idx).days / 365.25
    final_val = g.iloc[-1]
    total_ret = (final_val / INITIAL_CAPITAL - 1) * 100
    cagr = ((final_val / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    vol = daily_rets.std() * np.sqrt(252) * 100
    rf_daily = 0.04 / 252
    sharpe = ((daily_rets.mean() - rf_daily) * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() != 0 else 0
    downside = daily_rets[daily_rets < 0]
    ds_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = ((daily_rets.mean() - rf_daily) * 252) / ds_std if ds_std != 0 else 0
    rm = g.cummax()
    dd = (g - rm) / rm
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    var_5 = daily_rets.quantile(0.05) * 100

    monthly = g.resample("ME").last().pct_change().dropna()
    yearly = g.resample("YE").last().pct_change().dropna()
    worst_month = monthly.min() * 100 if len(monthly) > 0 else None
    worst_month_date = monthly.idxmin().strftime("%b %Y") if len(monthly) > 0 else "N/A"
    worst_year = yearly.min() * 100 if len(yearly) > 0 else None
    worst_year_date = yearly.idxmin().strftime("%Y") if len(yearly) > 0 else "N/A"
    best_year = yearly.max() * 100 if len(yearly) > 0 else None
    best_year_date = yearly.idxmax().strftime("%Y") if len(yearly) > 0 else "N/A"

    dd_end = dd.idxmin()
    post_dd = g.loc[dd_end:]
    rec_mask = post_dd >= rm.loc[dd_end]
    rec_days = (rec_mask.idxmax() - dd_end).days if rec_mask.any() else None

    # Unleveraged counterpart
    und_ticker = UNLEVERAGED_MAP[ticker]["underlying_etf"]
    und_cagr_val = None
    und_final_val = None
    actual_multiplier = None
    if und_ticker and und_ticker in raw.columns:
        und_prices = raw[und_ticker].dropna()
        common = und_prices.index.intersection(valid.index)
        if len(common) > 0:
            und_aligned = und_prices.loc[common]
            und_years = (common[-1] - common[0]).days / 365.25
            und_cagr_val = ((und_aligned.iloc[-1] / und_aligned.iloc[0]) ** (1 / und_years) - 1) * 100
            und_final_val = und_aligned.iloc[-1] / und_aligned.iloc[0]
            actual_multiplier = cagr / und_cagr_val if und_cagr_val != 0 else 0

    all_stats.append({
        "Ticker": ticker,
        "Start": start_idx.strftime("%Y-%m-%d"),
        "End": end_idx.strftime("%Y-%m-%d"),
        "Final Value": f"${final_val:,.0f}",
        "Total Return": f"{total_ret:+.1f}%",
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max DD": max_dd,
        "Var 5%": var_5,
        "Worst Mo": worst_month,
        "Worst Mo Date": worst_month_date,
        "Worst Yr": worst_year,
        "Worst Yr Date": worst_year_date,
        "Best Yr": best_year,
        "Best Yr Date": best_year_date,
        "Recovery (days)": rec_days,
        "Unleveraged ETF": und_ticker or "N/A",
        "Underlying CAGR": und_cagr_val,
        "Actual Multiplier": actual_multiplier,
        "ER": EXPENSE_RATIOS[ticker],
})

# ============================
# SPY BENCHMARK STATS
# ============================
spy_lp = raw[BENCHMARK].dropna()
spy_g = (spy_lp / spy_lp.iloc[0]) * INITIAL_CAPITAL
spy_dr = spy_g.pct_change().dropna()
spy_yrs = (spy_lp.index[-1] - spy_lp.index[0]).days / 365.25
spy_cagr = ((spy_g.iloc[-1] / INITIAL_CAPITAL) ** (1 / spy_yrs) - 1) * 100
spy_total_ret = (spy_g.iloc[-1] / INITIAL_CAPITAL - 1) * 100
spy_vol = spy_dr.std() * np.sqrt(252) * 100
spy_sharpe = ((spy_dr.mean() - rf_daily) * 252) / (spy_dr.std() * np.sqrt(252))
spy_ds = spy_dr[spy_dr < 0].std() * np.sqrt(252)
spy_sortino = ((spy_dr.mean() - rf_daily) * 252) / spy_ds if spy_ds != 0 else 0
spy_rm = spy_g.cummax()
spy_dd = (spy_g - spy_rm) / spy_rm
spy_maxdd = spy_dd.min() * 100
spy_calmar = spy_cagr / abs(spy_maxdd) if spy_maxdd != 0 else 0
spy_var5 = spy_dr.quantile(0.05) * 100
spy_wm = spy_g.resample("ME").last().pct_change().min() * 100
spy_wm_date = spy_g.resample("ME").last().pct_change().idxmin().strftime("%b %Y")
spy_wy = spy_g.resample("YE").last().pct_change().min() * 100
spy_wy_date = spy_g.resample("YE").last().pct_change().idxmin().strftime("%Y")
spy_by = spy_g.resample("YE").last().pct_change().max() * 100
spy_by_date = spy_g.resample("YE").last().pct_change().idxmax().strftime("%Y")
spy_dd_end = spy_dd.idxmin()
spy_rec_mask = spy_g.loc[spy_dd_end:] >= spy_rm.loc[spy_dd_end]
spy_rec = (spy_rec_mask.idxmax() - spy_dd_end).days if spy_rec_mask.any() else None

all_stats.append({
    "Ticker": BENCHMARK,
    "Start": spy_lp.index[0].strftime("%Y-%m-%d"),
    "End": spy_lp.index[-1].strftime("%Y-%m-%d"),
    "Final Value": f"${spy_g.iloc[-1]:,.0f}",
    "Total Return": f"{spy_total_ret:+.1f}%",
    "CAGR": spy_cagr,
    "Volatility": spy_vol,
    "Sharpe": spy_sharpe,
    "Sortino": spy_sortino,
    "Calmar": spy_calmar,
    "Max DD": spy_maxdd,
    "Var 5%": spy_var5,
    "Worst Mo": spy_wm,
    "Worst Mo Date": spy_wm_date,
    "Worst Yr": spy_wy,
    "Worst Yr Date": spy_wy_date,
    "Best Yr": spy_by,
    "Best Yr Date": spy_by_date,
    "Recovery (days)": spy_rec,
    "Unleveraged ETF": "N/A",
    "Underlying CAGR": None,
    "Actual Multiplier": None,
    "ER": 0.09,
})

# ============================
# DIVERSIFIED PORTFOLIO
# ============================
common_start = max(raw[t].dropna().index[0] for t in TICKERS if t in raw.columns)
common_data = raw[TICKERS].loc[common_start:].dropna()
common_ret = common_data.pct_change().dropna()
port_ret = common_ret.mean(axis=1)
pg = (1 + port_ret).cumprod() * INITIAL_CAPITAL
individual_curves["DIVERSIFIED"] = pg

p_years = (pg.index[-1] - pg.index[0]).days / 365.25
p_final = pg.iloc[-1]
p_cagr = ((p_final / INITIAL_CAPITAL) ** (1 / p_years) - 1) * 100
p_rets = pg.pct_change().dropna()
p_vol = p_rets.std() * np.sqrt(252) * 100
p_sharpe = ((p_rets.mean() - rf_daily) * 252) / (p_rets.std() * np.sqrt(252))
p_ds = p_rets[p_rets < 0].std() * np.sqrt(252)
p_sortino = ((p_rets.mean() - rf_daily) * 252) / p_ds if p_ds != 0 else 0
p_rm = pg.cummax()
p_dd = (pg - p_rm) / p_rm
p_maxdd = p_dd.min() * 100
p_calmar = p_cagr / abs(p_maxdd)
p_var5 = p_rets.quantile(0.05) * 100
p_wm = pg.resample("ME").last().pct_change().min() * 100
p_wm_date = pg.resample("ME").last().pct_change().idxmin().strftime("%b %Y")
p_wy = pg.resample("YE").last().pct_change().min() * 100
p_wy_date = pg.resample("YE").last().pct_change().idxmin().strftime("%Y")
p_by = pg.resample("YE").last().pct_change().max() * 100
p_by_date = pg.resample("YE").last().pct_change().idxmax().strftime("%Y")
p_dd_end = p_dd.idxmin()
p_rec_mask = pg.loc[p_dd_end:] >= p_rm.loc[p_dd_end]
p_rec = (p_rec_mask.idxmax() - p_dd_end).days if p_rec_mask.any() else None

all_stats.append({
    "Ticker": "DIVERSIFIED",
    "Start": pg.index[0].strftime("%Y-%m-%d"),
    "End": pg.index[-1].strftime("%Y-%m-%d"),
    "Final Value": f"${p_final:,.0f}",
    "Total Return": f"{(p_final / INITIAL_CAPITAL - 1) * 100:+.1f}%",
    "CAGR": p_cagr,
    "Volatility": p_vol,
    "Sharpe": p_sharpe,
    "Sortino": p_sortino,
    "Calmar": p_calmar,
    "Max DD": p_maxdd,
    "Var 5%": p_var5,
    "Worst Mo": p_wm,
    "Worst Mo Date": p_wm_date,
    "Worst Yr": p_wy,
    "Worst Yr Date": p_wy_date,
    "Best Yr": p_by,
    "Best Yr Date": p_by_date,
    "Recovery (days)": p_rec,
    "Unleveraged ETF": "N/A",
    "Underlying CAGR": None,
    "Actual Multiplier": None,
    "ER": 0.81,
})

# ============================
# PRINT COMPREHENSIVE RESULTS
# ============================
print()
print("=" * 130)
print(f"LEVERAGED ETF BACKTEST  |  ${INITIAL_CAPITAL:,} invested  |  {START_DATE} to {END_DATE} (15 years)")
print("=" * 130)

# Core table
print()
print("CORE PERFORMANCE")
print("-" * 130)
header = f"{'Ticker':<12} {'Start':<12} {'End':<12} {'Final Value':>12} {'Total Ret':>10} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>7} {'Calmar':>7} {'MaxDD':>8} {'VaR5%':>7}"
print(header)
print("-" * 130)
for s in all_stats:
    label = f"*{s['Ticker']}" if s["Ticker"] == BENCHMARK else s['Ticker']
    print(f"{label:<12} {s['Start']:<12} {s['End']:<12} {s['Final Value']:>12} {s['Total Return']:>10} "
          f"{s['CAGR']:>6.1f}% {s['Volatility']:>6.1f}% {s['Sharpe']:>7.2f} {s['Sortino']:>7.2f} {s['Calmar']:>7.2f} "
          f"{s['Max DD']:>7.1f}% {s['Var 5%']:>6.1f}%")

# Risk detail table
print()
print("RISK & DRAWDOWN DETAIL")
print("-" * 130)
header2 = f"{'Ticker':<12} {'Worst Month':>12} {'When':<12} {'Worst Year':>11} {'When':<8} {'Best Year':>10} {'When':<8} {'Recovery':>10}"
print(header2)
print("-" * 130)
for s in all_stats:
    wm = f"{s['Worst Mo']:.1f}%" if s['Worst Mo'] is not None else "N/A"
    wy = f"{s['Worst Yr']:.1f}%" if s['Worst Yr'] is not None else "N/A"
    by = f"{s['Best Yr']:.1f}%" if s['Best Yr'] is not None else "N/A"
    rec = f"{s['Recovery (days)']}d" if s['Recovery (days)'] else "Still down"
    label = f"*{s['Ticker']}" if s["Ticker"] == BENCHMARK else s['Ticker']
    print(f"{label:<12} {wm:>12} {s['Worst Mo Date']:<12} {wy:>11} {s['Worst Yr Date']:<8} {by:>10} {s['Best Yr Date']:<8} {rec:>10}")

# Leverage comparison
print()
print("LEVERAGED vs UNLEVERAGED COUNTERPARTS")
print("-" * 130)
header3 = f"{'Leveraged':<10} {'Lev':<5} {'Underlying':<10} {'Under CAGR':>12} {'Lev CAGR':>10} {'Multiplier':>11} {'Under $':>12} {'Lev $':>12} {'ER':>6}"
print(header3)
print("-" * 130)
for s in all_stats:
    if s["Ticker"] == "DIVERSIFIED":
        continue
    if s["Ticker"] == BENCHMARK:
        print(f"{'SPY':<10} {'1x':<5} {'S&P 500':<10} {'14.0%':>12} {spy_cagr:>9.1f}% {'1.00x':>11} {s['Final Value']:>12} {s['Final Value']:>12} {'0.09%':>6}")
        continue
    info = UNLEVERAGED_MAP[s["Ticker"]]
    lev = f"{info['leverage']}x"
    und_name = s["Unleveraged ETF"] or "N/A"
    und_cagr = f"{s['Underlying CAGR']:.1f}%" if s['Underlying CAGR'] is not None else "N/A"
    mult = f"{s['Actual Multiplier']:.2f}x" if s['Actual Multiplier'] is not None else "N/A"
    und_val = f"${INITIAL_CAPITAL * s['Underlying CAGR']:.0f}" if s['Underlying CAGR'] is not None else "N/A"
    # Calculate underlying final value properly
    if s["Unleveraged ETF"] and s["Unleveraged ETF"] in raw.columns:
        lp = raw[s["Ticker"]].dropna()
        up = raw[s["Unleveraged ETF"]].dropna()
        common = lp.index.intersection(up.index)
        uv = INITIAL_CAPITAL * (up.loc[common].iloc[-1] / up.loc[common].iloc[0])
        und_val = f"${uv:,.0f}"
    else:
        und_val = "N/A"
    print(f"{s['Ticker']:<10} {lev:<5} {und_name:<10} {und_cagr:>12} {s['CAGR']:>9.1f}% {mult:>11} {und_val:>12} {s['Final Value']:>12} {s['ER']:>5.2f}%")

# Expense ratio drag
print()
print("EXPENSE RATIO DRAG (estimated)")
print("-" * 130)
for s in all_stats:
    if s["Ticker"] == "DIVERSIFIED" or s["Ticker"] == BENCHMARK:
        continue
    er = s["ER"]
    ticker = s["Ticker"]
    lev = UNLEVERAGED_MAP[ticker]["leverage"]
    start_d = pd.Timestamp(s["Start"])
    end_d = pd.Timestamp(s["End"])
    years = (end_d - start_d).days / 365.25
    true_est = s["CAGR"] + er
    cumul_drag = INITIAL_CAPITAL * ((1 + true_est / 100) ** years - (1 + s["CAGR"] / 100) ** years)
    print(f"  {ticker}: ER = {er:.2f}%/yr | ~{years:.1f}yr cost: ${abs(cumul_drag):,.0f} | Est true leverage CAGR (0% fees): ~{true_est:.1f}%")

# Correlation
print()
print("CORRELATION MATRIX (daily returns)")
print("-" * 100)
corr_tickers = TICKERS + [BENCHMARK]
lev_returns = raw[corr_tickers].dropna().pct_change().dropna()
corr = lev_returns.corr()
print(corr.round(3).to_string())

# Add SPY to rankings
all_stats_for_ranking = all_stats + [{
    "Ticker": BENCHMARK, "CAGR": spy_cagr, "Volatility": spy_vol, "Sharpe": spy_sharpe,
    "Sortino": spy_sortino, "Calmar": spy_calmar, "Max DD": spy_maxdd,
    "Recovery (days)": spy_rec, "Actual Multiplier": None,
}]

# Rankings
print()
print("=" * 80)
print("RANKINGS")
print("=" * 80)
rankings = [
    ("Best CAGR", "CAGR", True),
    ("Best Sharpe", "Sharpe", True),
    ("Best Sortino", "Sortino", True),
    ("Best Calmar (return/drawdown)", "Calmar", True),
    ("Lowest Volatility", "Volatility", False),
    ("Smallest Max Drawdown", "Max DD", False),
    ("Fastest Recovery", "Recovery (days)", False),
    ("Closest to Advertised Leverage", "Actual Multiplier", True),
]
for label, col, desc in rankings:
    vals = [(s["Ticker"], s[col]) for s in all_stats_for_ranking if s[col] is not None]
    if desc:
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
    else:
        vals_sorted = sorted(vals, key=lambda x: abs(x[1]))
    winner = vals_sorted[0]
    runner = vals_sorted[1] if len(vals_sorted) > 1 else ("", "")
    spy_rank = next((i+1 for i, (t, _) in enumerate(vals_sorted) if t == BENCHMARK), "N/A")
    print(f"  {label:38s} --> {winner[0]} ({winner[1]:.1f})  [SPY: #{spy_rank}]")

# ============================
# CHARTS (5-panel + table)
# ============================
individual_curves[BENCHMARK] = spy_g

fig = plt.figure(figsize=(20, 28))
gs = gridspec.GridSpec(4, 2, height_ratios=[3, 2, 2, 1.5], hspace=0.35, wspace=0.3)
colors = plt.cm.tab10(np.linspace(0, 1, len(TICKERS) + 1))

# 1. Growth (full width)
ax1 = fig.add_subplot(gs[0, :])
for i, ticker in enumerate(TICKERS):
    if ticker in individual_curves:
        ax1.plot(individual_curves[ticker].index, individual_curves[ticker],
                 label=ticker, linewidth=1.3, color=colors[i])
ax1.plot(individual_curves["DIVERSIFIED"].index, individual_curves["DIVERSIFIED"],
         label="DIVERSIFIED", linewidth=2.5, color=colors[-1], linestyle="--")
ax1.plot(individual_curves[BENCHMARK].index, individual_curves[BENCHMARK],
         label=f"{BENCHMARK} (Benchmark)", linewidth=2.2, color="gray", linestyle="-.", alpha=0.9)
ax1.set_title(f"Growth of ${INITIAL_CAPITAL:,} in Leveraged ETFs vs {BENCHMARK} ({START_DATE} - {END_DATE})",
              fontsize=14, fontweight="bold")
ax1.set_ylabel("Portfolio Value ($)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Rolling 1-Year
ax2 = fig.add_subplot(gs[1, 0])
for i, ticker in enumerate(TICKERS):
    if ticker in individual_curves:
        r1y = individual_curves[ticker].pct_change(252) * 100
        ax2.plot(r1y.index, r1y, label=ticker, linewidth=0.9, color=colors[i])
r1y_p = individual_curves["DIVERSIFIED"].pct_change(252) * 100
ax2.plot(r1y_p.index, r1y_p, label="DIVERSIFIED", linewidth=2, color=colors[-1], linestyle="--")
r1y_spy = individual_curves[BENCHMARK].pct_change(252) * 100
ax2.plot(r1y_spy.index, r1y_spy, label=BENCHMARK, linewidth=2, color="gray", linestyle="-.", alpha=0.9)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_title("Rolling 1-Year Return", fontsize=12, fontweight="bold")
ax2.set_ylabel("Return (%)")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax2.legend(fontsize=7, ncol=3)
ax2.grid(True, alpha=0.3)

# 3. Drawdown
ax3 = fig.add_subplot(gs[1, 1])
for i, ticker in enumerate(TICKERS):
    if ticker in individual_curves:
        g = individual_curves[ticker]
        dd = (g - g.cummax()) / g.cummax() * 100
        ax3.plot(dd.index, dd, linewidth=0.8, color=colors[i], label=ticker)
dd_p = (individual_curves["DIVERSIFIED"] - individual_curves["DIVERSIFIED"].cummax()) / individual_curves["DIVERSIFIED"].cummax() * 100
ax3.plot(dd_p.index, dd_p, linewidth=2, color=colors[-1], linestyle="--", label="DIVERSIFIED")
spy_g = individual_curves[BENCHMARK]
dd_spy = (spy_g - spy_g.cummax()) / spy_g.cummax() * 100
ax3.plot(dd_spy.index, dd_spy, linewidth=2, color="gray", linestyle="-.", label=BENCHMARK, alpha=0.9)
ax3.set_title("Drawdown from Peak", fontsize=12, fontweight="bold")
ax3.set_ylabel("Drawdown (%)")
ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax3.legend(fontsize=7, ncol=3)
ax3.grid(True, alpha=0.3)

# 4. Correlation
ax4 = fig.add_subplot(gs[2, 0])
im = ax4.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
ax4.set_xticks(range(len(corr.columns)))
ax4.set_yticks(range(len(corr.columns)))
ax4.set_xticklabels(corr.columns, fontsize=9, rotation=45)
ax4.set_yticklabels(corr.columns, fontsize=9)
for ri in range(len(corr)):
    for ci in range(len(corr)):
        ax4.text(ci, ri, f"{corr.values[ri, ci]:.2f}", ha="center", va="center",
                 fontsize=10, fontweight="bold",
                 color="white" if abs(corr.values[ri, ci]) > 0.7 else "black")
ax4.set_title("Correlation Matrix", fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

# 5. Risk-Return Scatter
ax5 = fig.add_subplot(gs[2, 1])
for s in all_stats:
    if s["Ticker"] == BENCHMARK:
        marker, sz, ec, color_override = "s", 150, "black", "black"
    elif s["Ticker"] == "DIVERSIFIED":
        marker, sz, ec, color_override = "D", 150, "black", None
    else:
        marker, sz, ec, color_override = "o", 90, "none", None
    plot_color = color_override if color_override else "steelblue"
    ax5.scatter(s["Volatility"], s["CAGR"], s=sz, marker=marker, edgecolors=ec, linewidths=2, zorder=5, color=plot_color if s["Ticker"] != "DIVERSIFIED" else colors[-1])
    ax5.annotate(s["Ticker"], (s["Volatility"], s["CAGR"]),
                 textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight="bold")
ax5.set_title("Risk-Return Profile", fontsize=12, fontweight="bold")
ax5.set_xlabel("Annualized Volatility (%)")
ax5.set_ylabel("CAGR (%)")
ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax5.grid(True, alpha=0.3)

# 6. Summary Table (bottom, full width)
ax_table = fig.add_subplot(gs[3, :])
ax_table.axis("off")

# Build table data
table_headers = ["Ticker", "CAGR", "Vol", "Sharpe", "Sortino", "Calmar", "Max DD", "Worst Yr", "Best Yr", "Recovery", "Final Value"]
table_data = []
for s in all_stats:
    wm = f"{s['Worst Yr']:.1f}%" if s['Worst Yr'] is not None else "N/A"
    by = f"{s['Best Yr']:.1f}%" if s['Best Yr'] is not None else "N/A"
    rec = f"{s['Recovery (days)']}d" if s['Recovery (days)'] else "Still down"
    row = [
        s["Ticker"],
        f"{s['CAGR']:.1f}%",
        f"{s['Volatility']:.1f}%",
        f"{s['Sharpe']:.2f}",
        f"{s['Sortino']:.2f}",
        f"{s['Calmar']:.2f}",
        f"{s['Max DD']:.1f}%",
        wm,
        by,
        rec,
        s["Final Value"],
    ]
    table_data.append(row)

table = ax_table.table(
    cellText=table_data,
    colLabels=table_headers,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.5)

# Style header
for j in range(len(table_headers)):
    cell = table[0, j]
    cell.set_text_props(fontweight="bold", color="white")
    cell.set_facecolor("#2c3e50")

# Style rows - alternate colors, highlight SPY and DIVERSIFIED
for i, s in enumerate(all_stats):
    row_idx = i + 1
    if s["Ticker"] == BENCHMARK:
        bg = "#d5e8d4"
    elif s["Ticker"] == "DIVERSIFIED":
        bg = "#fff2cc"
    elif i % 2 == 0:
        bg = "#f8f9fa"
    else:
        bg = "#ffffff"
    for j in range(len(table_headers)):
        table[row_idx, j].set_facecolor(bg)
        if s["Ticker"] in (BENCHMARK, "DIVERSIFIED"):
            table[row_idx, j].set_text_props(fontweight="bold")

ax_table.set_title("Performance Summary", fontsize=13, fontweight="bold", pad=20)

# ============================
# DCA PROJECTION: $100/month for 40 years
# ============================
DCA_MONTHLY = 100
DCA_YEARS = 40
DCA_MONTHS = DCA_YEARS * 12

print()
print("=" * 140)
print(f"DCA PROJECTION: ${DCA_MONTHLY}/month for {DCA_YEARS} years (${DCA_MONTHLY * DCA_MONTHS:,} total invested)")
print("=" * 140)

# Compute historical annual returns for scenario building
proj_results = {}
for ticker in TICKERS + [BENCHMARK]:
    lp = raw[ticker].dropna()
    if len(lp) < 252:
        continue
    daily_ret = lp.pct_change().dropna()
    annual_ret = lp.resample("YE").last().pct_change().dropna()

    avg_annual = annual_ret.mean()
    std_annual = annual_ret.std()
    worst_25 = annual_ret.quantile(0.25)
    best_25 = annual_ret.quantile(0.75)

    # Scenario definitions
    scenarios = {
        "Worst Case (25th %ile)": worst_25,
        "Average (Historical Mean)": avg_annual,
        "Best Case (75th %ile)": best_25,
    }

    proj_results[ticker] = {"scenarios": {}, "annual_returns": annual_ret}

    # Forward projection using3 scenarios
    for scenario_name, annual_r in scenarios.items():
        monthly_r = (1 + annual_r) ** (1/12) - 1
        # DCA: invest $100/month
        portfolio = 0
        for m in range(DCA_MONTHS):
            portfolio = (portfolio + DCA_MONTHLY) * (1 + monthly_r)
        proj_results[ticker]["scenarios"][scenario_name] = {
            "final": portfolio,
            "annual_rate": annual_r * 100,
        }

# Print projection table
print()
print(f"{'Ticker':<12} {'Scenario':<30} {'Ann. Rate':>10} {'Final Value':>14} {'Total In':>10} {'Profit':>14}")
print("-" * 140)

for ticker in TICKERS + [BENCHMARK]:
    if ticker not in proj_results:
        continue
    total_invested = DCA_MONTHLY * DCA_MONTHS
    for scenario_name in ["Worst Case (25th %ile)", "Average (Historical Mean)", "Best Case (75th %ile)"]:
        s = proj_results[ticker]["scenarios"][scenario_name]
        profit = s["final"] - total_invested
        print(f"{ticker:<12} {scenario_name:<30} {s['annual_rate']:>9.1f}% ${s['final']:>13,.0f} ${total_invested:>9,} ${profit:>+13,.0f}")
    print()

# ============================
# LUMP SUM PROJECTION: $250 once, held for 40 years
# ============================
LUMP_SUM = 250
LUMP_YEARS = 40

print()
print("=" * 140)
print(f"LUMP SUM PROJECTION: ${LUMP_SUM} once, held for {LUMP_YEARS} years")
print("=" * 140)

lump_results = {}
for ticker in TICKERS + [BENCHMARK]:
    lp = raw[ticker].dropna()
    if len(lp) < 252:
        continue
    annual_ret = lp.resample("YE").last().pct_change().dropna()

    avg_annual = annual_ret.mean()
    worst_25 = annual_ret.quantile(0.25)
    best_25 = annual_ret.quantile(0.75)

    scenarios = {
        "Worst Case (25th %ile)": worst_25,
        "Average (Historical Mean)": avg_annual,
        "Best Case (75th %ile)": best_25,
    }

    lump_results[ticker] = {"scenarios": {}}
    for scenario_name, annual_r in scenarios.items():
        final = LUMP_SUM * (1 + annual_r) ** LUMP_YEARS
        lump_results[ticker]["scenarios"][scenario_name] = {
            "final": final,
            "annual_rate": annual_r * 100,
        }

print()
print(f"{'Ticker':<12} {'Scenario':<30} {'Ann. Rate':>10} {'Final Value':>14} {'Profit':>14} {'Multiplier':>12}")
print("-" * 140)

for ticker in TICKERS + [BENCHMARK]:
    if ticker not in lump_results:
        continue
    for scenario_name in ["Worst Case (25th %ile)", "Average (Historical Mean)", "Best Case (75th %ile)"]:
        s = lump_results[ticker]["scenarios"][scenario_name]
        profit = s["final"] - LUMP_SUM
        mult = s["final"] / LUMP_SUM
        print(f"{ticker:<12} {scenario_name:<30} {s['annual_rate']:>9.1f}% ${s['final']:>13,.0f} ${profit:>+13,.0f} {mult:>10.1f}x")
    print()

# ============================
# PROJECTION CHARTS
# ============================
# Figure 1: DCA Projections
fig2 = plt.figure(figsize=(18, 16))
gs2 = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

# 1. DCA Bar chart
ax_proj1 = fig2.add_subplot(gs2[0, :])
tickers_with_data = [t for t in TICKERS + [BENCHMARK] if t in proj_results]
x = np.arange(len(tickers_with_data))
width = 0.25
scenarios_list = ["Worst Case (25th %ile)", "Average (Historical Mean)", "Best Case (75th %ile)"]
scenario_colors = ["#e74c3c", "#3498db", "#2ecc71"]

for si, (sc_name, sc_color) in enumerate(zip(scenarios_list, scenario_colors)):
    vals = [proj_results[t]["scenarios"][sc_name]["final"] for t in tickers_with_data]
    # Cap display at 1B for readability
    display_vals = [min(v, 1e9) for v in vals]
    bars = ax_proj1.bar(x + (si - 1) * width, display_vals, width, label=sc_name, color=sc_color, alpha=0.85)
    for bar, val in zip(bars, vals):
        label = f"${val/1e9:.1f}B" if val >= 1e9 else f"${val/1e6:.1f}M" if val >= 1e6 else f"${val/1e3:.0f}K"
        ax_proj1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(display_vals)*0.01,
                     label, ha="center", va="bottom", fontsize=7, fontweight="bold")

ax_proj1.set_xticks(x)
ax_proj1.set_xticklabels(tickers_with_data, fontsize=10, fontweight="bold")
ax_proj1.set_ylabel("Portfolio Value ($)")
ax_proj1.set_title(f"$100/month DCA for {DCA_YEARS} Years (${DCA_MONTHLY * DCA_MONTHS:,} invested)", fontsize=14, fontweight="bold")
ax_proj1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_proj1.legend(fontsize=10)
ax_proj1.grid(True, alpha=0.3, axis="y")
ax_proj1.axhline(y=DCA_MONTHLY * DCA_MONTHS, color="black", linestyle="--", linewidth=1, alpha=0.5)

# 2. Lump Sum Bar chart
ax_proj_lump = fig2.add_subplot(gs2[1, 0])
lump_tickers = [t for t in TICKERS + [BENCHMARK] if t in lump_results]
x_l = np.arange(len(lump_tickers))

for si, (sc_name, sc_color) in enumerate(zip(scenarios_list, scenario_colors)):
    vals = [lump_results[t]["scenarios"][sc_name]["final"] for t in lump_tickers]
    display_vals = [min(v, 1e8) for v in vals]
    bars = ax_proj_lump.bar(x_l + (si - 1) * width, display_vals, width, label=sc_name, color=sc_color, alpha=0.85)
    for bar, val in zip(bars, vals):
        label = f"${val/1e9:.1f}B" if val >= 1e9 else f"${val/1e6:.1f}M" if val >= 1e6 else f"${val/1e3:.0f}K"
        ax_proj_lump.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(display_vals)*0.01,
                         label, ha="center", va="bottom", fontsize=7, fontweight="bold")
ax_proj_lump.set_xticks(x_l)
ax_proj_lump.set_xticklabels(lump_tickers, fontsize=10, fontweight="bold")
ax_proj_lump.set_ylabel("Portfolio Value ($)")
ax_proj_lump.set_title(f"${LUMP_SUM} Once, Held {LUMP_YEARS} Years", fontsize=12, fontweight="bold")
ax_proj_lump.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax_proj_lump.legend(fontsize=8)
ax_proj_lump.grid(True, alpha=0.3, axis="y")
ax_proj_lump.axhline(y=LUMP_SUM, color="black", linestyle="--", linewidth=1, alpha=0.5)

# 3. Lump Sum Multiplier
ax_proj_mult = fig2.add_subplot(gs2[1, 1])
for si, (sc_name, sc_color) in enumerate(zip(scenarios_list, scenario_colors)):
    mults = [lump_results[t]["scenarios"][sc_name]["final"] / LUMP_SUM for t in lump_tickers]
    display_mults = [min(m, 1e6) for m in mults]
    bars = ax_proj_mult.bar(x_l + (si - 1) * width, display_mults, width, label=sc_name, color=sc_color, alpha=0.85)
    for bar, m in zip(bars, mults):
        label = f"{m/1e6:.1f}Mx" if m >= 1e6 else f"{m/1e3:.0f}Kx" if m >= 1e3 else f"{m:.0f}x"
        ax_proj_mult.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(display_mults)*0.01,
                         label, ha="center", va="bottom", fontsize=7, fontweight="bold")
ax_proj_mult.set_xticks(x_l)
ax_proj_mult.set_xticklabels(lump_tickers, fontsize=10, fontweight="bold")
ax_proj_mult.set_ylabel("Multiplier (x of $250)")
ax_proj_mult.set_title(f"${LUMP_SUM} Lump Sum: Return Multiplier After {LUMP_YEARS} Years", fontsize=12, fontweight="bold")
ax_proj_mult.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax_proj_mult.legend(fontsize=8)
ax_proj_mult.grid(True, alpha=0.3, axis="y")

fig2.suptitle("INVESTMENT PROJECTIONS", fontsize=16, fontweight="bold", y=1.01)
plt.savefig("backtest_projection.png", dpi=150, bbox_inches="tight")
print("Projection chart saved to backtest_projection.png")

plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
print("\nChart saved to backtest_results.png")
