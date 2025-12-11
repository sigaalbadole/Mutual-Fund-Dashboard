from flask import Flask, render_template, request, redirect
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import timedelta

app = Flask(__name__)

excel_path = "data/Book.xlsx"
xls = pd.ExcelFile(excel_path)
FUND_HOUSES = xls.sheet_names


# -------------------------------------------------------------
# SAFE VALUE PICKER (fixes all IndexErrors)
# -------------------------------------------------------------
def safe_pick(df, mask, column, default=0):
    row = df.loc[mask, column]
    return row.iloc[0] if not row.empty else default


# -------------------------------------------------------------
# Helper: return NAV value on or before a given date
# -------------------------------------------------------------
def get_nav_before(df, target_date):
    df2 = df[df["Date"] <= target_date]
    if df2.empty:
        return None
    return df2.sort_values("Date").iloc[-1]["Net Asset Value"]


# -------------------------------------------------------------
# Compute CAGRs & Sharpe Ratio for each fund
# Using 1yr, 2yr, 3yr, 4yr ONLY
# -------------------------------------------------------------
def compute_cagr_and_sharpe(df):

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # MASTER FUND LIST (ensures consistency across all periods)
    ALL_FUNDS = sorted(df["Scheme Name"].unique().tolist())

    periods = {
        "CAGR 1yr": 365,
        "CAGR 2yr": 365 * 2,
        "CAGR 3yr": 365 * 3,
        "CAGR 4yr": 365 * 4,
    }

    last_date = df["Date"].max()
    results = []

    for scheme in ALL_FUNDS:

        temp = df[df["Scheme Name"] == scheme].copy()
        temp = temp.sort_values("Date")

        row = {"Scheme Name": scheme}

        # Last NAV if present
        if temp.empty:
            last_nav = None
        else:
            last_nav = temp.iloc[-1]["Net Asset Value"]

        # ---- CAGR values ----
        for col, days in periods.items():

            if last_nav is None:
                row[col] = 0
                continue

            start_date = last_date - timedelta(days=days)
            start_nav = get_nav_before(temp, start_date)

            if start_nav is None:
                row[col] = 0       # KEY FIX: ensure consistent columns
            else:
                cagr = (last_nav / start_nav) ** (365 / days) - 1
                row[col] = cagr * 100

        # ---- Sharpe Ratio ---
        if temp.shape[0] > 1:
            temp["Return"] = temp["Net Asset Value"].pct_change()
            rf = (1 + 0.04)**(1/365) - 1
            excess = temp["Return"] - rf
            row["Sharpe Ratio"] = excess.mean() / excess.std() if excess.std() != 0 else 0
        else:
            row["Sharpe Ratio"] = 0

        results.append(row)

    df_out = pd.DataFrame(results)

    return df_out.fillna(0)



# -------------------------------------------------------------
# Compute Nifty metrics for same periods (1yr, 2yr, 3yr, 4yr)
# -------------------------------------------------------------
def compute_nifty_metrics():

    nifty = pd.read_csv("data/Nifty 50 Historical Data.csv")

    nifty.columns = nifty.columns.str.strip().str.lower()
    nifty = nifty.rename(columns={"price": "Close", "date": "Date"})
    nifty["Close"] = nifty["Close"].astype(str).str.replace(",", "").astype(float)
    nifty["Date"] = pd.to_datetime(nifty["Date"])
    nifty = nifty.sort_values("Date")

    last_date = nifty["Date"].max()
    last_close = safe_pick(nifty, nifty["Date"] == last_date, "Close", default=0)

    periods = {
        "CAGR 1yr": 365,
        "CAGR 2yr": 365 * 2,
        "CAGR 3yr": 365 * 3,
        "CAGR 4yr": 365 * 4,
    }

    rows = []
    for name, days in periods.items():
        lookback = last_date - timedelta(days=days)
        df2 = nifty[nifty["Date"] <= lookback]

        if df2.empty:
            rows.append({"Period": name, "CAGR (%)": 0})
            continue

        start_close = df2.iloc[-1]["Close"]
        cagr = ((last_close / start_close) ** (365 / days) - 1) * 100

        rows.append({"Period": name, "CAGR (%)": cagr})

    return pd.DataFrame(rows)


# -------------------------------------------------------------
# Plot 1 — CAGR vs Nifty (with safe indexing)
# -------------------------------------------------------------
def make_cagr_subplot(cagr_df, nifty_cagr, periods):

    funds = cagr_df["Scheme Name"].unique()
    palette = px.colors.qualitative.Plotly
    colors = {fund: palette[i % len(palette)] for i, fund in enumerate(funds)}

    rows = (len(periods) + 1) // 2
    cols = 2 if len(periods) > 1 else 1

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=periods, shared_yaxes=False
    )

    positions = [(i // 2 + 1, i % 2 + 1) for i in range(len(periods))]

    for period, (r, c) in zip(periods, positions):

        nifty_value = safe_pick(nifty_cagr, nifty_cagr["Period"] == period, "CAGR (%)", 0)

        for fund in funds:

            fund_value = safe_pick(
                cagr_df,
                cagr_df["Scheme Name"] == fund,
                period,
                default=0
            )

            sharpe = safe_pick(
                cagr_df,
                cagr_df["Scheme Name"] == fund,
                "Sharpe Ratio",
                default=0
            )

            fig.add_trace(
                go.Bar(
                    x=[fund],
                    y=[fund_value],
                    marker=dict(
                        color=colors[fund],
                        line=dict(width=1, color=colors[fund])
                    ),
                    hovertemplate=f"{fund}<br>{period}: {fund_value:.2f}%<br>Sharpe: {sharpe:.2f}<extra></extra>"
                ),
                row=r, col=c
            )

        fig.add_trace(
            go.Scatter(
                x=funds,
                y=[nifty_value] * len(funds),
                mode="lines",
                line=dict(color="black", width=2),
                hovertemplate=f"Nifty {period}: {nifty_value:.2f}%<extra></extra>"
            ),
            row=r, col=c
        )

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(
        height=400 * rows,
        width=1000,
        barmode="group",
        showlegend=False,
        title_text="Fund CAGR vs Nifty (1yr, 2yr, 3yr, 4yr)"
    )

    return fig.to_html(full_html=False)


# -------------------------------------------------------------
# Plot 2 — NAV history
# -------------------------------------------------------------
def make_nav_plot(nav_df, nifty_cagr, periods):

    selected = nav_df.copy()
    for period in periods:
        nifty_val = safe_pick(nifty_cagr, nifty_cagr["Period"] == period, "CAGR (%)", 0)
        selected = selected[selected[period] > nifty_val]

    if selected.empty:
        return "<p>No funds outperform Nifty.</p>"

    mf = selected.copy()
    mf["Date"] = pd.to_datetime(mf["Date"])
    mf = mf.sort_values(["Scheme Name", "Date"])

    cagr_cols = [col for col in nav_df.columns if col.startswith("CAGR")]

    fig = px.line(
        mf,
        x="Date",
        y="Net Asset Value",
        color="Scheme Name",
        hover_data={col: ':.2f' for col in cagr_cols},
        title="Historical NAV of Outperforming Funds"
    )

    fig.update_layout(height=600, width=900, legend_title="Fund")

    return fig.to_html(full_html=False)


# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        house = request.form["fund_house"]
        periods = request.form.getlist("periods")
        qs = "&".join([f"periods={p}" for p in periods])
        return redirect(f"/charts/{house}?{qs}")
    return render_template("index.html", fund_houses=FUND_HOUSES)


@app.route("/charts/<fund_house>")
def charts(fund_house):

    df = xls.parse(fund_house)
    df = df[['Scheme Name', 'Net Asset Value', 'Date']].dropna()
    df["Scheme Name"] = df["Scheme Name"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Net Asset Value"] = df["Net Asset Value"].astype(float)

    # Keep only Growth + Regular
    df = df[
        df["Scheme Name"].str.contains("Growth", case=False) &
        df["Scheme Name"].str.contains("Regular", case=False)
    ].reset_index(drop=True)

    cagr_df = compute_cagr_and_sharpe(df)
    nifty_cagr = compute_nifty_metrics()

    selected_periods = request.args.getlist("periods")
    if not selected_periods:
        selected_periods = ["CAGR 1yr", "CAGR 2yr", "CAGR 3yr", "CAGR 4yr"]

    fig1_html = make_cagr_subplot(cagr_df, nifty_cagr, selected_periods)

    nav_df = df.merge(cagr_df, on="Scheme Name", how="left")

    fig2_html = make_nav_plot(nav_df, nifty_cagr, selected_periods)

    return render_template("charts.html", fig1=fig1_html, fig2=fig2_html, fund_house=fund_house)


if __name__ == "__main__":
    app.run(debug=True)
