import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODEL_COLS = {
    "Persistence": "persistence_mw",
    "Previous day": "previous_day_mw",
    "Linear Regression": "linear_regression_mw",
    "Random Forest": "random_forest_mw",
    "Gradient Boosting": "gradient_boosting_mw",
}

st.set_page_config(page_title="India Demand Forecasting", layout="wide")


@st.cache_data
def load_predictions(h):
    return pd.read_csv(RES / f"predictions_test_h{h}.csv", parse_dates=["origin_time", "target_time"])


@st.cache_data
def load_warnings(h):
    return pd.read_csv(RES / f"peak_warnings_test_h{h}.csv", parse_dates=["target_time"])


@st.cache_data
def load_history():
    return pd.read_csv(ROOT / "data" / "processed" / "india_demand_weather.csv", parse_dates=["datetime"])


@st.cache_data
def load_tables():
    final = pd.read_csv(RES / "tables" / "final_test_results.csv")
    perf = pd.read_csv(RES / "tables" / "peak_warning_performance_test.csv")
    best = json.load(open(RES / "tables" / "best_params.json"))
    return final, perf, best


def date_window(label, series, default_days, key, anchor=None):
    lo, hi = series.min().date(), series.max().date()
    end = (anchor or series.max()).date()
    start = max(lo, (pd.Timestamp(end) - pd.Timedelta(days=default_days)).date())
    chosen = st.date_input(label, value=(start, min(hi, end)), min_value=lo, max_value=hi, key=key)
    if isinstance(chosen, tuple) and len(chosen) == 2:
        return pd.Timestamp(chosen[0]), pd.Timestamp(chosen[1]) + pd.Timedelta(days=1)
    return pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)


try:
    final, perf, best = load_tables()
    history = load_history()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run notebooks 02, 06 and 07 and commit their outputs first.")
    st.stop()

st.title("Short-Term Electricity Demand Forecasting - Indian Power Grid")
st.caption("CSE570 Machine Learning with Python | Hourly national demand, 1-hour and 24-hour-ahead forecasts, "
           "test period Jan 2023 - Apr 2024")

tab_fc, tab_cmp, tab_hist, tab_warn, tab_about = st.tabs(
    ["Forecast vs actual", "Model comparison", "Historical demand", "High-demand warnings", "About"])

# ---------------- Forecast vs actual ----------------
with tab_fc:
    c1, c2 = st.columns([1, 3])
    with c1:
        h = st.radio("Forecast horizon", [1, 24], format_func=lambda x: f"{x} hour ahead", key="fc_h")
        pred = load_predictions(h)
        best_model = best[str(h)]["best_model"]
        models = st.multiselect("Models", list(MODEL_COLS), default=[best_model, "Previous day"], key="fc_models")
        peak_time = pred.loc[pred["actual_mw"].idxmax(), "target_time"]
        t0, t1 = date_window("Date range", pred["target_time"], 6, "fc_dates", anchor=peak_time + pd.Timedelta(days=3))
    w = pred[(pred["target_time"] >= t0) & (pred["target_time"] < t1)]
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w["target_time"], y=w["actual_mw"], name="Actual", line=dict(color="black", width=2)))
        for m in models:
            fig.add_trace(go.Scatter(x=w["target_time"], y=w[MODEL_COLS[m]], name=m, line=dict(width=1.4)))
        fig.update_layout(height=430, yaxis_title="Demand (MW)", margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
    if len(w) and models:
        rows = []
        for m in models:
            err = w[MODEL_COLS[m]] - w["actual_mw"]
            rows.append({"Model": m, "MAE (MW)": err.abs().mean(), "MAPE (%)": (err.abs() / w["actual_mw"]).mean() * 100})
        st.markdown("**Accuracy in the selected window**")
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)

# ---------------- Model comparison ----------------
with tab_cmp:
    st.markdown("Final evaluation on the held-out **test set** (Jan 2023 - Apr 2024). Models were tuned with "
                "time-series cross-validation on 2019-2022 data only.")
    f = final.copy(); f["Horizon"] = f["horizon_h"].astype(str) + " h"
    fig = px.bar(f, x="model", y="MAE (MW)", color="Horizon", barmode="group", height=420)
    st.plotly_chart(fig)
    st.dataframe(final.drop(columns=["horizon_h"]).assign(Horizon=final["horizon_h"].astype(str) + " h")
                 .set_index(["Horizon", "model"]))

# ---------------- Historical demand ----------------
with tab_hist:
    region = st.selectbox("Region", ["national", "northern", "western", "eastern", "southern", "north_eastern"])
    daily = history.set_index("datetime")[region].resample("D").mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="Daily mean", line=dict(color="#9bb7d4", width=1)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily.rolling(30, center=True).mean(), name="30-day mean", line=dict(color="#c0392b", width=2)))
    fig.update_layout(height=380, yaxis_title="MW", title=f"{region.replace('_', ' ').title()} demand, 2019 - 2024",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)
    piv = (history.assign(hour=history["datetime"].dt.hour, month=history["datetime"].dt.month)
           .pivot_table(index="hour", columns="month", values=region, aggfunc="mean"))
    st.plotly_chart(px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", origin="lower",
                              labels=dict(x="Month", y="Hour", color="MW"), title="Average demand by hour and month", height=400))

# ---------------- Warnings ----------------
with tab_warn:
    h = st.radio("Forecast horizon", [1, 24], format_func=lambda x: f"{x} hour ahead", key="w_h", horizontal=True)
    tl = load_warnings(h)
    p = perf[perf["horizon_h"] == h]
    st.markdown("A **high-demand hour** is one where demand exceeds the 95th percentile of the previous 365 days. "
                "A **warning** is issued when the forecast exceeds the threshold (minus a safety margin chosen on validation data).")
    cols = st.columns(len(p))
    for col, (_, r) in zip(cols, p.iterrows()):
        col.metric(r["method"], f"F1 {r['F1']:.2f}", f"recall {r['recall']:.2f} | precision {r['precision']:.2f}", delta_color="off")
    st.dataframe(p.set_index("method")[["margin", "true_high_hours", "precision", "recall", "F1", "false_alarm_rate",
                                         "true_episodes", "episode_recall", "episode_precision"]])
    t0, t1 = date_window("Date range", tl["target_time"], 30, "w_dates", anchor=pd.Timestamp("2023-06-30"))
    w = tl[(tl["target_time"] >= t0) & (tl["target_time"] < t1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["target_time"], y=w["actual_mw"], name="Actual", line=dict(color="black", width=1.2)))
    fig.add_trace(go.Scatter(x=w["target_time"], y=w["predicted_mw"], name="Forecast", line=dict(color="#d95f0e", width=1)))
    fig.add_trace(go.Scatter(x=w["target_time"], y=w["threshold_mw"], name="High-demand threshold", line=dict(color="red", dash="dash")))
    ww = w[w["warning"] == 1]
    fig.add_trace(go.Scatter(x=ww["target_time"], y=ww["predicted_mw"], mode="markers", name="Warning", marker=dict(color="blue", size=5)))
    fig.update_layout(height=430, yaxis_title="Demand (MW)", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig)
    days = (tl[tl["warning"] == 1].assign(date=lambda d: d["target_time"].dt.date).groupby("date")
            .agg(warning_hours=("warning", "sum"), peak_forecast_mw=("predicted_mw", "max"), peak_actual_mw=("actual_mw", "max"))
            .sort_index(ascending=False).head(15).round(0))
    st.markdown("**Most recent days with warnings**")
    st.dataframe(days)

# ---------------- About ----------------
with tab_about:
    st.markdown("""
**Data:** India hourly national and regional demand (Jan 2019 - Apr 2024) and hourly weather for five regional cities.
**Models:** Linear Regression, Random Forest, Gradient Boosting versus persistence and previous-day baselines.
**Validation:** chronological split (train 2019-2021, validation 2022, test 2023-2024) and time-series cross-validation.
**Limitations:** weather at the forecast time is actual weather standing in for a weather forecast, so weather gains are optimistic;
the dashboard shows precomputed test-period results, not live data.
""")