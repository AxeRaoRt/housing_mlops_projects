import os
import base64
import numpy as np
import plotly.graph_objects as go
import plotly.express as px 
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Config
# -----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8001")
PRED_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5001")

ASSETS_DIR = Path(__file__).parent / "assets"
DATA_PATH = Path(__file__).parent.parent / "data" / "creditcard.csv"


def b64_image(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def pretty_json(obj) -> str:
    import json
    return json.dumps(obj, indent=2, ensure_ascii=False)


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def risk_proxy(row: pd.Series) -> float:
    import numpy as np
    v14 = abs(float(row.get("V14", 0.0)))
    v12 = abs(float(row.get("V12", 0.0)))
    v10 = abs(float(row.get("V10", 0.0)))
    amt = float(row.get("Amount", 0.0))

    z = 0.85*v14 + 0.65*v12 + 0.55*v10 + 0.12*np.log1p(amt)
    return float(1 / (1 + np.exp(-0.35*(z - 5.0))))


# Load images (safe if missing)
BG = b64_image(ASSETS_DIR / "bg.png")
CARDS_TOP = b64_image(ASSETS_DIR / "cards_top_right.png")
CARDS_BOTTOM = b64_image(ASSETS_DIR / "cards_bottom_left.png")
PAY_HAND = b64_image(ASSETS_DIR / "payment_hand.png")




# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="AI Fraud Detection — Neon Dashboard", layout="wide")

# -----------------------------
# Neon CSS (SAFE + KPI readable)
# -----------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Space+Grotesk:wght@500;700&display=swap');

:root {{
  --bg0: #070913;
  --bg1: #0b1026;
  --card: rgba(10, 14, 32, 0.78);
  --card2: rgba(12, 16, 38, 0.62);
  --cyan: #41f0ff;
  --purple: #b06cff;
  --pink: #ff4fd8;
  --text: rgba(240, 244, 255, 0.92);
  --muted: rgba(205, 214, 255, 0.62);
  --shadow: 0 18px 55px rgba(0,0,0,0.55);
}}

.stApp {{
  background:
    radial-gradient(1100px 520px at 15% 10%, rgba(65,240,255,0.12), transparent 55%),
    radial-gradient(900px 520px at 85% 20%, rgba(176,108,255,0.14), transparent 55%),
    radial-gradient(900px 520px at 50% 90%, rgba(255,79,216,0.10), transparent 55%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
  color: var(--text);
}}

.block-container {{
  padding-top: 1.2rem !important;
  padding-bottom: 2rem !important;
}}

h1, h2, h3, h4 {{
  font-family: "Space Grotesk", Inter, sans-serif !important;
  color: var(--text) !important;
}}
p, div, span, label {{
  font-family: Inter, sans-serif !important;
}}
.hero {{
  position: relative;
  padding: 1.8rem 1.8rem;
  padding-right: 380px;   /* ✅ reserve space for large image */
  border-radius: 20px;
  background: linear-gradient(180deg, rgba(10,14,32,0.85), rgba(10,14,32,0.55));
  border: 1px solid rgba(65,240,255,0.18);
  box-shadow: var(--shadow);
  backdrop-filter: blur(14px);
  overflow: hidden;
}}

.hero::after {{
  content: "";
  position: absolute;
  inset: -2px;
  background:
    radial-gradient(600px 160px at 20% 20%, rgba(65,240,255,0.22), transparent 60%),
    radial-gradient(620px 160px at 85% 40%, rgba(176,108,255,0.22), transparent 60%);
  filter: blur(6px);
  opacity: 0.85;
  pointer-events: none;
  z-index: 1;
}}

.hero h1 {{
  margin: 0;
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  position: relative;
  z-index: 2;
}}

.hero p {{
  margin-top: .55rem;
  color: var(--muted);
  font-size: 1.02rem;
  position: relative;
  z-index: 2;
}}

.hero-img {{
  position: absolute;
  top: -20px;         /* ✅ lift it slightly for cinematic look */
  right: -40px;       /* ✅ push slightly outside edge */
  width: 560px;       /* ✅ large dramatic image */
  max-width: 60%;
  opacity: 0.98;
  pointer-events: none;
  z-index: 2;
  filter: drop-shadow(0 0 30px rgba(65,240,255,0.25));
}}

@media (max-width: 1100px) {{
  .hero {{
    padding-right: 1.8rem;  /* remove reserved space */
  }}
  .hero-img {{
    position: relative;
    top: 0;
    right: 0;
    width: 260px;
    margin-top: 20px;
  }}
}}





.card {{
  border-radius: 20px;
  background: linear-gradient(180deg, var(--card), var(--card2));
  border: 1px solid rgba(255,255,255,0.06);
  box-shadow: var(--shadow);
  backdrop-filter: blur(16px);
  padding: 1.15rem 1.15rem;
  position: relative;
  overflow: hidden;
}}

.card::before {{
  content:"";
  position:absolute;
  inset:-1px;
  border-radius: 20px;
  padding: 1px;
  background: linear-gradient(135deg, rgba(65,240,255,0.30), rgba(176,108,255,0.24), rgba(255,79,216,0.18));
  -webkit-mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0.55;
  pointer-events:none;
}}

.badge {{
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  padding: .34rem .7rem;
  border-radius: 999px;
  font-weight: 800;
  font-size: .88rem;
  letter-spacing: 0.01em;
}}

.badge-ok {{
  background: rgba(65,240,255,0.10);
  color: rgba(206, 255, 255, 0.95);
  border: 1px solid rgba(65,240,255,0.30);
  box-shadow: 0 0 18px rgba(65,240,255,0.12);
}}

.badge-bad {{
  background: rgba(255,79,216,0.12);
  color: rgba(255, 205, 245, 0.95);
  border: 1px solid rgba(255,79,216,0.32);
  box-shadow: 0 0 18px rgba(255,79,216,0.12);
}}

.stButton > button {{
  border-radius: 14px !important;
  padding: 0.8rem 1rem !important;
  font-weight: 800 !important;
  border: 1px solid rgba(65,240,255,0.22) !important;
  background: rgba(10,14,32,0.65) !important;
  color: var(--text) !important;
}}
.stButton > button:hover {{
  border-color: rgba(176,108,255,0.38) !important;
  box-shadow: 0 0 22px rgba(176,108,255,0.18) !important;
  transform: translateY(-1px);
}}

div[data-testid="stButton"] > button[kind="primary"] {{
  background: linear-gradient(90deg, rgba(65,240,255,0.20), rgba(176,108,255,0.20)) !important;
  border: 1px solid rgba(65,240,255,0.34) !important;
  box-shadow: 0 0 26px rgba(65,240,255,0.14) !important;
}}

.small-muted {{
  color: var(--muted);
  font-size: .9rem;
}}

/* --- KPI / metric readability --- */
div[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(10,14,32,0.78), rgba(12,16,38,0.60));
  border: 1px solid rgba(65,240,255,0.16);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 18px 55px rgba(0,0,0,0.45);
}}

div[data-testid="stMetricLabel"] {{
  color: rgba(205, 214, 255, 0.85) !important;
  font-weight: 700 !important;
}}

div[data-testid="stMetricValue"] {{
  color: rgba(240, 244, 255, 0.98) !important;
  font-weight: 900 !important;
  text-shadow: 0 0 18px rgba(65,240,255,0.10);
}}

div[data-testid="stMetricDelta"] {{
  color: rgba(65,240,255,0.85) !important;
  font-weight: 700 !important;
}}

/* ===== FORCE KPI WHITE BOLD ===== */

div[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(10,14,32,0.92), rgba(12,16,38,0.80));
  border: 1px solid rgba(65,240,255,0.25);
  border-radius: 18px;
  padding: 20px 18px;
  box-shadow: 0 0 30px rgba(65,240,255,0.12);
}}

div[data-testid="stMetric"] label {{
  color: #ffffff !important;
  font-weight: 800 !important;
  font-size: 1rem !important;
  letter-spacing: 0.5px;
  opacity: 1 !important;
}}

div[data-testid="stMetric"] div {{
  color: #ffffff !important;
  font-weight: 900 !important;
  font-size: 2.1rem !important;
}}

div[data-testid="stMetric"] * {{
  color: #ffffff !important;
}}


</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Session init
# -----------------------------
if "payload" not in st.session_state:
    st.session_state.payload = None


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    f"""
<div class="hero">
  <h1>🧠 AI Fraud Detection Dashboard</h1>
  <p>Streamlit frontend → FastAPI <code>/predict</code> → Prometheus → Grafana (Neon Version C)</p>
  {"<img class='hero-img' src='data:image/png;base64," + BG + "' />" if BG else ""}
</div>
""",
    unsafe_allow_html=True,
)


st.write("")

# -----------------------------
# KPI strip (UI only)
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Transactions", "1,257,943")
k2.metric("Active Alerts", "8,452")
k3.metric("High-Risk Users", "5,210")
k4.metric("Chargeback Rate", "4.3%")

st.write("")


# -----------------------------
# Visualizations (no location needed)
# -----------------------------
df_all = load_dataset(DATA_PATH)

# ===== Row 1: Full-width trend =====
st.markdown("<div class='card'><h3>📈 Fraud Probability Over Time</h3>", unsafe_allow_html=True)

if not df_all.empty and "Time" in df_all.columns:
    tmp = df_all.copy()

    # sample for speed if needed
    if len(tmp) > 25000:
        tmp = tmp.sample(25000, random_state=42)

    tmp["risk_proxy"] = tmp.apply(risk_proxy, axis=1)
    tmp["minute"] = (tmp["Time"] // 60).astype(int)

    trend = (
        tmp.groupby("minute", as_index=False)["risk_proxy"]
           .mean()
           .sort_values("minute")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["minute"],
        y=trend["risk_proxy"],
        mode="lines",
        name="Risk trend"
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Time (minutes)"),
        yaxis=dict(title="Risk (proxy)", range=[0, 1]),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: This is a proxy risk curve using |V14|, |V12|, |V10| and Amount — not the model output.")
else:
    st.info("Add data/creditcard.csv to enable time-based visuals.")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# ===== Row 2: Split layout =====
colL, colR = st.columns([1.35, 1])

# ---- Left: Amount distribution
with colL:
    st.markdown("<div class='card'><h3>📊 Transaction Amount Distribution</h3>", unsafe_allow_html=True)

    if not df_all.empty and "Amount" in df_all.columns:
        tmp = df_all.copy()
        if "Class" in tmp.columns:
            tmp["Label"] = tmp["Class"].map({0: "Legit", 1: "Fraud"}).fillna("Unknown")
        else:
            tmp["Label"] = "Unknown"

        bins = [0, 10, 50, 100, 500, 1000, 5000, tmp["Amount"].max() + 1]
        labels = ["<10", "10-50", "50-100", "100-500", "500-1k", "1k-5k", ">5k"]
        tmp["amt_bin"] = pd.cut(tmp["Amount"], bins=bins, labels=labels, include_lowest=True)

        bar = (
            tmp.groupby(["amt_bin", "Label"], as_index=False)
               .size()
               .rename(columns={"size": "count"})
        )

        fig_bar = px.bar(
            bar,
            x="amt_bin",
            y="count",
            color="Label",
            barmode="group"
        )
        fig_bar.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        fig_bar.update_xaxes(title="", showgrid=False)
        fig_bar.update_yaxes(title="", showgrid=True, gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Add data/creditcard.csv to enable amount distribution.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Right: Donut + Contributing signals
with colR:
    st.markdown("<div class='card'><h3>🧿 Fraud vs Legit Ratio</h3>", unsafe_allow_html=True)

    if not df_all.empty and "Class" in df_all.columns:
        fraud_n = int((df_all["Class"] == 1).sum())
        legit_n = int((df_all["Class"] == 0).sum())
        total_n = max(fraud_n + legit_n, 1)

        donut = go.Figure(data=[go.Pie(
            labels=["Legit", "Fraud"],
            values=[legit_n, fraud_n],
            hole=0.7
        )])
        donut.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend_orientation="h",
            legend_y=-0.15,
        )
        st.plotly_chart(donut, use_container_width=True)
        st.caption(f"Fraud rate: {fraud_n/total_n*100:.2f}% ({fraud_n:,} / {total_n:,})")
    else:
        st.info("Dataset missing Class column or not loaded.")

    st.markdown("<h3 style='margin-top:10px;'>📌 Contributing Signals</h3>", unsafe_allow_html=True)

    if st.session_state.payload:
        p = st.session_state.payload
        signals = {
            "Signal A (|V14|)": abs(float(p.get("V14", 0.0))),
            "Signal B (|V12|)": abs(float(p.get("V12", 0.0))),
            "Signal C (|V10|)": abs(float(p.get("V10", 0.0))),
        }
        sig_df = pd.DataFrame({"signal": list(signals.keys()), "value": list(signals.values())})
        fig_sig = px.bar(sig_df, x="value", y="signal", orientation="h")
        fig_sig.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_sig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", title="")
        fig_sig.update_yaxes(title="")
        st.plotly_chart(fig_sig, use_container_width=True)
        st.caption("UI-only signals derived from PCA features.")
    else:
        st.caption("Load a sample transaction to see contributing signals.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")


# -----------------------------
# Top row: API Status + Quick Actions
# -----------------------------
left, right = st.columns([1.25, 1])

with left:
    st.markdown("<div class='card'><h3>✅ API Status</h3>", unsafe_allow_html=True)

    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        if r.status_code == 200:
            st.markdown("<span class='badge badge-ok'>🛡️ API OK</span>", unsafe_allow_html=True)
            data = r.json()

            # Optional readable time display
            ts = data.get("timestamp_utc")
            if ts:
                dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                dt_local = dt_utc.astimezone()
                st.caption(
                    f"🕒 Health timestamp — UTC: {dt_utc.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"| Local: {dt_local.strftime('%Y-%m-%d %H:%M:%S')}"
                )

           
        else:
            st.markdown("<span class='badge badge-bad'>⚠️ API ERROR</span>", unsafe_allow_html=True)
            st.code(r.text)

    except Exception as e:
        st.markdown("<span class='badge badge-bad'>❌ API Unreachable</span>", unsafe_allow_html=True)
        st.write(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'><h3>⚡ Quick Actions</h3>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        if st.button("🔄 Load sample transaction", use_container_width=True):
            if DATA_PATH.exists():
                df = pd.read_csv(DATA_PATH)
                sample = df.sample(1).iloc[0].drop("Class", errors="ignore")
                st.session_state.payload = sample.to_dict()
                st.success("Loaded a random row from creditcard.csv")
            else:
                st.warning("data/creditcard.csv not found. Put the dataset in /data.")

    with colB:
        if st.button("🧹 Reset inputs", use_container_width=True):
            st.session_state.payload = None
            st.info("Inputs reset.")

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# MLflow Model Registry Info
# -----------------------------
st.markdown("<div class='card'><h3>🔬 MLflow Model Registry</h3>", unsafe_allow_html=True)

mlflow_col1, mlflow_col2 = st.columns([1, 1])

with mlflow_col1:
    try:
        # Query MLflow REST API for registered model info
        mlflow_api = f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get-latest-versions"
        r_ml = requests.post(mlflow_api, json={"name": "fraud-model"}, timeout=3)
        if r_ml.status_code == 200:
            versions = r_ml.json().get("model_versions", [])
            if versions:
                for mv in versions:
                    stage = mv.get("current_stage", mv.get("aliases", ["—"]))
                    ver = mv.get("version", "?")
                    status = mv.get("status", "?")
                    created = mv.get("creation_timestamp", 0)
                    run_id = mv.get("run_id", "—")

                    if isinstance(created, int) and created > 0:
                        dt = datetime.fromtimestamp(created / 1000)
                        created_str = dt.strftime("%Y-%m-%d %H:%M")
                    else:
                        created_str = "—"

                    st.markdown(
                        f"**Version {ver}** — Stage: `{stage}` — Status: `{status}`"
                    )
                    st.caption(f"Created: {created_str} | Run ID: `{run_id[:12]}…`")
            else:
                st.caption("No model versions registered yet.")
        else:
            st.caption(f"MLflow API returned {r_ml.status_code}")
    except Exception as e:
        st.caption(f"MLflow not reachable: {e}")

with mlflow_col2:
    st.markdown(f"[Open MLflow UI ↗]({MLFLOW_URL})")
    st.caption("Track experiments, compare runs, manage model stages.")

    # Show health load_source if available
    try:
        r_h = requests.get(HEALTH_ENDPOINT, timeout=3)
        if r_h.status_code == 200:
            data_h = r_h.json()
            load_src = data_h.get("load_source", "unknown")
            mv = data_h.get("model_version", "—")
            st.info(f"API model: **{mv}** (source: `{load_src}`)")
    except Exception:
        pass

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Input + Predict
# -----------------------------
c1, c2 = st.columns([1.3, 1])

with c1:
    st.markdown("<div class='card'><h3>🧾 Transaction</h3>", unsafe_allow_html=True)

    if st.session_state.payload:
        p = st.session_state.payload
        st.markdown("**Transaction Summary**")
        st.write(f"Amount: **${float(p.get('Amount', 0.0)):.2f}**")
        st.write(f"Time: **{p.get('Time', 0.0)}**")

        # UI-only signals
        signals = {
            "Signal A (|V14|)": abs(float(p.get("V14", 0.0))),
            "Signal B (|V12|)": abs(float(p.get("V12", 0.0))),
            "Signal C (|V10|)": abs(float(p.get("V10", 0.0))),
        }
        st.caption("Derived signals (UI-only)")
        st.markdown(
            f"""
            <div style="
              margin-top: 8px;
              border-radius: 16px;
              padding: 12px 14px;
              background: rgba(10,14,32,0.55);
              border: 1px solid rgba(65,240,255,0.16);
              font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
              color: rgba(240,244,255,0.92);
              white-space: pre-wrap;
            ">{pretty_json(signals)}</div>
            """,
            unsafe_allow_html=True,
        )

        st.caption("Advanced PCA features hidden for clarity.")
    else:
        st.info("Load a sample transaction to enable prediction.")

    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><h3>🚀 Prediction</h3>", unsafe_allow_html=True)

    if st.session_state.payload:
        if st.button("🚀 Predict", type="primary", use_container_width=True):
            try:
                resp = requests.post(PRED_ENDPOINT, json=st.session_state.payload, timeout=10)
                out = resp.json()

                proba = float(out.get("fraud_probability", 0.0))
                is_fraud = bool(out.get("is_fraud", False))

                if is_fraud:
                    st.markdown("<span class='badge badge-bad'>🚨 FRAUD SUSPECTED</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='badge badge-ok'>✅ LEGIT TRANSACTION</span>", unsafe_allow_html=True)

                st.metric("Fraud Probability", f"{proba:.4f}")
                st.caption(f"Model version: {out.get('model_version', 'unknown')}")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.caption("Load a transaction first.")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Footer imagery (optional)
# -----------------------------
if CARDS_BOTTOM or PAY_HAND:
    st.write("")
    st.markdown("<div class='card'><h3>🛰️ Visual Panel</h3>", unsafe_allow_html=True)
    cols = st.columns(2)
    if CARDS_BOTTOM:
        cols[0].image(f"data:image/png;base64,{CARDS_BOTTOM}", use_container_width=True)
    if PAY_HAND:
        cols[1].image(f"data:image/png;base64,{PAY_HAND}", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown(
    "<p class='small-muted'>© AI Fraud Detection — Streamlit + FastAPI + MLflow + Prometheus + Grafana</p>",
    unsafe_allow_html=True,
)
