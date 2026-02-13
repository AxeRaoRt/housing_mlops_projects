# streamlit/streamlit_app.py
import os
import time
import json
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
import pydeck as pdk

# ----------------------------
# Page config + light styling
# ----------------------------
st.set_page_config(
    page_title="California Housing — MLOps Demo",
    page_icon="🏠",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")

st.markdown(
    """
    <style>
      .small-muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
      .kpi { padding: 14px; border-radius: 14px; border: 1px solid rgba(0,0,0,0.08); }
      .kpi h3 { margin: 0; }
      .kpi p { margin: 6px 0 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Header
# ----------------------------
left, right = st.columns([3, 1])
with left:
    st.title("🏠 California Housing — MLOps Prediction Demo")
    st.write(
        "This Streamlit UI calls the **FastAPI `/predict`** endpoint. "
        "Metrics go to **Prometheus/Grafana**."
    )
with right:
    try:
        r = requests.get(HEALTH_URL, timeout=2)
        if r.status_code == 200:
            st.success("API: UP ✅")
        else:
            st.warning(f"API: {r.status_code}")
    except Exception:
        st.error("API: DOWN ❌")

st.divider()

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Inputs")

preset = st.sidebar.selectbox(
    "Preset scenarios",
    ["Custom", "Urban (LA-like)", "Coastal (SF-like)", "Inland (Central Valley-like)"],
)

def preset_values(name: str):
    if name == "Urban (LA-like)":
        return dict(MedInc=3.2, HouseAge=25, AveRooms=5.1, AveBedrms=1.0, Population=1200, AveOccup=3.0, Latitude=34.05, Longitude=-118.25)
    if name == "Coastal (SF-like)":
        return dict(MedInc=6.5, HouseAge=35, AveRooms=5.5, AveBedrms=1.1, Population=900, AveOccup=2.4, Latitude=37.77, Longitude=-122.42)
    if name == "Inland (Central Valley-like)":
        return dict(MedInc=2.4, HouseAge=18, AveRooms=4.9, AveBedrms=1.0, Population=2000, AveOccup=3.6, Latitude=36.74, Longitude=-119.78)
    return None

vals = preset_values(preset)

MedInc = st.sidebar.slider("MedInc (Median Income)", 0.5, 15.0, float(vals["MedInc"]) if vals else 3.2, 0.1)
HouseAge = st.sidebar.slider("HouseAge", 1.0, 60.0, float(vals["HouseAge"]) if vals else 25.0, 1.0)

AveRooms = st.sidebar.number_input("AveRooms", min_value=1.0, max_value=50.0, value=float(vals["AveRooms"]) if vals else 5.1, step=0.1)
AveBedrms = st.sidebar.number_input("AveBedrms", min_value=0.5, max_value=10.0, value=float(vals["AveBedrms"]) if vals else 1.0, step=0.1)

Population = st.sidebar.number_input("Population", min_value=1.0, max_value=50000.0, value=float(vals["Population"]) if vals else 1200.0, step=50.0)
AveOccup = st.sidebar.number_input("AveOccup", min_value=0.5, max_value=20.0, value=float(vals["AveOccup"]) if vals else 3.0, step=0.1)

Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, float(vals["Latitude"]) if vals else 34.05, 0.01)
Longitude = st.sidebar.slider("Longitude", -124.0, -114.0, float(vals["Longitude"]) if vals else -118.25, 0.01)

payload = {
    "MedInc": float(MedInc),
    "HouseAge": float(HouseAge),
    "AveRooms": float(AveRooms),
    "AveBedrms": float(AveBedrms),
    "Population": float(Population),
    "AveOccup": float(AveOccup),
    "Latitude": float(Latitude),
    "Longitude": float(Longitude),
}

st.sidebar.subheader("Load test")
rps = st.sidebar.slider("Requests per click", 1, 25, 1)
timeout_s = st.sidebar.slider("Timeout (s)", 1, 20, 10)

# ----------------------------
# Main layout
# ----------------------------
colA, colB = st.columns([1.15, 1.0])

with colA:
    st.markdown("### Input preview")
    st.code(json.dumps(payload, indent=2), language="json")

    st.markdown("### Location (map)")

    # Map without Mapbox token using Carto basemap (usually fixes blank map)
    df_map = pd.DataFrame([{"lat": payload["Latitude"], "lon": payload["Longitude"]}])

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position="[lon, lat]",
        get_radius=2500,
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(
        latitude=payload["Latitude"],
        longitude=payload["Longitude"],
        zoom=6,
    )

    deck = pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "Lat: {lat}\nLon: {lon}"},
    )

    st.pydeck_chart(deck)

with colB:
    st.markdown("### Prediction")

    btn1, btn2 = st.columns(2)
    do_predict = btn1.button("🚀 Predict", use_container_width=True)
    clear_hist = btn2.button("🧹 Clear history", use_container_width=True)

    if clear_hist:
        st.session_state.history = []
        st.success("History cleared.")

    if do_predict:
        preds = []
        latencies = []
        errors = 0

        for _ in range(rps):
            t0 = time.time()
            try:
                resp = requests.post(API_URL, json=payload, timeout=timeout_s)
                latency = time.time() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    preds.append(float(data.get("prediction")))
                    latencies.append(float(data.get("latency_seconds", latency)))
                else:
                    errors += 1
            except Exception:
                errors += 1

        if preds:
            pred_value = float(sum(preds) / len(preds))
            latency_value = float(sum(latencies) / len(latencies)) if latencies else None

            st.markdown(
                f"""
                <div class="kpi">
                  <h3>Predicted value</h3>
                  <p style="font-size: 1.6rem;"><b>{pred_value:.4f}</b></p>
                  <p class="small-muted">Average over {len(preds)} request(s)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if latency_value is not None:
                st.markdown(
                    f"""
                    <div class="kpi" style="margin-top: 10px;">
                      <h3>Latency</h3>
                      <p style="font-size: 1.4rem;"><b>{latency_value:.3f}s</b></p>
                      <p class="small-muted">Client-side mean latency</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.session_state.history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "prediction": pred_value,
                    "latency_s": latency_value if latency_value is not None else None,
                    **payload,
                }
            )
        else:
            st.error("No successful prediction. Check API_URL and whether FastAPI is running.")
            if errors:
                st.caption(f"Errors: {errors}/{rps}")

    st.markdown("### History")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        k1, k2, k3 = st.columns(3)
        k1.metric("Requests stored", len(hist_df))
        k2.metric("Last prediction", f"{hist_df.iloc[-1]['prediction']:.4f}")
        if hist_df["latency_s"].notna().any():
            k3.metric("Last latency (s)", f"{float(hist_df.iloc[-1]['latency_s']):.3f}")
        else:
            k3.metric("Last latency (s)", "—")

        st.line_chart(hist_df.set_index("time")[["prediction"]])
        if hist_df["latency_s"].notna().any():
            st.line_chart(hist_df.set_index("time")[["latency_s"]])

        with st.expander("Show history table"):
            st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("No history yet. Click **Predict** to add entries.")
