import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import gdown

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MODIS Fire Type Classifier | AI Decision Support",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS (Dark Theme + Ember/Flame Visual Accents) ───────────────────────
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0b0f19;
        color: #f3f4f6;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1f1610 0%, #2a1208 50%, #151b28 100%);
        border: 1px solid rgba(255, 87, 34, 0.25);
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6b35, #f7931e, #ff4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        line-height: 1.5;
        max-width: 900px;
    }

    /* Card Panels */
    .feature-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        border-color: rgba(255, 107, 53, 0.4);
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f97316;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Result Containers */
    .verdict-box {
        border-radius: 16px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    .verdict-veg {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(15, 23, 42, 0.95));
        border: 2px solid #ef4444;
    }
    .verdict-static {
        background: linear-gradient(135deg, rgba(234, 88, 12, 0.15), rgba(15, 23, 42, 0.95));
        border: 2px solid #f97316;
    }
    .verdict-offshore {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(15, 23, 42, 0.95));
        border: 2px solid #0284c7;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.8rem !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 15px rgba(234, 88, 12, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(234, 88, 12, 0.5) !important;
    }

    /* Metric Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-orange { background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.35); }
    .badge-blue { background: rgba(14, 165, 233, 0.18); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.35); }
    .badge-green { background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); }

    /* Footer Disclaimer */
    .disclaimer-box {
        background-color: #0f172a;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 2.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Artifacts Loading with Cloud Auto-Download ────────────────────────────────
MODEL_PATH = "best_fire_detection_model.pkl"
SCALER_PATH = "scaler.pkl"
GDRIVE_FILE_ID = "1ib7_s0Swzr-U0Cc7lr1dJ9f37ZDkD3gs"

@st.cache_resource(show_spinner=False)
def load_ml_pipeline():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📦 Fetching Random Forest model weights from secure storage (first run only)..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Missing 'scaler.pkl' in root directory.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_ml_pipeline()
    pipeline_loaded = True
except Exception as e:
    pipeline_loaded = False
    pipeline_error = str(e)

# ─── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🔥 MODIS Satellite Fire Type Classifier</div>
    <div class="hero-subtitle">
        AI-powered decision support system that classifies thermal anomalies across India into 
        <b>Vegetation Fires</b>, <b>Static Industrial Land Sources</b>, and <b>Offshore Anomalies</b> 
        using NASA Terra & Aqua satellite sensor telemetry.
    </div>
    <div style="margin-top: 1rem;">
        <span class="badge-pill badge-orange">NASA FIRMS Data (2021–2023)</span>
        <span class="badge-pill badge-green">Random Forest: 97.8% Accuracy</span>
        <span class="badge-pill badge-blue">SMOTE Resampled</span>
        <span class="badge-pill badge-orange">40% Verification Time Reduction</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Preset Scenarios for Rapid Demonstrations ─────────────────────────────────
PRESETS = {
    "Select a preset scenario...": None,
    "🌲 Severe Forest Wildfire (Simlipal / Uttarakhand)": {
        "brightness": 348.5,
        "bright_t31": 302.4,
        "track": 1.2,
        "frp": 58.6,
        "scan": 1.4,
        "confidence": "high"
    },
    "🌾 Agricultural Crop Stubble Burning (Punjab / Haryana)": {
        "brightness": 322.0,
        "bright_t31": 298.5,
        "track": 1.1,
        "frp": 16.4,
        "scan": 1.1,
        "confidence": "nominal"
    },
    "🌆 Industrial Gas Flare / Petrochemical Refinery (Bokaro / Jamnagar)": {
        "brightness": 306.9,
        "bright_t31": 295.2,
        "track": 1.2,
        "frp": 9.3,
        "scan": 1.6,
        "confidence": "low"
    },
    "🌊 Offshore Oil Rig / Coastal Gas Extraction (Mumbai High Offshore)": {
        "brightness": 331.4,
        "bright_t31": 294.0,
        "track": 1.0,
        "frp": 24.5,
        "scan": 1.0,
        "confidence": "high"
    }
}

# ─── Navigation Tabs ───────────────────────────────────────────────────────────
tab_predict, tab_benchmark, tab_features, tab_about = st.tabs([
    "🛰️ Real-Time Classifier",
    "📊 Model Performance & Metrics",
    "🔍 Satellite Feature Dictionary",
    "ℹ️ Project Architecture & NASA FIRMS"
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: Real-Time Classifier
# ───────────────────────────────────────────────────────────────────────────────
with tab_predict:
    if not pipeline_loaded:
        st.error(f"❌ Failed to load model pipeline: {pipeline_error}")
        st.stop()

    # Preset selector bar
    col_preset, col_reset = st.columns([4, 1])
    with col_preset:
        selected_preset = st.selectbox(
            "⚡ Quick Scenario Loader (Populate with real-world satellite events):",
            options=list(PRESETS.keys()),
            index=0
        )
    with col_reset:
        st.write("") # spacing
        st.write("")
        if st.button("🔄 Reset Inputs"):
            st.session_state.clear()
            st.rerun()

    # Preset values handler
    default_vals = PRESETS.get(selected_preset, None) or {
        "brightness": 315.0,
        "bright_t31": 295.0,
        "track": 1.1,
        "frp": 14.5,
        "scan": 1.2,
        "confidence": "nominal"
    }

    # Main Input Form
    with st.form("fire_classifier_form"):
        col_thermal, col_geom = st.columns(2)

        with col_thermal:
            st.markdown("""
            <div class="feature-card">
                <div class="card-title">🌡️ Thermal & Radiative Measurements</div>
            """, unsafe_allow_html=True)
            
            in_brightness = st.number_input(
                "🌞 Brightness Temperature (Band 21/22)",
                min_value=250.0,
                max_value=505.0,
                value=float(default_vals["brightness"]),
                step=0.5,
                help="Brightness temperature of the fire pixel in Kelvin (K). High values signify active, intense combustion."
            )
            st.caption(f"Equivalent: **{in_brightness - 273.15:.1f} °C**")

            in_bright_t31 = st.number_input(
                "🌡️ Background Surface Temperature (Band 31)",
                min_value=220.0,
                max_value=400.0,
                value=float(default_vals["bright_t31"]),
                step=0.5,
                help="Background ground surface temperature in Kelvin (K). Helps isolate true thermal anomaly from hot desert soil."
            )
            st.caption(f"Equivalent: **{in_bright_t31 - 273.15:.1f} °C**")

            in_frp = st.number_input(
                "🔥 Fire Radiative Power - FRP (MW)",
                min_value=0.0,
                max_value=6000.0,
                value=float(default_vals["frp"]),
                step=0.5,
                help="Radiative heat energy released in Megawatts. Higher FRP correlates with higher biomass consumption."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_geom:
            st.markdown("""
            <div class="feature-card">
                <div class="card-title">🛰️ Satellite Geometry & Reliability</div>
            """, unsafe_allow_html=True)

            in_scan = st.number_input(
                "🧪 Pixel Scan Dimension (km)",
                min_value=1.0,
                max_value=4.8,
                value=float(default_vals["scan"]),
                step=0.1,
                help="Spatial pixel dimension along the satellite scan angle. (1.0 km at nadir up to 4.8 km at scan edge)."
            )

            in_track = st.number_input(
                "📍 Pixel Track Dimension (km)",
                min_value=1.0,
                max_value=2.0,
                value=float(default_vals["track"]),
                step=0.1,
                help="Spatial pixel dimension along the satellite flight path orbit (1.0 to 2.0 km)."
            )

            conf_options = ["low", "nominal", "high"]
            conf_default_idx = conf_options.index(default_vals["confidence"]) if default_vals["confidence"] in conf_options else 1
            in_confidence = st.selectbox(
                "📊 Detection Confidence Rating",
                options=conf_options,
                index=conf_default_idx,
                help="NASA algorithmic quality score assessing cloud proximity, sun glint, and signal-to-noise ratio."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Primary Action Trigger
        btn_predict = st.form_submit_button("🚀 Run Classification Analysis", use_container_width=True)

    # ─── Inference Execution ──────────────────────────────────────────────────
    if btn_predict:
        with st.spinner("Evaluating multi-spectral satellite telemetry through Random Forest ensemble..."):
            confidence_map = {"low": 0, "nominal": 1, "high": 2}
            conf_encoded = confidence_map[in_confidence]

            # Model expects exact 6 features: [brightness, bright_t31, track, frp, scan, confidence]
            raw_vector = np.array([[in_brightness, in_bright_t31, in_track, in_frp, in_scan, conf_encoded]])
            
            # Validation Check
            if in_brightness < in_bright_t31:
                st.warning("⚠️ **Thermal Anomaly Notice:** Fire brightness temperature (Band 21/22) is lower than background surface temperature (Band 31). This may represent an inverted or cold atmospheric pixel.")

            # Scaling & Prediction
            scaled_vector = scaler.transform(raw_vector)
            prediction = int(model.predict(scaled_vector)[0])
            
            # Extract Genuine Class Probabilities if available
            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                probabilities = model.predict_proba(scaled_vector)[0]
                classes = list(model.classes_)
                prob_dict = {cls: prob for cls, prob in zip(classes, probabilities)}
            else:
                prob_dict = {}

        # ─── Displaying Results ───────────────────────────────────────────────
        st.markdown("### 🎯 Classification Verdict & Triage Intelligence")

        if prediction == 0:
            verdict_class = "verdict-veg"
            icon = "🌲🔥"
            title = "VEGETATION FIRE DETECTED"
            subtext = "High-confidence open biomass combustion: Active Wildfire, Forest Fire, or Crop Residue Burning."
            action_protocol = "Alert District Forest Officer (DFO) and deploy local ranger surveillance or satellite burn perimeter tracker."
        elif prediction == 2:
            verdict_class = "verdict-static"
            icon = "🌆🔥"
            title = "OTHER STATIC LAND SOURCE DETECTED"
            subtext = "Persistent stationary thermal anomaly: Industrial Refinery, Gas Flaring, Steel Plant, or Brick Kiln."
            action_protocol = "No immediate wildfire response needed. Cross-reference with state industrial pollution emission logs."
        elif prediction == 3:
            verdict_class = "verdict-offshore"
            icon = "🌊🔥"
            title = "OFFSHORE THERMAL EVENT DETECTED"
            subtext = "Maritime thermal anomaly: Offshore Oil Rig, Gas Drilling Platform, or Marine Vessel."
            action_protocol = "Notify Indian Coast Guard / Maritime Safety Cell. Ensure offshore platform compliance."
        else:
            verdict_class = "verdict-static"
            icon = "⚠️"
            title = "UNRECOGNIZED THERMAL SOURCE"
            subtext = "Coordinates outside trained distribution parameters."
            action_protocol = "Manual inspection recommended."

        pred_confidence = prob_dict.get(prediction, 1.0) * 100

        st.markdown(f"""
        <div class="verdict-box {verdict_class}">
            <div style="font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem;">
                {icon} {title}
            </div>
            <div style="font-size: 1.05rem; color: #e5e7eb; margin-bottom: 0.8rem;">
                {subtext}
            </div>
            <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 0.8rem 1rem; border-left: 4px solid currentColor;">
                <b>Recommended Field Action:</b> {action_protocol}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─── Probability Distribution Section ─────────────────────────────────
        if has_proba:
            st.markdown("#### 📊 Model Probability Distribution Across Fire Classes")
            col_p1, col_p2, col_p3 = st.columns(3)
            
            p_veg = prob_dict.get(0, 0.0) * 100
            p_static = prob_dict.get(2, 0.0) * 100
            p_offshore = prob_dict.get(3, 0.0) * 100

            with col_p1:
                st.metric("🌲 Vegetation Fire", f"{p_veg:.1f}%")
                st.progress(float(p_veg) / 100.0)
            with col_p2:
                st.metric("🌆 Static Industrial", f"{p_static:.1f}%")
                st.progress(float(p_static) / 100.0)
            with col_p3:
                st.metric("🌊 Offshore Anomaly", f"{p_offshore:.1f}%")
                st.progress(float(p_offshore) / 100.0)

        # ─── Input Telemetry Inspector ────────────────────────────────────────
        with st.expander("📋 Review Submitted Satellite Telemetry Payload", expanded=False):
            df_review = pd.DataFrame([{
                "Band 21/22 Brightness": f"{in_brightness:.1f} K ({in_brightness - 273.15:.1f} °C)",
                "Band 31 Ground Temp": f"{in_bright_t31:.1f} K ({in_bright_t31 - 273.15:.1f} °C)",
                "Temperature Delta (ΔT)": f"{in_brightness - in_bright_t31:.1f} K",
                "Fire Radiative Power": f"{in_frp:.1f} MW",
                "Scan Dimension": f"{in_scan:.2f} km",
                "Track Dimension": f"{in_track:.2f} km",
                "Detection Confidence": in_confidence.capitalize()
            }])
            st.dataframe(df_review, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: Model Performance & Metrics
# ───────────────────────────────────────────────────────────────────────────────
with tab_benchmark:
    st.markdown("### 📈 Model Evaluation & Empirical Benchmarks")
    st.write("Performance evaluation on 137,131 independent test observations across 3 years of NASA MODIS data over India:")

    # Top Metric Tiles
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Final Model Accuracy", "97.79%", "+4.8% vs Decision Tree")
    with m2:
        st.metric("Vegetation F1-Score", "0.97", "Precision: 0.97 | Recall: 0.96")
    with m3:
        st.metric("Static Land F1-Score", "0.97", "Precision: 0.96 | Recall: 0.97")
    with m4:
        st.metric("Offshore F1-Score", "1.00", "Precision: 1.00 | Recall: 1.00")

    st.markdown("#### 🔬 Algorithm Comparison Benchmark")
    benchmark_data = pd.DataFrame({
        "Algorithm": ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Decision Tree", "Random Forest (Final 🏆)"],
        "Test Accuracy (%)": [58.58, 93.11, 94.96, 97.79],
        "Handling of Non-Linearity": ["Poor (Linear Hyperplane)", "Good (Distance-Based)", "High (Prone to Variance)", "Optimal (Bagged Ensemble)"],
        "Inference Speed": ["< 1 ms", "~ 28 ms", "< 1 ms", "~ 3 ms"]
    })
    st.dataframe(benchmark_data, use_container_width=True)

    st.markdown("""
    #### ⚖️ Class Imbalance Handling via SMOTE
    - **Raw Imbalance:** Class 0 had **257,625 samples**, Class 2 had **13,550 samples**, and Class 3 had only **42 samples**.
    - **Strategy:** Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the feature space to synthetically interpolate minority classes up to **182,841 samples per class** ($N = 548,523$).
    - **Result:** Prevented the model from collapsing into majority-class bias while achieving 100% precision on rare offshore platform incidents.
    """)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3: Feature Dictionary & Physics
# ───────────────────────────────────────────────────────────────────────────────
with tab_features:
    st.markdown("### 🔬 Multi-Spectral Thermal Satellite Physics")
    
    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.markdown("""
        **1. Band 21/22 Brightness ($T_{21}$ in Kelvin)**
        - Mid-Infrared channel centered at **3.96 µm**.
        - Highly sensitive to high-temperature emissions ($> 600\text{ K}$) characteristic of flaming and smoldering fires.
        
        **2. Band 31 Ground Temperature ($T_{31}$ in Kelvin)**
        - Thermal Infrared channel centered at **11.0 µm**.
        - Measures background land surface and cloud-top ambient temperature.
        - The temperature difference $\Delta T = T_{21} - T_{31}$ is the primary indicator of an active combustion hotspot.
        
        **3. Fire Radiative Power (FRP in Megawatts)**
        - Quantifies radiant heat output governed by the Stefan-Boltzmann law ($E \propto T^4$).
        - Industrial flares emit concentrated point-source FRP, whereas wildfires exhibit distributed, high-variance FRP.
        """)
    with col_feat2:
        st.markdown("""
        **4. Pixel Scan Angle (`scan` in km)**
        - Spatial dimension across the sensor sweep.
        - At nadir (directly below satellite), pixel size is $1.0\text{ km} \times 1.0\text{ km}$.
        - At off-nadir scan edges ($> 50^\circ$), distortion expands the ground footprint up to $4.8\text{ km}$.
        
        **5. Satellite Flight Track (`track` in km)**
        - Spatial dimension along the satellite orbital trajectory ($1.0\text{ km}$ to $2.0\text{ km}$).
        
        **6. Detection Confidence (`confidence`)**
        - Categorical quality heuristic computed from background cloud proximity, water bodies, and solar glint angles.
        """)

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4: Project Architecture & NASA FIRMS
# ───────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### ℹ️ Project Context & Real-World Utility")
    
    st.markdown("""
    #### 🎯 The Objective
    State forest agencies and disaster management cells in India receive automated fire alerts via satellite feeds. 
    However, alerts frequently flag **benign or expected industrial heat**—such as steel factories, brick kilns, and refinery gas flares—triggering costly field inspections and **alert fatigue**.
    
    This project demonstrates an **automated ML triage engine** that filters raw NASA thermal feeds in real time, determining whether a heat anomaly warrants forestry deployment or represents commercial static infrastructure.

    #### 🛰️ Satellite Constellation
    - **Terra Satellite (EOS AM-1):** Sun-synchronous morning overpasses (~10:30 AM local time).
    - **Aqua Satellite (EOS PM-1):** Sun-synchronous afternoon overpasses (~1:30 PM local time).
    - Both satellites provide **2 to 4 daily sweeps** across the Indian subcontinent.

    #### 👨‍💻 Developer & Attribution
    - **Developer:** T Venkata Ram Charan Reddy
    - **Program:** AICTE Internship Machine Learning Project
    - **Dataset:** NASA FIRMS (Fire Information for Resource Management System) MODIS Collection 6.1
    """)

# ─── Disclaimer Banner ─────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <b>Operational Decision-Support Disclaimer:</b> 
    This application is an academic Machine Learning proof-of-concept developed using historical NASA MODIS observation data (2021–2023). 
    It is designed as an analytical triage layer to demonstrate ML capability and should be used to augment, not supersede, official satellite monitoring portals (e.g., FSI Van Agni, ISRO Bhuvan) or ground authority verification protocols.
</div>
""", unsafe_allow_html=True)
