import datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# Page Configuration & Modern Styling
# ==============================================================================
st.set_page_config(
    page_title="Smart Retail AI - 3D Demand & Inventory System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">📦 Next-Gen Retail AI: Demand Forecasting & 3D Inventory Ops</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Live Weather API Sync • 3D Spatial Visuals • Real-Time XGBoost Inference • Smart Audio Alerts</div>',
    unsafe_allow_html=True,
)

# ==============================================================================
# 1. API Fetching Function: Live Weather Integration (Open-Meteo API - Free/No Key)
# ==============================================================================
REGION_COORDINATES = {
    "North": {"lat": 28.6139, "lon": 77.2090, "city": "Delhi/North Region"},
    "South": {"lat": 12.9716, "lon": 77.5946, "city": "Bengaluru/South Region"},
    "East": {"lat": 22.5726, "lon": 88.3639, "city": "Kolkata/East Region"},
    "West": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai/West Region"},
}


def fetch_live_weather_api(region_name):
    """Open-Meteo Public API se live real-time weather fetch karta hai."""
    try:
        coords = REGION_COORDINATES.get(region_name, REGION_COORDINATES["North"])
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,relative_humidity_2m,precipitation,weather_code&timezone=auto"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json().get("current", {})
            temp = data.get("temperature_2m", 25.0)
            precip = data.get("precipitation", 0.0)
            w_code = data.get("weather_code", 0)

            # Map weather code to dataset condition
            if precip > 0.5 or w_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                weather_cond = "Rainy"
            elif w_code in [71, 73, 75, 77, 85, 86]:
                weather_cond = "Snowy"
            elif w_code in [1, 2, 3, 45, 48]:
                weather_cond = "Cloudy"
            else:
                weather_cond = "Sunny"

            return {
                "success": True,
                "city": coords["city"],
                "temperature": temp,
                "precipitation": precip,
                "condition": weather_cond,
            }
    except Exception as e:
        pass

    return {
        "success": False,
        "city": region_name,
        "temperature": 28.0,
        "precipitation": 0.0,
        "condition": "Sunny",
    }


# ==============================================================================
# 2. Audio Alert Generator (JavaScript Audio Synthesizer)
# ==============================================================================
def play_audio_alert(alert_type="success"):
    """Browser web audio API ke through alert sound play karta hai."""
    if alert_type == "danger":
        # Alert Beep (High-pitch oscillating warning tone)
        js_code = """
        <script>
            var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, audioCtx.currentTime);
            osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.4);
            gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.4);
        </script>
        """
    else:
        # Success Chime (Pleasant upward chime)
        js_code = """
        <script>
            var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            var osc = audioCtx.createOscillator();
            var gain = audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
            osc.frequency.setValueAtTime(659.25, audioCtx.currentTime + 0.15); // E5
            osc.frequency.setValueAtTime(783.99, audioCtx.currentTime + 0.3); // G5
            gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
            osc.connect(gain);
            gain.connect(audioCtx.destination);
            osc.start();
            osc.stop(audioCtx.currentTime + 0.5);
        </script>
        """
    st.components.v1.html(js_code, height=0)


# ==============================================================================
# 3. Model & Artifact Loader
# ==============================================================================
@st.cache_resource
def load_ml_assets():
    model = joblib.load("retail_demand_model.pkl")
    features = joblib.load("model_features.pkl")
    mappings = joblib.load("category_mappings.pkl")
    return model, features, mappings


try:
    model, model_features, mappings = load_ml_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(
        f"⚠️ Model files load nahi ho saki: {e}. Kripya check karein ki `.pkl` files project root mein hain."
    )

# ==============================================================================
# 4. Sidebar Controls
# ==============================================================================
st.sidebar.header("⚙️ Store & Product Configuration")

if assets_loaded:
    selected_region = st.sidebar.selectbox("🗺️ Store Region", mappings["regions"])

    # Fetch live weather data automatically based on region
    with st.sidebar:
        with st.spinner("Fetching Live Weather API..."):
            weather_data = fetch_live_weather_api(selected_region)

        if weather_data["success"]:
            st.success(
                f"📡 **Live API Sync ({weather_data['city']}):**\n\n"
                f"🌡️ {weather_data['temperature']} °C | 🌧️ {weather_data['precipitation']} mm ({weather_data['condition']})"
            )
        else:
            st.info("Weather API fallback mode active.")

    selected_store = st.sidebar.selectbox("🏬 Store ID", mappings["stores"])
    selected_category = st.sidebar.selectbox("🏷️ Category", mappings["categories"])
    selected_product = st.sidebar.selectbox("🔍 Product ID", mappings["products"])

    # Auto-selected or override weather
    selected_weather = st.sidebar.selectbox(
        "🌦️ Weather Condition",
        mappings["weather"],
        index=mappings["weather"].index(weather_data["condition"])
        if weather_data["condition"] in mappings["weather"]
        else 0,
    )
    selected_season = st.sidebar.selectbox(
        "🍂 Seasonality", mappings["seasonality"]
    )

    st.sidebar.subheader("💰 Pricing & Stock Parameters")
    price = st.sidebar.number_input(
        "Unit Price ($)", min_value=1.0, max_value=500.0, value=55.0
    )
    competitor_price = st.sidebar.number_input(
        "Competitor Price ($)", min_value=1.0, max_value=500.0, value=52.0
    )
    discount = st.sidebar.slider(
        "Discount Applied (%)", min_value=0, max_value=50, value=10
    )
    inventory = st.sidebar.number_input(
        "Current Shelf Inventory", min_value=0, max_value=2000, value=250
    )
    units_ordered = st.sidebar.number_input(
        "Pending Units Ordered", min_value=0, max_value=1000, value=90
    )
    demand_forecast = st.sidebar.number_input(
        "Baseline Forecast", min_value=0.0, max_value=1000.0, value=140.0
    )
    holiday = st.sidebar.radio(
        "🎉 Holiday / Promotion Active",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    pred_date = st.sidebar.date_input(
        "📅 Forecast Date", value=datetime.date.today()
    )

    # ==============================================================================
    # 5. Main Dashboard Tabs
    # ==============================================================================
    tab1, tab2, tab3 = st.tabs(
        [
            "🚀 AI Demand Forecasting",
            "🌐 3D Inventory Matrix",
            "📊 Analytics & Simulation",
        ]
    )

    # ---------------- TAB 1: Prediction & Live Alert ----------------
    with tab1:
        st.subheader("🎯 Real-Time Sales Inference")

        colA, colB = st.columns([2, 1])

        with colA:
            if st.button("⚡ Run AI Prediction", use_container_width=True):
                # Format feature vector
                input_data = {
                    "Inventory Level": [inventory],
                    "Units Ordered": [units_ordered],
                    "Demand Forecast": [demand_forecast],
                    "Price": [price],
                    "Discount": [discount],
                    "Holiday/Promotion": [holiday],
                    "Competitor Pricing": [competitor_price],
                    "Year": [pred_date.year],
                    "Month": [pred_date.month],
                    "Day": [pred_date.day],
                    "DayOfWeek": [pred_date.weekday()],
                    "IsWeekend": [1 if pred_date.weekday() >= 5 else 0],
                }

                input_df = pd.DataFrame(input_data)

                # Match One-Hot Columns
                for col in model_features:
                    if col not in input_df.columns:
                        input_df[col] = 0

                # Set activated categorical variables
                for active_col in [
                    f"Category_{selected_category}",
                    f"Region_{selected_region}",
                    f"Weather Condition_{selected_weather}",
                    f"Seasonality_{selected_season}",
                    f"Store ID_{selected_store}",
                    f"Product ID_{selected_product}",
                ]:
                    if active_col in input_df.columns:
                        input_df[active_col] = 1

                input_df = input_df[model_features]

                # Prediction
                raw_pred = model.predict(input_df)[0]
                pred_sales = max(0, int(round(raw_pred)))

                # Display Output Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("🔮 Expected Demand", f"{pred_sales} Units")
                m2.metric("📦 Available Stock", f"{inventory} Units")

                gap = inventory - pred_sales
                if gap < 0:
                    m3.metric(
                        "🚨 Deficit / Stockout Risk",
                        f"{abs(gap)} Units Short",
                        delta=f"{gap}",
                        delta_color="inverse",
                    )
                    st.error(
                        f"⚠️ **CRITICAL INVENTORY ALERT:** Forecasted sales ({pred_sales}) exceed current on-hand units ({inventory}). Reorder immediately!"
                    )
                    play_audio_alert("danger")
                else:
                    m3.metric(
                        "✅ Inventory Buffer",
                        f"{gap} Units Safe",
                        delta=f"{gap}",
                    )
                    st.success(
                        f"✨ **OPTIMAL STOCK:** Healthy inventory level to fulfill forecasted demand ({pred_sales} units)."
                    )
                    play_audio_alert("success")

                # Gauge Visualization
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=pred_sales,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={
                            "text": "Demand vs Capacity",
                            "font": {"size": 20},
                        },
                        delta={"reference": inventory, "increasing": {"color": "red"}},
                        gauge={
                            "axis": {
                                "range": [None, max(inventory * 1.5, pred_sales * 1.3)]
                            },
                            "bar": {"color": "#1E3A8A"},
                            "steps": [
                                {
                                    "range": [0, inventory],
                                    "color": "#D1FAE5",
                                }
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": inventory,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

        with colB:
            st.markdown("### 🏬 Live Environment Summary")
            st.info(
                f"""
                - **Target Store**: `{selected_store}` ({selected_region})
                - **Item**: `{selected_product}` | `{selected_category}`
                - **Live Temperature**: `{weather_data['temperature']} °C`
                - **Precipitation**: `{weather_data['precipitation']} mm`
                - **Competitor Index**: `${competitor_price} (vs ${price})`
                """
            )

    # ---------------- TAB 2: Interactive 3D Spatial Inventory ----------------
    with tab2:
        st.subheader("🌐 3D Multidimensional Inventory Space")
        st.markdown(
            "Rotate and zoom into the 3D surface plot mapping **Price ($)**, **Inventory Level**, and **Predicted Demand** across store categories."
        )

        # Generate sample multidimensional data for 3D rendering
        np.random.seed(42)
        categories_sample = (
            mappings["categories"] * 10
        )  # Groceries, Toys, Electronics, Furniture, Clothing
        sim_price = np.random.uniform(15, 95, len(categories_sample))
        sim_inventory = np.random.uniform(50, 480, len(categories_sample))
        sim_demand = (
            sim_inventory * 0.45
            + (100 - sim_price) * 0.8
            + np.random.normal(0, 15, len(categories_sample))
        )
        sim_demand = np.clip(sim_demand, 10, 500)

        df_3d = pd.DataFrame(
            {
                "Category": categories_sample,
                "Price": sim_price,
                "Inventory Level": sim_inventory,
                "Demand (Units)": sim_demand,
            }
        )

        fig_3d = px.scatter_3d(
            df_3d,
            x="Price",
            y="Inventory Level",
            z="Demand (Units)",
            color="Category",
            size="Demand (Units)",
            hover_data=["Category", "Price", "Inventory Level", "Demand (Units)"],
            opacity=0.85,
            title="3D Retail Cluster: Price vs Inventory vs Projected Demand",
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Price ($)",
                yaxis_title="Inventory Capacity",
                zaxis_title="Demand (Units Sold)",
            ),
            height=650,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---------------- TAB 3: Historical & Forecasting Analytics ----------------
    with tab3:
        st.subheader("📈 Price Sensitivity & Elasticity Simulation")

        test_prices = np.linspace(10, 100, 20)
        simulated_sales = []

        for p in test_prices:
            # Simple elasticity curve simulation
            sim_pred = (
                demand_forecast
                * (1 + discount / 100)
                * (competitor_price / (p + 1e-3)) ** 0.65
            )
            simulated_sales.append(sim_pred)

        fig_elasticity = px.line(
            x=test_prices,
            y=simulated_sales,
            labels={"x": "Price ($)", "y": "Projected Demand"},
            title="Simulated Demand Elasticity Curve",
        )
        fig_elasticity.add_vline(
            x=price, line_dash="dash", line_color="green", annotation_text="Selected Price"
        )
        fig_elasticity.update_layout(height=400)
        st.plotly_chart(fig_elasticity, use_container_width=True)

else:
    st.info(
        "Model setup incomplete. Ensure `retail_demand_model.pkl`, `model_features.pkl`, and `category_mappings.pkl` exist in the directory."
    )