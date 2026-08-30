Set-Content -Path README.md -Value @'
<div align="center">

# 📦 Smart Retail AI: 3D Demand & Inventory Forecasting
### *End-to-End Time Series ML Pipeline • Live Weather API • 3D Spatial Analytics • Smart Audio Alerts*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://retail-demand-forecasting-ml-gsoypcyccegh9bzgu4vh7x.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-XGBoost%20%7C%20Tree--Ensemble-brightgreen.svg)](https://xgboost.readthedocs.io/)

[🌐 View Live Web App](https://retail-demand-forecasting-ml-gsoypcyccegh9bzgu4vh7x.streamlit.app/) • [📂 GitHub Repository](https://github.com/Anuska111/retail-demand-forecasting-ml)

</div>

### 🖥️ Dashboard Architecture & Tab Modules

<table>
  <thead>
    <tr>
      <th align="left">Module Tab</th>
      <th align="left">Core Functionality</th>
      <th align="left">Primary Tech / Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>🚀 AI Demand Forecasting</b></td>
      <td>Real-time demand inference with interactive capacity gauge & live stockout warnings</td>
      <td><code>XGBoost</code> / <code>Scikit-Learn</code>, Audio API, Gauge</td>
    </tr>
    <tr>
      <td><b>🌐 3D Inventory Matrix</b></td>
      <td>3D multidimensional cluster plots across categories, price elasticities, and store limits</td>
      <td>Plotly Express 3D Scatter Engine</td>
    </tr>
    <tr>
      <td><b>📊 Analytics & Simulation</b></td>
      <td>Price sensitivity simulations, competitor pricing index, and historical trends</td>
      <td>Dynamic Elasticity Curve Models</td>
    </tr>
  </tbody>
</table>

### 🛠️ Technology Stack

<table>
  <thead>
    <tr>
      <th align="left">Domain</th>
      <th align="left">Technologies / Tools</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>🐍 Languages</b></td>
      <td><code>Python 3.10+</code></td>
    </tr>
    <tr>
      <td><b>🧠 ML & Analytics</b></td>
      <td><code>XGBoost</code>, <code>Scikit-Learn</code>, <code>Statsmodels</code>, <code>Pandas</code>, <code>NumPy</code>, <code>Joblib</code></td>
    </tr>
    <tr>
      <td><b>🌐 APIs & Networking</b></td>
      <td><code>Open-Meteo REST API</code>, <code>Requests</code></td>
    </tr>
    <tr>
      <td><b>🎨 Frontend & UI</b></td>
      <td><code>Streamlit</code>, <code>Plotly (3D & Indicators)</code>, <code>HTML5/CSS3</code>, <code>Web Audio API</code></td>
    </tr>
    <tr>
      <td><b>⚙️ Environment & DevOps</b></td>
      <td><code>PyCharm</code>, <code>Git</code>, <code>GitHub</code>, <code>Streamlit Cloud</code></td>
    </tr>
  </tbody>
</table> 
 ### 📂 Repository File Tree

<table>
  <thead>
    <tr>
      <th align="left">File / Resource</th>
      <th align="left">Type</th>
      <th align="left">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>app.py</code></td>
      <td>Application Core</td>
      <td>Main full-stack Streamlit dashboard, data processing, and UI pipeline</td>
    </tr>
    <tr>
      <td><code>retail_store_inventory.csv</code></td>
      <td>Data Asset</td>
      <td>Historical transactional records (73,100 rows × 15 operational features)</td>
    </tr>
    <tr>
      <td><code>retail_demand_model.pkl</code></td>
      <td>Trained Weights</td>
      <td>Serialized Gradient Boosting / XGBoost regression inference model</td>
    </tr>
    <tr>
      <td><code>model_features.pkl</code></td>
      <td>Metadata</td>
      <td>Encoded feature column indices for one-hot vector alignment</td>
    </tr>
    <tr>
      <td><code>category_mappings.pkl</code></td>
      <td>Metadata</td>
      <td>Serialized store, region, and category mapping dictionaries</td>
    </tr>
    <tr>
      <td><code>requirements.txt</code></td>
      <td>Configuration</td>
      <td>Python dependencies specification file for production deployment</td>
    </tr>
    <tr>
      <td><code>README.md</code></td>
      <td>Documentation</td>
      <td>Project overview, system architecture flowchart, and setup guide</td>
    </tr>
  </tbody>
</table>
---

### 🔄 System Architecture & Data Pipeline

```mermaid
flowchart TD
    subgraph Data_Sources ["1. Input & Real-Time Sync"]
        A[("Historical Sales CSV")]
        B["Live Open-Meteo REST API"]
        C["User Inputs (Price, Discount)"]
    end

    subgraph Preprocessing ["2. Feature Pipeline"]
        D["Time-Series Feature Extraction"]
        E["One-Hot Categorical Encoding"]
    end

    subgraph ML_Engine ["3. ML Inference Engine"]
        F["Trained XGBoost Regressor"]
        G["Predicted Demand (Units)"]
    end

    subgraph UI_Modules ["4. Streamlit Dashboard"]
        H["🚀 Real-Time Demand Metrics"]
        I["🌐 Interactive 3D Spatial Matrix"]
        J["📈 Price Elasticity Simulation"]
    end

    subgraph Alert_System ["5. Stockout Decision & Alerts"]
        K{"Demand > Shelf Inventory?"}
        L["🚨 Deficit Alert & Sound"]
        M["✅ Optimal Stock Level"]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    G --> J
    G --> K
    K -- "Yes" --> L
    K -- "No" --> M

