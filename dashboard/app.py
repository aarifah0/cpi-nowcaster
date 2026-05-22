"""
CPI Nowcaster Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="CPI Nowcaster | Real-Time Inflation",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- GLOBAL CSS ----
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">

<style>
/* ---- RESET & BASE ---- */
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: #000000;
    background-image: 
        radial-gradient(ellipse at 20% 20%, rgba(0, 255, 170, 0.03) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(107, 47, 179, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(20, 241, 149, 0.02) 0%, transparent 70%);
}

/* ---- TYPOGRAPHY ---- */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #FFFFFF !important;
    letter-spacing: -0.02em !important;
}

p, span, div, li, a {
    font-family: 'Inter', sans-serif !important;
}

/* ---- GLASS CARD ---- */
.glass-card {
    background: rgba(18, 18, 24, 0.7);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    border-color: rgba(0, 255, 170, 0.25);
    box-shadow: 0 0 30px rgba(0, 255, 170, 0.05), 0 8px 32px rgba(0, 0, 0, 0.4);
    transform: translateY(-2px);
}

/* ---- NAVBAR ---- */
.navbar {
    background: rgba(10, 10, 10, 0.8);
    backdrop-filter: blur(30px);
    -webkit-backdrop-filter: blur(30px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding: 14px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -16px -16px 24px -16px;
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.nav-orb {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #00FFAA;
    box-shadow: 0 0 20px rgba(0, 255, 170, 0.5), 0 0 60px rgba(0, 255, 170, 0.2);
}

.nav-title {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 1.1rem;
    color: #FFFFFF;
    letter-spacing: -0.01em;
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    color: #9CA3AF;
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00FFAA;
    box-shadow: 0 0 8px rgba(0, 255, 170, 0.6);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ---- HERO METRIC ---- */
.hero-metric {
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 5rem;
    color: #00FFAA;
    line-height: 1;
    letter-spacing: -0.04em;
    text-shadow: 0 0 80px rgba(0, 255, 170, 0.3);
}

.hero-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #6B7280;
    font-weight: 600;
}

/* ---- STAT CARD ---- */
.stat-value {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #00FFAA;
    line-height: 1;
}

.stat-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6B7280;
    font-weight: 600;
    margin-top: 4px;
}

/* ---- DATA ROW ---- */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    transition: background 0.2s;
}

.data-row:hover {
    background: rgba(0, 255, 170, 0.03);
    border-radius: 8px;
    padding: 12px 12px;
}

.data-label { color: #9CA3AF; font-size: 0.85rem; }
.data-value { color: #FFFFFF; font-weight: 600; font-size: 0.85rem; }
.data-highlight { color: #00FFAA; font-weight: 700; }

/* ---- BUTTON ---- */
.btn-ghost {
    border: 1px solid rgba(0, 255, 170, 0.3);
    color: #00FFAA;
    background: transparent;
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.3s;
}

.btn-ghost:hover {
    background: #00FFAA;
    color: #000000;
    box-shadow: 0 0 30px rgba(0, 255, 170, 0.3);
}

/* ---- SECTION TITLE ---- */
.section-title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    margin-bottom: 4px;
}

.section-subtitle {
    font-size: 0.8rem;
    color: #6B7280;
}

/* ---- CHART AREA ---- */
.chart-container {
    background: rgba(18, 18, 24, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
}

/* ---- FEATURE ROW ---- */
.feature-icon {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    margin-bottom: 12px;
}

.feature-icon-green { background: rgba(0, 255, 170, 0.1); color: #00FFAA; border: 1px solid rgba(0, 255, 170, 0.2); }
.feature-icon-purple { background: rgba(107, 47, 179, 0.1); color: #8B5CF6; border: 1px solid rgba(107, 47, 179, 0.2); }
.feature-icon-blue { background: rgba(30, 64, 175, 0.1); color: #60A5FA; border: 1px solid rgba(30, 64, 175, 0.2); }

/* ---- FOOTER ---- */
.footer {
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    padding-top: 24px;
    margin-top: 48px;
}

/* ---- OVERRIDES ---- */
.stPlotlyChart { border-radius: 12px; }
div[data-testid="stVerticalBlock"] { gap: 0; }
</style>
""", unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data(ttl=60)
def load_nowcast():
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "nowcast.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

nowcast = load_nowcast()

# ---- NAVBAR ----
st.markdown(f"""
<div class="navbar">
    <div class="nav-logo">
        <div class="nav-orb"></div>
        <span class="nav-title">CPI Nowcaster</span>
    </div>
    <div class="nav-status">
        <div class="status-dot"></div>
        <span>Live • {nowcast['date'] if nowcast else 'No data'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- MAIN CONTENT ----
if nowcast:
    # ---- HERO SECTION ----
    delta_val = nowcast['nowcast'] - nowcast.get('latest_actual', 0) if nowcast.get('latest_actual') else 0
    delta_str = f"+{delta_val:.2f}%" if delta_val >= 0 else f"{delta_val:.2f}%"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 40px 0 48px 0;">
        <p class="hero-label">Current CPI Inflation Nowcast</p>
        <p class="hero-metric">{nowcast['nowcast']:.2f}%</p>
        <p style="color: {'#00FFAA' if delta_val >= 0 else '#EF4444'}; font-weight: 600; font-size: 1rem; margin-top: 8px;">
            {delta_str} vs latest official
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- STATS GRID ----
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p class="stat-value">{nowcast.get('latest_actual', 0):.2f}%</p>
            <p class="stat-label">Latest Official CPI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p class="stat-value">{nowcast['rmse_historical']:.2f}%</p>
            <p class="stat-label">Model RMSE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p class="stat-value">{nowcast['data_available']['training_samples']}</p>
            <p class="stat-label">Training Samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <p class="stat-value">{nowcast['data_available']['monthly_series']}+{nowcast['data_available']['daily_series']}</p>
            <p class="stat-label">Data Series</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

    # ---- CHART + FEATURES ----
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Feature Importance</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Top drivers of the current nowcast</p>', unsafe_allow_html=True)
        
        if nowcast.get('top_features'):
            features_df = pd.DataFrame(nowcast['top_features'])
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=features_df['feature'][::-1],
                x=features_df['importance'][::-1],
                orientation='h',
                marker=dict(
                    color=features_df['importance'][::-1],
                    colorscale=[[0, 'rgba(0,255,170,0.2)'], [1, '#00FFAA']],
                    line_width=0,
                ),
                text=[f"{v:.1%}" for v in features_df['importance'][::-1]],
                textposition='outside',
                textfont=dict(color='#9CA3AF', size=11),
                hovertemplate='%{y}: %{x:.1%}<extra></extra>'
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#9CA3AF',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, color='#FFFFFF'),
                margin=dict(l=0, r=40, t=0, b=0),
                height=240,
                bargap=0.4,
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Data Pipeline</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Current snapshot</p>', unsafe_allow_html=True)
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        pipeline_data = [
            ("Date", nowcast['date'], False),
            ("Monthly Series", str(nowcast['data_available']['monthly_series']), True),
            ("Daily Series", str(nowcast['data_available']['daily_series']), True),
            ("Model", "MIDAS + XGBoost", False),
            ("Validation", "Expanding Window CV", False),
            ("Automation", "GitHub Actions (daily)", False),
            ("Deployment", "Streamlit Cloud", False),
        ]
        
        for label, value, highlight in pipeline_data:
            val_class = "data-highlight" if highlight else "data-value"
            st.markdown(f"""
            <div class="data-row">
                <span class="data-label">{label}</span>
                <span class="{val_class}">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

    # ---- METHODOLOGY ----
    st.markdown('<p class="section-title" style="margin-bottom: 16px;">How It Works</p>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div class="feature-icon feature-icon-green">◈</div>
            <h4 style="color: #FFFFFF; font-weight: 700; margin: 0 0 8px 0;">Data Pipeline</h4>
            <p style="color: #9CA3AF; font-size: 0.82rem; line-height: 1.6; margin: 0;">
            Fetches 5 monthly macro indicators and 3 daily financial series from FRED, 
            respecting actual publication release calendars.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div class="feature-icon feature-icon-purple">◇</div>
            <h4 style="color: #FFFFFF; font-weight: 700; margin: 0 0 8px 0;">Ragged Edge Engine</h4>
            <p style="color: #9CA3AF; font-size: 0.82rem; line-height: 1.6; margin: 0;">
            Aligns mixed-frequency data without future-peeking. Handles publication lags 
            and creates real-time feature snapshots.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown("""
        <div class="glass-card" style="height: 100%;">
            <div class="feature-icon feature-icon-blue">◎</div>
            <h4 style="color: #FFFFFF; font-weight: 700; margin: 0 0 8px 0;">MIDAS + XGBoost</h4>
            <p style="color: #9CA3AF; font-size: 0.82rem; line-height: 1.6; margin: 0;">
            MIDAS compresses daily data into monthly signals. XGBoost captures non-linear 
            economic patterns. Validated with time-series-safe expanding window CV.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---- FOOTER ----
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #4B5563; font-size: 0.75rem;">Powered by FRED • GitHub Actions • Streamlit Cloud</span>
            <span style="color: #374151; font-size: 0.7rem;">© 2026 CPI Nowcaster</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("⚠️ No nowcast data found. Run `python src/nowcast.py` to generate predictions.")
