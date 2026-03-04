"""
dashboard/app.py
----------------
Interactive Streamlit dashboard for the Predictive Maintenance project.
Simulates a real PropertyOS-style equipment health monitoring interface.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropertyOS — Equipment Health Monitor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px; padding: 20px; text-align: center;
        border: 1px solid #3d4470;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #ffffff; }
    .metric-label { font-size: 0.85rem; color: #9ba3c2; margin-top: 4px; }
    .status-critical { color: #ff4b4b; font-weight: 700; }
    .status-warning  { color: #ffa500; font-weight: 700; }
    .status-healthy  { color: #00cc88; font-weight: 700; }
    .helix-header {
        background: linear-gradient(90deg, #1F3864, #2d5aa0);
        padding: 16px 24px; border-radius: 10px; margin-bottom: 20px;
        display: flex; align-items: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Generate data if not present
    if not os.path.exists(os.path.join(base, 'data/train_FD001.csv')):
        os.chdir(base)
        import sys; sys.path.insert(0, base)
        from data.generate_data import generate_cmapss_data
        generate_cmapss_data()

    train = pd.read_csv(os.path.join(base, 'data/train_FD001.csv'))

    # Compute RUL
    max_c = train.groupby('engine_id')['cycle'].max().rename('max_cycle')
    train = train.merge(max_c, on='engine_id')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train['will_fail_soon'] = (train['RUL'] <= 30).astype(int)

    # Failure risk score (simple degradation proxy)
    train['risk_score'] = (1 - train['RUL'] / train['max_cycle']) * 100

    return train


@st.cache_data
def load_report():
    try:
        with open('assets/model_report.json') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_importance():
    try:
        return pd.read_csv('assets/feature_importance.csv')
    except:
        return None


# ── Helper functions ──────────────────────────────────────────────────────────
def status_label(rul):
    if rul <= 30:   return "🔴 CRITICAL", "status-critical"
    if rul <= 60:   return "🟠 WARNING",  "status-warning"
    return "🟢 HEALTHY", "status-healthy"


def gauge_chart(value, title, max_val=100, threshold_warn=40, threshold_crit=70):
    color = "#ff4b4b" if value >= threshold_crit else "#ffa500" if value >= threshold_warn else "#00cc88"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'color': '#ffffff', 'size': 14}},
        number={'suffix': '%', 'font': {'color': '#ffffff', 'size': 24}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#9ba3c2'},
            'bar':  {'color': color, 'thickness': 0.25},
            'bgcolor': '#1e2130',
            'bordercolor': '#3d4470',
            'steps': [
                {'range': [0, threshold_warn],  'color': '#1a3a2a'},
                {'range': [threshold_warn, threshold_crit], 'color': '#3a2a10'},
                {'range': [threshold_crit, max_val], 'color': '#3a1515'},
            ],
            'threshold': {'line': {'color': '#ffffff', 'width': 2}, 'value': value}
        }
    ))
    fig.update_layout(
        paper_bgcolor='#1e2130', plot_bgcolor='#1e2130',
        font_color='#ffffff', height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    report = load_report()
    importance = load_importance()

    # Header
    st.markdown("""
    <div class="helix-header">
        <span style="font-size:1.8rem; margin-right:12px">🔧</span>
        <div>
            <div style="font-size:1.4rem; font-weight:700; color:white">PropertyOS — Equipment Health Monitor</div>
            <div style="font-size:0.85rem; color:#BDD7EE">Predictive Maintenance Dashboard  |  NASA CMAPSS Dataset  |  Built by Wasiq Bakhsh</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.image("https://via.placeholder.com/200x60/1F3864/FFFFFF?text=HelixIntel", use_column_width=True)
    st.sidebar.markdown("### Filters")

    all_engines = sorted(df['engine_id'].unique())
    selected_engine = st.sidebar.selectbox("Select Engine", all_engines, index=0)
    risk_threshold  = st.sidebar.slider("Risk Alert Threshold (%)", 0, 100, 60)
    show_all_sensors = st.sidebar.checkbox("Show all sensors", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About This Project")
    st.sidebar.markdown("""
    **Predictive Maintenance Model**
    - Dataset: NASA CMAPSS
    - Models: Random Forest, Gradient Boosting
    - Tasks: RUL Regression + Failure Classification
    - Built for: HelixIntel PropertyOS internship application
    """)

    # ── Top KPI cards ─────────────────────────────────────────────────────────
    last_readings = df.groupby('engine_id').last().reset_index()
    total     = len(last_readings)
    critical  = (last_readings['RUL'] <= 30).sum()
    warning   = ((last_readings['RUL'] > 30) & (last_readings['RUL'] <= 60)).sum()
    healthy   = (last_readings['RUL'] > 60).sum()
    avg_risk  = last_readings['risk_score'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total}</div><div class="metric-label">Total Engines</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff4b4b">{critical}</div><div class="metric-label">🔴 Critical</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ffa500">{warning}</div><div class="metric-label">🟠 Warning</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00cc88">{healthy}</div><div class="metric-label">🟢 Healthy</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_risk:.0f}%</div><div class="metric-label">Avg Fleet Risk</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Engine Detail", "🚨 Fleet Alert Board", "📊 Model Performance", "📈 Feature Importance"])

    # ── Tab 1: Engine Detail ──────────────────────────────────────────────────
    with tab1:
        engine_df = df[df['engine_id'] == selected_engine].sort_values('cycle')
        last_row  = engine_df.iloc[-1]
        rul       = int(last_row['RUL'])
        risk      = last_row['risk_score']
        status, css = status_label(rul)

        st.markdown(f"### Engine #{selected_engine} — <span class='{css}'>{status}</span>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.plotly_chart(gauge_chart(risk, "Risk Score"), use_container_width=True)
        with col2:
            st.metric("Remaining Useful Life", f"{rul} cycles")
            st.metric("Total Cycles Run",      f"{int(last_row['cycle'])}")
            st.metric("Max Life",              f"{int(last_row['max_cycle'])} cycles")
            st.metric("Failure Imminent",      "YES ⚠️" if rul <= 30 else "NO ✓")

        with col3:
            # Sensor degradation over time
            key_sensors = ['s2','s3','s4','s11','s14','s17'] if not show_all_sensors else \
                          ['s2','s3','s4','s7','s9','s11','s12','s14','s15','s17','s20','s21']

            fig = make_subplots(rows=2, cols=3, subplot_titles=key_sensors[:6],
                                shared_xaxes=False)
            colors = px.colors.qualitative.Set2
            for idx, sensor in enumerate(key_sensors[:6]):
                r, c = divmod(idx, 3)
                fig.add_trace(
                    go.Scatter(x=engine_df['cycle'], y=engine_df[sensor],
                               mode='lines', name=sensor,
                               line=dict(color=colors[idx], width=1.5),
                               showlegend=False),
                    row=r+1, col=c+1
                )
            fig.update_layout(
                height=320, paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
                font_color='#9ba3c2', title_text="Key Sensor Readings Over Time",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            fig.update_xaxes(gridcolor='#2d3250', zeroline=False)
            fig.update_yaxes(gridcolor='#2d3250', zeroline=False)
            st.plotly_chart(fig, use_container_width=True)

        # RUL trajectory
        st.markdown("#### Degradation Timeline")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=engine_df['cycle'], y=engine_df['RUL'],
            mode='lines+markers', name='Remaining Useful Life',
            line=dict(color='#2d9cdb', width=2),
            marker=dict(size=3)
        ))
        fig2.add_hrect(y0=0,  y1=30, fillcolor="#ff4b4b", opacity=0.15, line_width=0, annotation_text="⚠️ Critical Zone")
        fig2.add_hrect(y0=30, y1=60, fillcolor="#ffa500", opacity=0.10, line_width=0, annotation_text="Warning Zone")
        fig2.update_layout(
            paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
            font_color='#9ba3c2', height=250,
            xaxis_title="Cycle", yaxis_title="Remaining Useful Life (cycles)",
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Fleet Alert Board ───────────────────────────────────────────────
    with tab2:
        st.markdown("### Fleet Alert Board — All Engines Ranked by Risk")

        fleet = last_readings[['engine_id','cycle','RUL','risk_score']].copy()
        fleet['Status'] = fleet['RUL'].apply(lambda x:
            '🔴 CRITICAL' if x <= 30 else '🟠 WARNING' if x <= 60 else '🟢 Healthy')
        fleet['Risk %']   = fleet['risk_score'].round(1)
        fleet['Action']   = fleet['RUL'].apply(lambda x:
            'Immediate maintenance required' if x <= 30 else
            'Schedule maintenance this week' if x <= 60 else 'Monitor normally')
        fleet = fleet.sort_values('risk_score', ascending=False)
        fleet.columns = ['Engine','Cycles Run','RUL (cycles)','Risk Score','Status','Risk %','Recommended Action']

        # Color rows
        def color_row(row):
            if '🔴' in row['Status']: return ['background-color: #2a1515'] * len(row)
            if '🟠' in row['Status']: return ['background-color: #2a1e10'] * len(row)
            return [''] * len(row)

        risk_filtered = fleet[fleet['Risk %'] >= (100 - risk_threshold)]
        st.dataframe(fleet.style.apply(color_row, axis=1), use_container_width=True, height=400)

        # Risk distribution
        fig3 = px.histogram(fleet, x='Risk %', nbins=20,
                             color_discrete_sequence=['#2d9cdb'],
                             title='Fleet Risk Score Distribution')
        fig3.add_vline(x=60, line_dash="dash", line_color="#ffa500", annotation_text="Warning")
        fig3.add_vline(x=70, line_dash="dash", line_color="#ff4b4b", annotation_text="Critical")
        fig3.update_layout(
            paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
            font_color='#9ba3c2', height=280,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3: Model Performance ───────────────────────────────────────────────
    with tab3:
        st.markdown("### Model Performance — Trained on NASA CMAPSS Dataset")

        if report:
            reg = report['regression']['results']
            clf = report['classification']['results']

            st.markdown("#### RUL Regression (Remaining Useful Life Prediction)")
            reg_df = pd.DataFrame([
                {'Model': m, 'RMSE': v['RMSE'], 'MAE': v['MAE'], 'R²': v['R2']}
                for m, v in reg.items()
            ])
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(reg_df, use_container_width=True, hide_index=True)
            with col2:
                fig4 = px.bar(reg_df, x='Model', y='RMSE', color='RMSE',
                              color_continuous_scale='RdYlGn_r',
                              title='RMSE by Model (lower is better)')
                fig4.update_layout(paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
                                   font_color='#9ba3c2', height=250,
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig4, use_container_width=True)

            st.markdown("#### Failure Classification (Will fail within 30 cycles?)")
            clf_df = pd.DataFrame([
                {'Model': m, 'Accuracy': v['Accuracy'], 'Precision': v['Precision'],
                 'Recall': v['Recall'], 'F1': v['F1'], 'AUC': v['AUC']}
                for m, v in clf.items()
            ])
            col3, col4 = st.columns([1, 2])
            with col3:
                st.dataframe(clf_df, use_container_width=True, hide_index=True)
            with col4:
                clf_melt = clf_df.melt(id_vars='Model', value_vars=['Precision','Recall','F1','AUC'])
                fig5 = px.bar(clf_melt, x='variable', y='value', color='Model',
                              barmode='group', title='Classification Metrics by Model',
                              color_discrete_sequence=px.colors.qualitative.Set2)
                fig5.update_layout(paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
                                   font_color='#9ba3c2', height=280,
                                   margin=dict(l=10, r=10, t=40, b=10),
                                   yaxis=dict(range=[0,1]))
                st.plotly_chart(fig5, use_container_width=True)

            # Actual vs Predicted scatter
            best_reg = report['regression']['best_model']
            actuals  = report['regression']['results'][best_reg]['actuals']
            preds    = report['regression']['results'][best_reg]['predictions']
            fig6 = px.scatter(x=actuals, y=preds, opacity=0.5,
                              labels={'x':'Actual RUL','y':'Predicted RUL'},
                              title=f'Actual vs Predicted RUL — {best_reg}',
                              color_discrete_sequence=['#2d9cdb'])
            fig6.add_shape(type='line', x0=0, y0=0, x1=max(actuals), y1=max(actuals),
                           line=dict(color='#ff4b4b', dash='dash'))
            fig6.update_layout(paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
                               font_color='#9ba3c2', height=300,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("Run train_models.py first to see model performance metrics here.")

    # ── Tab 4: Feature Importance ──────────────────────────────────────────────
    with tab4:
        st.markdown("### Feature Importance — What Predicts Equipment Failure?")
        if importance is not None:
            top20 = importance.head(20).sort_values('importance')
            fig7  = px.bar(top20, x='importance', y='feature', orientation='h',
                           color='importance', color_continuous_scale='Blues',
                           title='Top 20 Most Predictive Features (Random Forest)')
            fig7.update_layout(
                paper_bgcolor='#1e2130', plot_bgcolor='#0e1117',
                font_color='#9ba3c2', height=500,
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig7, use_container_width=True)

            st.markdown("""
            **How to interpret this:**
            - Features with **_mean_10** or **_mean_20** are rolling averages — they capture the *trend* of degradation
            - Raw sensor readings (s14, s9, s3) show current state
            - `cycle_norm` shows how far through the expected life the engine is
            - This mirrors how HelixIntel's PropertyOS identifies early warning signals before failure
            """)
        else:
            st.info("Run train_models.py first to generate feature importance data.")


if __name__ == "__main__":
    main()
