# ============================================================
# SMART QUALITY CONTROL DASHBOARD (Streamlit)
# ============================================================
# Author: [Your Name]
# Project: Smart Quality Analytics System
# Dataset: manufacturing-quality-control-dataset.csv
# ============================================================

# ============================================================
# SECTION 1: Import Required Libraries
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# SECTION 2: Load and Prepare the Dataset
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("manufacturing-quality-control-dataset.csv")
    df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce')
    df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    df['rework_required'] = df['rework_required'].astype(str).str.strip().str.lower()
    return df

df = load_data()


# ============================================================
# SECTION 3: Sidebar Filters
# ============================================================
st.sidebar.title("🔍 Filter Options")

factory = st.sidebar.multiselect("Factory Location", df['factory_location'].unique())
shift = st.sidebar.multiselect("Shift", df['shift'].unique())
inspector = st.sidebar.multiselect("Inspector ID", df['inspector_id'].unique())

filtered_df = df.copy()
if factory:
    filtered_df = filtered_df[filtered_df['factory_location'].isin(factory)]
if shift:
    filtered_df = filtered_df[filtered_df['shift'].isin(shift)]
if inspector:
    filtered_df = filtered_df[filtered_df['inspector_id'].isin(inspector)]


# ============================================================
# SECTION 4: Create Tabs for Each Dashboard
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "🏭 Factory Overview",
    "⚙️ Quality & Maintenance",
    "🧠 Root Cause Analysis"
])


# ============================================================
# SECTION 5: Tab 1 - Factory Overview
# ============================================================
with tab1:
    st.header("🏭 Factory Overview Dashboard")

    # --- KPIs ---
    total_defects = int(filtered_df['defect_count'].sum())
    avg_quality = round(filtered_df['quality_score'].mean(), 2)
    pct_rework = round((filtered_df['rework_required'].eq('yes').sum() /
                        filtered_df['batch_id'].nunique()) * 100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Defects", f"{total_defects:,}")
    col2.metric("Average Quality Score", avg_quality)
    col3.metric("% Batches Requiring Rework", f"{pct_rework}%")

    # --- Chart 1: Defects Over Time ---
    trend = filtered_df.groupby('production_date', as_index=False)['defect_count'].sum()
    fig_trend = px.line(trend, x='production_date', y='defect_count',
                        title="Defects Over Time", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Chart 2: Defects by Shift ---
    fig_shift = px.bar(filtered_df.groupby('shift', as_index=False)['defect_count'].sum(),
                       x='shift', y='defect_count', color='defect_count',
                       title="Defects by Shift", color_continuous_scale='Reds')
    st.plotly_chart(fig_shift, use_container_width=True)

    # --- Chart 3: Quality by Factory Location ---
    fig_factory = px.bar(filtered_df.groupby('factory_location', as_index=False)['quality_score'].mean(),
                         x='factory_location', y='quality_score',
                         title="Average Quality Score by Factory Location",
                         color='quality_score', color_continuous_scale='Blues')
    st.plotly_chart(fig_factory, use_container_width=True)


# ============================================================
# SECTION 6: Tab 2 - Quality & Maintenance
# ============================================================
with tab2:
    st.header("⚙️ Quality & Maintenance Analysis")

    # --- KPIs ---
    avg_defects_per_batch = round(filtered_df['defect_count'].mean(), 2)
    est_downtime_pct = round((filtered_df['defect_count'].sum() * 5) /
                             (filtered_df['batch_id'].nunique() * 60) * 100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Defects per Batch", avg_defects_per_batch)
    col2.metric("% Batches Requiring Rework", f"{pct_rework}%")
    col3.metric("Estimated Downtime %", f"{est_downtime_pct}%")

    # --- Chart 1: Defect Severity by Type ---
    fig_severity = px.box(filtered_df, x='defect_type', y='defect_severity',
                          color='factory_location', title="Defect Severity by Type & Location")
    st.plotly_chart(fig_severity, use_container_width=True)

    # --- Chart 2: Rework Trend Over Time ---
    rework_trend = (filtered_df.groupby('inspection_date')['rework_required']
                    .apply(lambda x: (x == 'yes').sum()).reset_index())
    fig_rework = px.line(rework_trend, x='inspection_date', y='rework_required',
                         title="Rework Trend Over Time", markers=True)
    st.plotly_chart(fig_rework, use_container_width=True)

    # --- Chart 3: Quality vs Defect Severity ---
    fig_scatter = px.scatter(filtered_df, x='quality_score', y='defect_severity',
                             color='factory_location', title="Quality vs Defect Severity")
    st.plotly_chart(fig_scatter, use_container_width=True)


# ============================================================
# SECTION 7: Tab 3 - Root Cause Analysis
# ============================================================
with tab3:
    st.header("🧠 Root Cause Analysis")

    # --- Chart 1: Defect Type Distribution ---
    fig_type = px.pie(filtered_df, names='defect_type', values='defect_count',
                      title="Defect Type Distribution")
    st.plotly_chart(fig_type, use_container_width=True)

    # --- Chart 2: Defects by Shift and Inspector ---
    heatmap = filtered_df.groupby(['shift', 'inspector_id'], as_index=False)['defect_count'].sum()
    fig_heat = px.density_heatmap(heatmap, x='shift', y='inspector_id', z='defect_count',
                                  title="Defects by Shift and Inspector",
                                  color_continuous_scale='Reds')
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Chart 3: Defect Severity Over Time ---
    severity_time = filtered_df.groupby('inspection_date', as_index=False)['defect_severity'].mean()
    fig_severity_time = px.line(severity_time, x='inspection_date', y='defect_severity',
                                title="Average Defect Severity Over Time", markers=True)
    st.plotly_chart(fig_severity_time, use_container_width=True)


# ============================================================
# SECTION 8: Footer / Info
# ============================================================
st.markdown("---")
st.caption("© 2025 Smart Quality Control Dashboard | Built with Streamlit + Plotly")


#

# ========================================
# 📘 Model 3: Quality Score Prediction
# ========================================


# ================================================
# 📊 Feature Importance (after training the model)
# ================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from Random Forest
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display top features in console
print("\n🔥 Top 10 Most Important Features Driving Quality:")
print(feature_importances.head(10))

# Save feature importance for Tableau
feature_importances.to_csv("quality_feature_importance_for_tableau.csv", index=False)
print("\n📂 File saved: quality_feature_importance_for_tableau.csv")

# Optional: Visualize feature importance directly in Python
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importances.head(10), x='Importance', y='Feature')
plt.title("Top 10 Factors Affecting Quality Score")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
# ================================