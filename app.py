import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

st.set_page_config(
    page_title="ExtraaLearn Lead Scorer",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    return model, feature_cols

try:
    model, feature_cols = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ExtraaLearn")
    st.caption("Lead Conversion Scorer")
    st.markdown("---")

    st.header("How to use")
    st.markdown("""
1. Upload a CSV of leads
2. Results appear automatically
3. Explore the Data Insights tab
4. Download the scored file
    """)

    st.markdown("---")
    st.markdown("**Required columns**")
    st.code(
        "age\ncurrent_occupation\nfirst_interaction\nprofile_completed\n"
        "website_visits\ntime_spent_on_website\npage_views_per_visit\n"
        "last_activity\nprint_media_type1\nprint_media_type2\n"
        "digital_media\neducational_channels\nreferral",
        language=None
    )

    st.markdown("---")
    st.markdown("**Priority tiers**")
    st.markdown("🔴 **High** — ≥ 60% probability")
    st.markdown("🟡 **Medium** — 40–59% probability")
    st.markdown("🟢 **Low** — < 40% probability")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🎓 ExtraaLearn Lead Conversion Scorer")
st.markdown(
    "Upload a CSV of leads to predict which ones are most likely to convert to paid customers. "
    "Results are ranked by conversion probability so your sales team knows exactly where to focus."
)

if not model_loaded:
    st.error(
        "**Model files not found.** Run the last cell in the Jupyter notebook "
        "to generate `model.pkl` and `feature_columns.pkl`, then restart this app."
    )
    st.stop()

# ── File upload + scoring (shared across all tabs) ────────────────────────────
uploaded = st.file_uploader("Upload leads CSV", type=["csv"])

df = None
results = None

if uploaded is not None:
    df = pd.read_csv(uploaded)

    with st.spinner("Scoring leads..."):
        proc = df.copy()
        if "ID" in proc.columns:
            proc.drop("ID", axis=1, inplace=True)
        if "status" in proc.columns:
            proc.drop("status", axis=1, inplace=True)

        encoded = pd.get_dummies(proc, drop_first=True)
        encoded = encoded.reindex(columns=feature_cols, fill_value=0)
        probs = model.predict_proba(encoded)[:, 1]

    results = df.copy()
    results["Conversion Probability (%)"] = (probs * 100).round(1)
    results["Priority"] = pd.cut(
        probs,
        bins=[0, 0.4, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    results["Priority"] = results["Priority"].astype(str)
    results = results.sort_values("Conversion Probability (%)", ascending=False).reset_index(drop=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_score, tab_data, tab_model = st.tabs(["📊 Score Leads", "📈 Data Insights", "🔍 Model Insights"])

PRIORITY_COLORS = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#28a745"}

# ── Tab 1: Score Leads ────────────────────────────────────────────────────────
with tab_score:
    if results is None:
        st.info("Upload a CSV file above to get started. You can use `ExtraaLearn.csv` as a sample.")
    else:
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head(5), use_container_width=True)
        st.caption(f"{len(df):,} leads loaded")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Leads", f"{len(results):,}")
        c2.metric("🔴 High Priority", int((results["Priority"] == "High").sum()))
        c3.metric("🟡 Medium Priority", int((results["Priority"] == "Medium").sum()))
        c4.metric("🟢 Low Priority", int((results["Priority"] == "Low").sum()))

        st.markdown("---")
        table_tab, chart_tab = st.tabs(["Ranked Table", "Charts"])

        with table_tab:
            def highlight_priority(val):
                mapping = {
                    "High":   "background-color: #f8d7da; color: #721c24",
                    "Medium": "background-color: #fff3cd; color: #856404",
                    "Low":    "background-color: #d4edda; color: #155724",
                }
                return mapping.get(val, "")

            styled = results.style.map(highlight_priority, subset=["Priority"])
            st.dataframe(styled, use_container_width=True, height=500)

            csv_bytes = results.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Scored Leads",
                data=csv_bytes,
                file_name="scored_leads.csv",
                mime="text/csv",
            )

        with chart_tab:
            col_left, col_right = st.columns(2)

            with col_left:
                fig_hist = px.histogram(
                    results,
                    x="Conversion Probability (%)",
                    color="Priority",
                    color_discrete_map=PRIORITY_COLORS,
                    nbins=20,
                    title="Conversion Probability Distribution",
                )
                fig_hist.update_layout(bargap=0.05, legend_title_text="Priority")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_right:
                priority_counts = results["Priority"].value_counts().reset_index()
                priority_counts.columns = ["Priority", "Count"]
                fig_pie = px.pie(
                    priority_counts,
                    names="Priority",
                    values="Count",
                    color="Priority",
                    color_discrete_map=PRIORITY_COLORS,
                    title="Lead Priority Breakdown",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

# ── Tab 2: Data Insights ──────────────────────────────────────────────────────
with tab_data:
    if results is None:
        st.info("Upload a CSV file above to see data insights.")
    else:
        # ── Section 1: Lead Profile Summary ───────────────────────────────────
        st.subheader("Lead Profile Summary")
        st.markdown("A breakdown of who is in your uploaded dataset.")

        col1, col2, col3 = st.columns(3)

        with col1:
            occ_counts = results["current_occupation"].value_counts().reset_index()
            occ_counts.columns = ["Occupation", "Count"]
            fig = px.bar(
                occ_counts,
                x="Occupation",
                y="Count",
                title="Leads by Occupation",
                color="Occupation",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text="Count",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            interaction_counts = results["first_interaction"].value_counts().reset_index()
            interaction_counts.columns = ["Channel", "Count"]
            fig = px.pie(
                interaction_counts,
                names="Channel",
                values="Count",
                title="First Interaction Channel",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            profile_order = ["Low", "Medium", "High"]
            profile_counts = (
                results["profile_completed"]
                .value_counts()
                .reindex(profile_order, fill_value=0)
                .reset_index()
            )
            profile_counts.columns = ["Completion", "Count"]
            fig = px.bar(
                profile_counts,
                x="Completion",
                y="Count",
                title="Profile Completion Level",
                color="Completion",
                color_discrete_sequence=["#f8d7da", "#fff3cd", "#d4edda"],
                text="Count",
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, xaxis_title="",
                              xaxis={"categoryorder": "array", "categoryarray": profile_order})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Section 2: High vs Low Priority Comparison ────────────────────────
        st.subheader("High vs Low Priority: What's Different?")
        st.markdown("Side-by-side comparison of the key traits that separate your best leads from the rest.")

        high_low = results[results["Priority"].isin(["High", "Low"])].copy()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.box(
                high_low,
                x="Priority",
                y="time_spent_on_website",
                color="Priority",
                color_discrete_map=PRIORITY_COLORS,
                title="Time Spent on Website",
                points="outliers",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Seconds", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.box(
                high_low,
                x="Priority",
                y="age",
                color="Priority",
                color_discrete_map=PRIORITY_COLORS,
                title="Age Distribution",
                points="outliers",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Age", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            occ_priority = (
                high_low.groupby(["current_occupation", "Priority"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                occ_priority,
                x="current_occupation",
                y="Count",
                color="Priority",
                barmode="group",
                color_discrete_map=PRIORITY_COLORS,
                title="Occupation Mix: High vs Low Priority",
            )
            fig.update_layout(xaxis_title="", legend_title_text="Priority")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            profile_priority = (
                high_low.groupby(["profile_completed", "Priority"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                profile_priority,
                x="profile_completed",
                y="Count",
                color="Priority",
                barmode="group",
                color_discrete_map=PRIORITY_COLORS,
                title="Profile Completion: High vs Low Priority",
                category_orders={"profile_completed": ["Low", "Medium", "High"]},
            )
            fig.update_layout(xaxis_title="", legend_title_text="Priority")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── Section 3: Score Distribution by Segment ──────────────────────────
        st.subheader("Average Conversion Probability by Segment")
        st.markdown("Which groups in your data convert at the highest rate on average?")

        col1, col2 = st.columns(2)

        with col1:
            avg_occ = (
                results.groupby("current_occupation")["Conversion Probability (%)"]
                .mean()
                .round(1)
                .reset_index()
                .sort_values("Conversion Probability (%)", ascending=False)
            )
            fig = px.bar(
                avg_occ,
                x="current_occupation",
                y="Conversion Probability (%)",
                title="By Occupation",
                color="Conversion Probability (%)",
                color_continuous_scale="RdYlGn",
                text="Conversion Probability (%)",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_interaction = (
                results.groupby("first_interaction")["Conversion Probability (%)"]
                .mean()
                .round(1)
                .reset_index()
                .sort_values("Conversion Probability (%)", ascending=False)
            )
            fig = px.bar(
                avg_interaction,
                x="first_interaction",
                y="Conversion Probability (%)",
                title="By First Interaction Channel",
                color="Conversion Probability (%)",
                color_continuous_scale="RdYlGn",
                text="Conversion Probability (%)",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            avg_profile = (
                results.groupby("profile_completed")["Conversion Probability (%)"]
                .mean()
                .round(1)
                .reset_index()
            )
            fig = px.bar(
                avg_profile,
                x="profile_completed",
                y="Conversion Probability (%)",
                title="By Profile Completion",
                color="Conversion Probability (%)",
                color_continuous_scale="RdYlGn",
                text="Conversion Probability (%)",
                category_orders={"profile_completed": ["Low", "Medium", "High"]},
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            avg_activity = (
                results.groupby("last_activity")["Conversion Probability (%)"]
                .mean()
                .round(1)
                .reset_index()
                .sort_values("Conversion Probability (%)", ascending=False)
            )
            fig = px.bar(
                avg_activity,
                x="last_activity",
                y="Conversion Probability (%)",
                title="By Last Activity",
                color="Conversion Probability (%)",
                color_continuous_scale="RdYlGn",
                text="Conversion Probability (%)",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(xaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Model Insights ─────────────────────────────────────────────────────
with tab_model:
    st.subheader("Top Features Driving Lead Conversion")
    st.markdown(
        "The chart below shows which variables the model relies on most "
        "when predicting whether a lead will convert."
    )

    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(10)
    )

    fig_feat = px.bar(
        feat_df.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 10 Feature Importances — Tuned Random Forest",
        color="Importance",
        color_continuous_scale="Purples",
    )
    fig_feat.update_layout(yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(fig_feat, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Takeaways")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**What drives conversion**")
        st.markdown("""
- **Time spent on website** is the single strongest predictor — highly engaged leads convert far more often.
- **First interaction via Website** (vs Mobile App) is a strong positive signal.
- **High profile completion** indicates deeper intent and correlates with conversion.
        """)

    with col2:
        st.markdown("**How to use this in practice**")
        st.markdown("""
- Focus sales calls on **High priority** leads first.
- For **Medium** leads, a targeted email nudge may be enough to push them over the line.
- **Low priority** leads can go into a longer-term nurture sequence.
        """)
