
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load the model
model = joblib.load("model.pkl")

# App layout
st.set_page_config(page_title="Smart Lead Scoring Tool")
st.title("🔍 Smart Lead Scoring Tool")
st.write("Upload your leads.csv file to get lead quality predictions.")

# Preprocessing function (encoding logic)
def preprocess(df):
    title_map = {'CEO': 0, 'CTO': 1, 'Founder': 2, 'Marketing Director': 3,
                 'Sales Manager': 4, 'Intern': 5, 'Analyst': 6}
    industry_map = {'SaaS': 0, 'E-commerce': 1, 'Fintech': 2, 'Healthcare': 3,
                    'Real Estate': 4, 'Retail': 5, 'AI': 6}
    size_map = {'1-10': 0, '11-50': 1, '51-200': 2, '201-500': 3, '500+': 4}

    df["Title Encoded"] = df["Title"].map(title_map)
    df["Industry Encoded"] = df["Industry"].map(industry_map)
    df["Size Encoded"] = df["Company Size"].map(size_map)
    return df

# File uploader
uploaded_file = st.file_uploader("📁 Upload leads.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = preprocess(df)

    model_features = ["Title Encoded", "Industry Encoded", "Size Encoded", 
                      "Email Present", "LinkedIn Present", "Domain Score"]

    df["Lead Score"] = model.predict_proba(df[model_features])[:, 1] * 100
    df_sorted = df.sort_values("Lead Score", ascending=False)

    st.success("✅ Leads scored successfully!")
    st.dataframe(df_sorted[["Name", "Title", "Industry", "Domain Score", "Lead Score"]].head(10))

    # Download button
    csv = df_sorted.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Scored Leads", data=csv, file_name="scored_leads.csv", mime="text/csv")

    # SHAP explanation
    with st.expander("🔎 Show Feature Importance"):
        explainer = shap.Explainer(model, df[model_features])
        shap_values = explainer(df[model_features])
        shap_importance = np.abs(shap_values.values).mean(axis=0).ravel()

        feature_names = explainer.feature_names

        if len(feature_names) == len(shap_importance):
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': shap_importance
            }).sort_values(by='Importance', ascending=True)

            fig, ax = plt.subplots()
            ax.barh(shap_df['Feature'], shap_df['Importance'], color='steelblue')
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title("Feature Importance")
            st.pyplot(fig)
        else:
            st.error("⚠️ SHAP feature mismatch. Please verify input structure.")
