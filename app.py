
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load("model.pkl")

# App layout
st.set_page_config(page_title="Smart Lead Scoring Tool")
st.title("🔍 Smart Lead Scoring Tool")
st.write("Upload your leads.csv file to get lead quality predictions.")

# Preprocessing function with lowercased mapping
def preprocess(df):
    for col in ["Title", "Industry", "Company Size"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    title_map = {
        'ceo': 0, 'cto': 1, 'founder': 2, 'marketing director': 3,
        'sales manager': 4, 'intern': 5, 'analyst': 6
    }
    industry_map = {
        'saas': 0, 'e-commerce': 1, 'fintech': 2, 'healthcare': 3,
        'real estate': 4, 'retail': 5, 'ai': 6
    }
    size_map = {
        '1-10': 0, '11-50': 1, '51-200': 2, '201-500': 3, '500+': 4
    }

    df["Title Encoded"] = df["Title"].map(title_map)
    df["Industry Encoded"] = df["Industry"].map(industry_map)
    df["Size Encoded"] = df["Company Size"].map(size_map)
    return df

# File uploader
uploaded_file = st.file_uploader("📁 Upload leads.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["Name", "Title", "Industry", "Company Size",
                     "Email Present", "LinkedIn Present", "Domain Score"]
    df = df[[col for col in df.columns if col in required_cols]]

    df = preprocess(df)
    df.fillna(-1, inplace=True)

    model_features = ["Title Encoded", "Industry Encoded", "Size Encoded",
                      "Email Present", "LinkedIn Present", "Domain Score"]

    df["Lead Score"] = model.predict_proba(df[model_features])[:, 1] * 100
    df_sorted = df.sort_values("Lead Score", ascending=False)

    st.success("✅ Leads scored successfully!")
    st.dataframe(df_sorted[["Name", "Title", "Industry", "Domain Score", "Lead Score"]].head(10))

    csv = df_sorted.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Scored Leads", data=csv, file_name="scored_leads.csv", mime="text/csv")
