# 🚀 Smart Lead Scoring – Internship Project

This project was completed as part of a test submission for the **Machine Learning Engineer Internship at Caprae Capital**.

## 🔍 Objective

To build a lightweight, explainable ML tool to evaluate and rank B2B business leads based on contact and company features — helping prioritize outreach and reduce manual filtering.

## 📦 Tech Stack

- Python
- scikit-learn
- SHAP (explainability)
- Streamlit (UI)
- Google Colab (model building)

## 🧠 Key Features

- Simulated and labeled 150 leads based on realistic attributes.
- Trained a Random Forest classifier to predict lead quality.
- Scored leads using predicted probability and ranked them.
- Used SHAP to explain top influencing features.
- Built an interactive UI using Streamlit:
  - Upload leads.csv
  - View top leads and lead scores
  - Download results
  - See SHAP-based feature importance

## 📂 File Structure

| File               | Description                                  |
|--------------------|----------------------------------------------|
| `app.py`           | Streamlit application script                 |
| `model.pkl`        | Trained machine learning model               |
| `sample_leads.csv` | Sample input leads for testing the tool      |
| `requirements.txt` | List of required Python packages             |
| `report.pdf`       | One-page project summary for Caprae          |

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
