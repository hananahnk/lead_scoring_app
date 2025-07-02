# 🚀 Smart Lead Scoring – Internship Project

This project was submitted as part of the **Machine Learning Engineer Internship** task at **Caprae Capital**.

It is a lightweight, browser-based machine learning tool that helps rank B2B leads based on job title, company size, contact availability, and industry relevance.

---

## 🔗 Live Demo (No Installation Required)

You can test the working application directly in your browser:

👉 **[Launch the App](https://leadscoringapp-fvjnxlhcwqipb7lnophruj.streamlit.app/)**  
No sign-up or setup is needed.

---

## 🖱️ How to Use the Web App

1. **Prepare your input CSV** (or use the provided sample):  
   Your `leads.csv` file should have the following columns:
**Go to the app:**  
👉 [https://leadscoringapp-fvjnxlhcwqipb7lnophruj.streamlit.app](https://leadscoringapp-fvjnxlhcwqipb7lnophruj.streamlit.app)

3. **Upload your file** using the file uploader('leads.csv')

4. **View results instantly:**
- The top 10 leads (based on ML scores) will appear
- You can also download the full `scored_leads.csv`

---

## 💼 Objective

To automate lead prioritization using machine learning — scoring each lead with a probability from 0 to 100 that reflects its potential value, based on domain logic.

---

## 🧠 Key Features

- Built and trained a Random Forest model on 150 synthetic leads
- Encoded categorical features (title, industry, company size)
- Scored leads based on contact signals and role relevance
- Built a no-code Streamlit UI for business users to upload and analyze leads
- SHAP analysis was conducted during development (visuals attached separately)

---

## 🛠 Tech Stack

- Python (3.x)
- scikit-learn (Random Forest)
- pandas, numpy
- Streamlit (frontend)
- Google Colab (model training)
- Faker (synthetic data generation)

---

## 📁 Repository Contents

| File               | Description                                   |
|--------------------|-----------------------------------------------|
| `app.py`           | Streamlit application                         |
| `model.pkl`        | Trained ML model                              |
| `requirements.txt` | Python dependencies                           |
| `sample_leads.csv` | Example input file for testing                |
| `scored_leads.csv` | Example output file (after scoring)           |
| `report.pdf`       | One-page summary of the project               |
| `shap_plot.png`    | Optional feature importance plot              |

---

## ⚙️ Optional Local Setup

If you prefer to run it locally instead of the web:

```bash
pip install -r requirements.txt
streamlit run app.py
