# 🧼 Detergent Sales Analysis & Forecasting Web App

This project is a Streamlit-based interactive web application that analyzes detergent sales trends, investigates a reported 11% decline from 2018 to 2019, and uses a machine learning model to forecast daily sales quantity.

The app integrates transaction, product, and store datasets, visual insights, and forecasting tools to help optimize inventory, promotions, and business strategy.

---

## 🔍 Features

- 📊 Power BI visuals for overall sales drop and key insights
- 🏪 Data previews for transactions, products, and stores
- 🔎 Sales pattern analysis by day of the week
- 🤖 Machine Learning model using Random Forest to:
  - Forecast total quantity sold for a selected day
  - Highlight whether the day’s sales are above/below historical average
- 💡 Business recommendations to improve future sales

---

## 🧠 ML Model Details

- **Model**: Random Forest Regressor
- **Target**: Total sales quantity per day
- **Features used**:
  - `day_of_week` (0 = Monday, 6 = Sunday)
  - `is_weekend` (1 for Saturday/Sunday, else 0)

---

---

## 🧰 Requirements

Install dependencies using pip:

```bash
pip install streamlit pandas scikit-learn pillow

## How to Run
Place the following files in the same folder:

TransactionData.xlsx

ProductData.xlsx

Store.xlsx

profitloss.png

INSIGHTS.png

app.py (your Python script)

Open terminal/command prompt in that directory

Run the Streamlit app:
streamlit run app.py

