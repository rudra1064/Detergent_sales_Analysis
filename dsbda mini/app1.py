import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Data
transactions = pd.read_excel(r"C:\Users\kamle\Desktop\dsbda mini\TransactionData.xlsx")
products = pd.read_excel(r"C:\Users\kamle\Desktop\dsbda mini\ProductData.xlsx")
stores = pd.read_excel(r"C:\Users\kamle\Desktop\dsbda mini\Store.xlsx")

st.title("DETERGENT SALES ANALYSIS")

st.subheader("Report and Insights for Declining Sales (11% decrease from 2018 to 2019)")
st.write("This report analyzes the reasons behind the decline in sales and provides insights into inventory management and optimization.")

# Load Images
image = Image.open("profitloss.png")
st.image(image, caption="Power BI Dashboard", use_container_width=True)

image1 = Image.open("INSIGHTS.png")
st.image(image1, caption="Power BI Insights", use_container_width=True)

st.subheader("SOLUTIONS")
st.write("1. Increase inventory levels for high-selling products while reducing stock for low-performing ones.")
st.write("2. Offer discounts on slow-moving items to boost revenue.")
st.write("3. Implement targeted loyalty programs to retain Elite and Good customers.")
st.write("4. Enhance data tracking to analyze the Unknown Customer segment for better sales attribution.")
st.write("5. Adjust inventory to align with the increasing demand for budget-friendly products.")

st.markdown("---")

st.subheader("Data Preview")
st.write("### Transactions Data", transactions.head())
st.write("### Products Data", products.head())
st.write("### Stores Data", stores.head())

# Data Cleaning and Merging
transactions.dropna(inplace=True)
products.dropna(inplace=True)
stores.dropna(inplace=True)

transactions['TXN_DT'] = pd.to_datetime(transactions['TXN_DT'])
transactions['day_of_week'] = transactions['TXN_DT'].dt.dayofweek  # Monday=0, Sunday=6
transactions['is_weekend'] = transactions['day_of_week'].isin([5, 6]).astype(int)

# Compute average sales per day
avg_sales_by_day = transactions.groupby("day_of_week")["ITEM_QTY"].mean()

# Merge data
data = transactions.merge(products, on='UPC_ID', how='left').merge(stores, on='STORE_ID', how='left')

# -------------------------------
# Forecast Daily Sales Quantity
# -------------------------------
st.subheader("Daily Sales Forecasting Model")

# Create Daily Aggregated Dataset
daily_sales = transactions.groupby('TXN_DT')['ITEM_QTY'].sum().reset_index()
daily_sales['day_of_week'] = daily_sales['TXN_DT'].dt.dayofweek
daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)

# Model Inputs
X = daily_sales[['day_of_week', 'is_weekend']]
y = daily_sales['ITEM_QTY']

X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
score = model.score(X_test, y_test)

st.write(f"Model Accuracy: {score * 100:.2f}%")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# -------------------------------
# Prediction Input Section
# -------------------------------
st.subheader("Predict Total Sales for Selected Day")

day_of_week = st.selectbox("Choose a Day (0=Monday, 6=Sunday)", [0, 1, 2, 3, 4, 5, 6])
is_weekend = 1 if day_of_week in [5, 6] else 0

if st.button("Forecast Total Sales for Selected Day"):
    input_data = np.array([[day_of_week, is_weekend]])
    prediction = model.predict(input_data)
    st.write(f"📅 Forecasted Total Quantity Sold on Day {day_of_week}: **{prediction[0]:.2f} units**")

    # Historical comparison
    avg_day_sales = daily_sales[daily_sales['day_of_week'] == day_of_week]['ITEM_QTY'].mean()
    if prediction[0] > avg_day_sales:
        st.success("Predicted sales are lower than the average for this day.")
    else:
        st.warning("Predicted sales are higher than the usual average for this day.")
