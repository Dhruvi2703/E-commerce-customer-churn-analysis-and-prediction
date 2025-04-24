import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load(r"C:\Users\Patel\Desktop\intership\random_forest_model2.pkl")

# Default values including placeholder text
default_values = {
    "Tenure": "",
    "CityTier": "",
    "WarehouseToHome": "",
    "HourSpendOnApp": "",
    "NumberOfDeviceRegistered": "",
    "SatisfactionScore": "",
    "NumberOfAddress": "",
    "OrderAmountHikeFromlastYear": "",
    "CouponUsed": "",
    "OrderCount": "",
    "DaySinceLastOrder": "",
    "CashbackAmount": "",
    "PreferredLoginDevice": "Choose an option",
    "PreferredPaymentMode": "Choose an option",
    "PreferedOrderCat": "Choose an option",
    "MaritalStatus": "Choose an option"
}

# Reset form
def reset_form():
    for key, val in default_values.items():
        st.session_state[key] = val

# Initialize session state
for key, val in default_values.items():
    st.session_state.setdefault(key, val)

# Display image header (before the title)
#st.image("C:\Users\Patel\Desktop\Churn-Analysis.webp", use_column_width=True)



st.set_page_config(page_title="E-commerce Customer Churn Prediction", layout="centered")
# st.image(r"C:\Users\Patel\Desktop\Churn-Analysis.webp", use_column_width=False)
st.title("🛍️ E-commerce Customer Churn Prediction")
st.markdown("Fill in the details below to predict whether the customer will churn or not.")



# Form
with st.form("churn_form"):
    
    Tenure = st.text_input("Tenure (months)", value=st.session_state["Tenure"], key="Tenure")
    PreferredLoginDevice = st.selectbox(
        "Preferred Login Device",
        ["Choose an option", "Mobile", "Computer", "Tablet"],
        key="PreferredLoginDevice"
    )
    CityTier = st.text_input("City Tier (1-3)", value=st.session_state["CityTier"], key="CityTier")
    WarehouseToHome = st.text_input("Distance to Warehouse (km)", value=st.session_state["WarehouseToHome"], key="WarehouseToHome")
    PreferredPaymentMode = st.selectbox(
        "Payment Mode",
        ["Choose an option", "Debit Card", "Credit Card", "UPI", "Wallet"],
        key="PreferredPaymentMode"
    )
    Gender = st.radio("Gender", ["Male", "Female"])
    HourSpendOnApp = st.text_input("Hours Spent on App Per Day", value=st.session_state["HourSpendOnApp"], key="HourSpendOnApp")
    NumberOfDeviceRegistered = st.text_input("Number of Devices Registered", value=st.session_state["NumberOfDeviceRegistered"], key="NumberOfDeviceRegistered")
    PreferedOrderCat = st.selectbox(
        "Preferred Order Category",
        ["Choose an option", "Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery"],
        key="PreferedOrderCat"
    )
    SatisfactionScore = st.text_input("Satisfaction Score (1-5)", value=st.session_state["SatisfactionScore"], key="SatisfactionScore")
    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Choose an option", "Single", "Married", "Divorced"],
        key="MaritalStatus"
    )
    NumberOfAddress = st.text_input("Number of Addresses", value=st.session_state["NumberOfAddress"], key="NumberOfAddress")
    Complain = st.radio("Any Complaint?", ["No", "Yes"])
    OrderAmountHikeFromlastYear = st.text_input("Order Amount Increase (%)", value=st.session_state["OrderAmountHikeFromlastYear"], key="OrderAmountHikeFromlastYear")
    CouponUsed = st.text_input("Coupons Used Last Year", value=st.session_state["CouponUsed"], key="CouponUsed")
    OrderCount = st.text_input("Total Orders", value=st.session_state["OrderCount"], key="OrderCount")
    DaySinceLastOrder = st.text_input("Days Since Last Order", value=st.session_state["DaySinceLastOrder"], key="DaySinceLastOrder")
    CashbackAmount = st.text_input("Total Cashback Received", value=st.session_state["CashbackAmount"], key="CashbackAmount")

    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.form_submit_button("Predict Churn")
    with col2:
        reset_button = st.form_submit_button("Reset Form", on_click=reset_form)

# Categorical mappings
category_mappings = {
    "PreferredLoginDevice": {"Mobile": 1, "Computer": 2, "Tablet": 3},
    "PreferredPaymentMode": {"Debit Card": 2, "Credit Card": 1, "UPI": 4, "Wallet": 3},
    "Gender": {"Male": 1, "Female": 0},
    "PreferedOrderCat": {"Laptop & Accessory": 2, "Mobile Phone": 3, "Fashion": 1, "Grocery": 4},
    "MaritalStatus": {"Single": 2, "Married": 1, "Divorced": 3},
    "Complain": {"No": 0, "Yes": 1}
}

# Prediction
if predict_button:
    if PreferredLoginDevice == "Choose an option" or \
       PreferredPaymentMode == "Choose an option" or \
       PreferedOrderCat == "Choose an option" or \
       MaritalStatus == "Choose an option":
        st.error("Please select valid options from all dropdowns.")
    else:
        try:
            inputs = [
                int(Tenure),
                category_mappings["PreferredLoginDevice"][PreferredLoginDevice],
                int(CityTier),
                int(WarehouseToHome),
                category_mappings["PreferredPaymentMode"][PreferredPaymentMode],
                category_mappings["Gender"][Gender],
                int(HourSpendOnApp),
                int(NumberOfDeviceRegistered),
                category_mappings["PreferedOrderCat"][PreferedOrderCat],
                int(SatisfactionScore),
                category_mappings["MaritalStatus"][MaritalStatus],
                int(NumberOfAddress),
                category_mappings["Complain"][Complain],
                int(OrderAmountHikeFromlastYear),
                int(CouponUsed),
                int(OrderCount),
                int(DaySinceLastOrder),
                int(CashbackAmount)
            ]

            df_input = pd.DataFrame([inputs], columns=[
                "Tenure", "PreferredLoginDevice", "CityTier", "WarehouseToHome", "PreferredPaymentMode",
                "Gender", "HourSpendOnApp", "NumberOfDeviceRegistered", "PreferedOrderCat",
                "SatisfactionScore", "MaritalStatus", "NumberOfAddress", "Complain",
                "OrderAmountHikeFromlastYear", "CouponUsed", "OrderCount", "DaySinceLastOrder", "CashbackAmount"
            ])

            st.subheader("📋 Customer Details")
            st.dataframe(df_input)

            prediction = model.predict(np.array(inputs).reshape(1, -1))[0]
            result = "🚨 Churn" if prediction == 1 else "✅ No Churn"
            st.success(f"Prediction Result: **{result}**")

        except ValueError:
            st.error("Please enter valid numeric values in all text fields.")

# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)







