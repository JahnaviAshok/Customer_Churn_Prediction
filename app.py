import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Paths
MODEL_PATH = 'outputs/logistic_model_churn.jbl'
SCALER_PATH = 'outputs/scaler_churn.jbl'
FEATURE_PATH = 'outputs/feature_columns.joblib'
DATA_PATH = 'data/processed_telecom.csv'

# Load data and artifacts
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# Map churn values for better labeling
df['Churn'] = df['Churn'].map({0: "No", 1: "Yes"})

# Set Streamlit page
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Dashboard")
tab1, tab2 = st.tabs(["🔮 Prediction", "📈 Insights"])

# === PREDICTION TAB ===
with tab1:
    st.header("Predict Customer Churn")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
    contract_type = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'Contract_One year': 1 if contract_type == "One year" else 0,
        'Contract_Two year': 1 if contract_type == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    if 'TotalCharges' in feature_columns:
        input_df['TotalCharges'] = input_df['MonthlyCharges'] * input_df['tenure']
    scale_cols = ['tenure', 'MonthlyCharges', 'TotalCharges'] if 'TotalCharges' in feature_columns else ['tenure', 'MonthlyCharges']
    input_df[scale_cols] = scaler.transform(input_df[scale_cols])

    if st.button("Predict Churn"):
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        st.subheader("🔍 Prediction Result")
        st.write(f"**Churn Probability:** {prob:.2%}")
        if pred == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")

# === INSIGHTS TAB ===
with tab2:
    st.header("📈 Churn Data Trends")
    st.metric("📊 Overall Churn Rate", f"{df['Churn'].value_counts(normalize=True).get('Yes', 0):.2%}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Monthly Charges Distribution")
        fig1 = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay",
                            color_discrete_map={"Yes": "royalblue", "No": "lightblue"},
                            template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("⏳ Tenure by Churn")
        fig2 = px.box(df, x="Churn", y="tenure", color="Churn",
                      color_discrete_map={"Yes": "royalblue", "No": "lightblue"},
                      template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📄 Churn Rate by Contract Type")
        churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean()).reset_index()
        fig3 = px.bar(churn_by_contract, x="Contract", y="Churn",
                      color_discrete_sequence=["royalblue"], template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📶 Internet Service Type Distribution")
        fig4 = px.pie(df, names="InternetService", title="Internet Service Type",
                      template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)
