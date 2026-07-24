import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict"

cluster_actions = {
    0: {
        "action": """
        • Complete Transaction 
        """
    },
    1: {
        "action": """
        • Block Transaction
        """
    }
}

st.title("🛍️ Fraud Detection")
st.write("Predict fraud trasaction via our production FastAPI engine.")

with st.form("rfm_form"):
    accountAgeDays = st.number_input("Account Age (Days)", min_value=0, value=0, help="Number of days when account got created")
    paymentMethodAgeDays = st.number_input("Payment Type Age (Days)", min_value=0, value=0, help="Number of days when payment method got activated")

    submitted = st.form_submit_button("Predict Transaction Type")

if submitted:
    payload = {
        "accountAgeDays": int(accountAgeDays),
        "paymentMethodAgeDays": int(paymentMethodAgeDays)
    }

    try:
        # 2. Make an HTTP POST request to the FastAPI backend
        with st.spinner("Communicating with backend API engine..."):
            response = requests.post(BACKEND_URL, json=payload, timeout=5)
        
        # 3. Handle the response
        if response.status_code == 200:
            # Parse out the JSON results sent back from FastAPI
            result = response.json()
            prediction_class = result["prediction_class"]
            prediction = result["prediction"]

            st.success(f"### 🏷 Transaction Class: **{prediction}** (Type {prediction_class})")
            st.markdown("#### 📌 Recommended Actions:")
            st.write(cluster_actions[prediction_class]['action'])
        else:
            st.error(f"❌ Backend returned an error status: {response.status_code}")
            st.json(response.json())

    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI backend. Is your Uvicorn server running?")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}")