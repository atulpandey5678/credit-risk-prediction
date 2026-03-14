import streamlit as st
import requests
import os



def main():
    st.set_page_config(page_title="Credit Risk Prediction System", layout="wide")
    st.title("Credit Risk Prediction System :money_with_wings:")
    st.write("Enter the borrower's details below to predict the likelihood of loan default.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Financial Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=200000, value=15000)
        annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=65000)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=10.5)
        debt_to_income_ratio = st.number_input("Debt to Income Ratio (%)", min_value=0.0, max_value=100.0, value=20.0)
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
        
    with col2:
        st.subheader("Personal & Loan Details")
        employment_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
        loan_term = st.selectbox("Loan Term (months)", [36, 60])
        
        purposes = ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "medical", "other"]
        loan_purpose = st.selectbox("Loan Purpose", purposes)
        
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])

    if st.button("Predict Default Risk", use_container_width=True):
        data_dict = {
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "employment_length": employment_length,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "debt_to_income_ratio": debt_to_income_ratio,
            "loan_term": loan_term,
            "loan_purpose": loan_purpose,
            "home_ownership": home_ownership
        }
        
        try:
            # Send prediction request to FastAPI Service
            api_url = "http://localhost:8000/predict"
            response = requests.post(api_url, json=data_dict)
            
            if response.status_code == 200:
                result = response.json()
                
                st.divider()
                st.subheader("Prediction Results")
                
                prob = result["probability"]
                risk = result["risk_category"]
                
                if risk == "Low Risk":
                    st.success(f"**{risk}** (Probability of Default: {prob:.2%})")
                elif risk == "Medium Risk":
                    st.warning(f"**{risk}** (Probability of Default: {prob:.2%})")
                else:
                    st.error(f"**{risk}** (Probability of Default: {prob:.2%})")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the Prediction API. Is the FastAPI server running on localhost:8000?")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
