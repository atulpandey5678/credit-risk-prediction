import os
import sys
from pathlib import Path

import requests
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.prediction_pipeline import PredictionPipeline

API_URL = os.getenv("API_URL", "").rstrip("/")
USE_API = bool(API_URL)

LOAN_PURPOSES = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "medical",
    "other",
]


@st.cache_resource(show_spinner="Loading model...")
def load_pipeline() -> PredictionPipeline:
    return PredictionPipeline()


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def predict_via_api(payload: dict) -> dict:
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def render_results(result: dict) -> None:
    prob = result["probability"]
    risk = result["risk_category"]
    prediction = result.get("prediction", int(prob >= 0.5))

    st.divider()
    st.subheader("Prediction results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{prob:.1%}")
    c2.metric("Risk category", risk)
    c3.metric("Predicted default", "Yes" if prediction == 1 else "No")

    st.progress(min(max(prob, 0.0), 1.0))

    message = f"**{risk}** — probability of default: {prob:.2%}"
    if risk == "Low Risk":
        st.success(message)
    elif risk == "Medium Risk":
        st.warning(message)
    else:
        st.error(message)


def main() -> None:
    st.set_page_config(
        page_title="Credit Risk Prediction",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Credit Risk Prediction System")
    st.caption("Evaluate loan default risk from borrower financial and loan details.")

    model_ready = False
    pipeline = None
    api_ok = False

    if USE_API:
        api_ok = check_api_health()
        model_ready = api_ok
    else:
        try:
            pipeline = load_pipeline()
            model_ready = True
        except FileNotFoundError:
            model_ready = False
        except Exception as exc:
            model_ready = False
            st.session_state["model_error"] = str(exc)

    with st.sidebar:
        st.header("System status")
        if USE_API:
            if api_ok:
                st.success("Remote API connected")
            else:
                st.error("Remote API unreachable")
                st.caption(f"API_URL: {API_URL}")
        elif model_ready:
            st.success("Model loaded")
        else:
            st.error("Model not available")
            err = st.session_state.get("model_error")
            if err:
                st.caption(err)
            else:
                st.caption("Ensure `credit_risk_prediction/models/*.pkl` are in the repo.")

        st.divider()
        st.markdown("**Risk bands**")
        st.markdown("- **Low:** below 20%")
        st.markdown("- **Medium:** 20%–60%")
        st.markdown("- **High:** above 60%")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial details")
        loan_amount = st.number_input(
            "Loan amount ($)", min_value=500, max_value=200_000, value=15_000, step=500
        )
        annual_income = st.number_input(
            "Annual income ($)", min_value=10_000, max_value=2_000_000, value=65_000, step=1_000
        )
        interest_rate = st.number_input(
            "Interest rate (%)", min_value=1.0, max_value=40.0, value=10.5, step=0.1
        )
        debt_to_income_ratio = st.number_input(
            "Debt-to-income ratio (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.5
        )
        credit_score = st.slider("Credit score", min_value=300, max_value=850, value=700)

    with col2:
        st.subheader("Personal & loan details")
        employment_length = st.number_input(
            "Employment length (years)", min_value=0, max_value=50, value=5
        )
        loan_term = st.selectbox("Loan term (months)", [36, 60])
        loan_purpose = st.selectbox("Loan purpose", LOAN_PURPOSES)
        home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE"])

    payload = {
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "employment_length": int(employment_length),
        "annual_income": annual_income,
        "credit_score": int(credit_score),
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_term": int(loan_term),
        "loan_purpose": loan_purpose,
        "home_ownership": home_ownership,
    }

    if st.button(
        "Predict default risk",
        type="primary",
        use_container_width=True,
        disabled=not model_ready,
    ):
        try:
            if USE_API:
                result = predict_via_api(payload)
            else:
                result = pipeline.predict(payload)
        except requests.exceptions.HTTPError as exc:
            st.error(exc.response.text if exc.response is not None else str(exc))
            return
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
            return
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        render_results(result)


if __name__ == "__main__":
    main()
