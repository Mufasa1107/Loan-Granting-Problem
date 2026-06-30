import streamlit as st
from pathlib import Path
import pandas as pd
from loan_granting.predict import predict_loan_status
from loan_granting.config import MODEL_PATH
from loan_granting.modeling import load_model

MODEL = None


def load_pipeline():
    global MODEL
    if MODEL is None:
        MODEL = load_model(MODEL_PATH)
    return MODEL


def build_input_form() -> dict:
    st.sidebar.header("Customer profile")
    profile = {
        "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "Married": st.sidebar.selectbox("Married", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"]),
        "Education": st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"]),
        "Self_Employed": st.sidebar.selectbox("Self Employed", ["Yes", "No"]),
        "ApplicantIncome": st.sidebar.number_input("Applicant Income", min_value=0, value=5000, step=500),
        "CoapplicantIncome": st.sidebar.number_input("Coapplicant Income", min_value=0, value=0, step=500),
        "LoanAmount": st.sidebar.number_input("Loan Amount (in 1000s)", min_value=0, value=100, step=10),
        "Loan_Amount_Term": st.sidebar.selectbox("Loan Amount Term", [120, 180, 240, 300, 360, 480]),
        "Credit_History": st.sidebar.selectbox("Credit History", [1.0, 0.0]),
        "Property_Area": st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"]),
    }
    return profile


def main():
    st.set_page_config(page_title="Loan Granting Decision System", layout="centered")
    st.title("Loan Granting Prediction")
    st.markdown(
        "Use this interactive app to estimate whether a customer should be granted a loan based on BFSI features."
    )

    pipeline = load_pipeline()
    profile = build_input_form()

    if st.button("Predict Loan Decision"):
        prediction, probability = predict_loan_status(pipeline, profile)
        st.success(f"Loan decision: **{prediction}**")
        st.write(f"Approval probability: **{probability * 100:.1f}%**")

    st.markdown("---")
    st.markdown(
        "### How to use\n1. Adjust borrower details in the sidebar.\n2. Click 'Predict Loan Decision'.\n3. Review approval probability and model output."
    )


if __name__ == "__main__":
    main()
