from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_loan_data(raw_path: Path, n_rows: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    genders = ["Male", "Female"]
    married = ["Yes", "No"]
    dependents = ["0", "1", "2", "3+"]
    education = ["Graduate", "Not Graduate"]
    self_employed = ["Yes", "No"]
    property_area = ["Urban", "Semiurban", "Rural"]
    terms = [120, 180, 240, 300, 360, 480]

    applicant_income = rng.normal(5500, 2700, size=n_rows).clip(1000, 30000).round(0)
    coapplicant_income = rng.normal(1500, 1400, size=n_rows).clip(0, 15000).round(0)
    loan_amount = rng.normal(130, 40, size=n_rows).clip(20, 700).round(0)
    loan_term = rng.choice(terms, size=n_rows, p=[0.05, 0.1, 0.2, 0.25, 0.3, 0.1])
    credit_history = rng.choice([1.0, 0.0], size=n_rows, p=[0.85, 0.15])

    rows = []
    for i in range(n_rows):
        row = {
            "Gender": rng.choice(genders),
            "Married": rng.choice(married, p=[0.65, 0.35]),
            "Dependents": rng.choice(dependents, p=[0.6, 0.2, 0.12, 0.08]),
            "Education": rng.choice(education, p=[0.78, 0.22]),
            "Self_Employed": rng.choice(self_employed, p=[0.12, 0.88]),
            "ApplicantIncome": int(applicant_income[i]),
            "CoapplicantIncome": int(coapplicant_income[i]),
            "LoanAmount": int(loan_amount[i]),
            "Loan_Amount_Term": int(loan_term[i]),
            "Credit_History": float(credit_history[i]),
            "Property_Area": rng.choice(property_area, p=[0.35, 0.4, 0.25]),
        }
        score = 0.25 * np.log1p(row["ApplicantIncome"]) + 0.35 * np.log1p(row["CoapplicantIncome"])
        score += 1.2 * row["Credit_History"]
        score += 0.3 if row["Education"] == "Graduate" else -0.1
        score += -0.15 if row["Self_Employed"] == "Yes" else 0.1
        score += 0.1 if row["Property_Area"] == "Urban" else (-0.05 if row["Property_Area"] == "Semiurban" else -0.15)
        approval_prob = 1 / (1 + np.exp(-score + 1.2))
        row["Loan_Status"] = int(rng.random() < approval_prob)
        rows.append(row)

    df = pd.DataFrame(rows)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    return df
