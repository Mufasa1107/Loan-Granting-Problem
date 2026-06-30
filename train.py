from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from loan_granting.data_generator import generate_synthetic_loan_data
from loan_granting.data_loader import load_raw_data
from loan_granting.modeling import train_and_select_model, save_model, evaluate_model
from loan_granting.config import RAW_DATA_PATH, MODEL_PATH, MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DATA_PATH.exists():
        logging.info("No raw data found. Generating synthetic BFSI loan dataset for demonstration.")
        generate_synthetic_loan_data(RAW_DATA_PATH)

    data = load_raw_data(RAW_DATA_PATH)
    X = data.drop(columns=["Loan_Status"])
    y = data["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_pipeline, results = train_and_select_model(X_train, y_train, X_test, y_test)
    save_model(best_pipeline, MODEL_PATH)

    logging.info("Best model saved to %s", MODEL_PATH)
    evaluate_model(best_pipeline, X_test, y_test)


if __name__ == "__main__":
    main()
