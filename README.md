# Loan Granting Problem

This repository contains an end-to-end loan granting prediction project built for a BFSI-style dataset.

## Project structure

- `app.py`: Streamlit app for interactive loan decision predictions.
- `train.py`: Trains a classification pipeline and stores the best model.
- `src/loan_granting/`: Modular package with data generation, preprocessing, modeling, and prediction.
- `data/raw/`: Raw dataset storage.
- `models/`: Saved model artifacts.

## Setup

1. Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python train.py
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- A synthetic BFSI-style loan dataset is generated automatically if no raw data exists.
- The model uses logistic regression, random forest, and gradient boosting to select the best ROC AUC.
- The app predicts whether a customer should be granted a loan and reports approval probability.
