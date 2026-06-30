from setuptools import setup, find_packages

setup(
    name="loan-granting",
    version="0.1.0",
    description="End-to-end loan granting prediction system with Streamlit UI.",
    author="Loan Granting Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "streamlit>=1.20.0",
        "pandas>=2.0.0",
        "numpy>=1.25.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.0",
        "python-dateutil>=2.8.0",
    ],
    python_requires=">=3.9",
)
