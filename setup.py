from setuptools import setup, find_packages

setup(
    name="fedshield-ids",
    version="1.0.0",
    description="Privacy-Preserving Federated Learning Intrusion Detection System",
    author="PFE Project",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "flwr[simulation]>=1.8.0",
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "streamlit>=1.35.0",
        "plotly>=5.22.0",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
        "rich>=13.7.0",
        "shap>=0.44.0",
        "lime>=0.2.0.1",
    ],
    entry_points={"console_scripts": ["fedshield=main:cli"]},
)
