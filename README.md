# Olist MLOps Platform

Production MLOps and AI infrastructure for Brazilian E-Commerce data (100K+ real orders).

## Features

- **Demand Forecasting** - ML model with time-series features
- **Customer Support AI** - LangGraph workflow with intent classification
- **MLflow Tracking** - Experiment management and model registry
- **CI/CD Pipeline** - GitHub Actions automated testing
- **Docker Containerization** - Reproducible deployments
- **Live API** - Deployed on Render

## Tech Stack

| Category | Technologies |
|----------|--------------|
| ML & Forecasting | scikit-learn, pandas, numpy |
| MLOps | MLflow, GitHub Actions, Docker |
| AI Infrastructure | LangGraph, FastAPI |
| Deployment | Render, Uvicorn |
| Testing | pytest |

## Quick Start

```bash
# Clone
git clone https://github.com/jmahmud2/olist-mlops-platform.git
cd olist-mlops-platform

# Install
pip install -r requirements.txt

# Run API
uvicorn api.serve:app --reload

# Run tests
pytest tests/ -v