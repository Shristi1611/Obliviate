# Obliviate 
> **A SISA-Based Machine Unlearning System for Sepsis Risk Prediction**

Obliviate is a verifiable machine unlearning framework that enforces the "Right to be Forgotten" in clinical AI. Built on SISA architecture with PyTorch and TabNet on PhysioNet Sepsis ICU data, it executes fast, localized slice rollbacks during deletion requests—slashing retrain time while preserving predictive accuracy.

---

##  Key Features

- **Fast Machine Unlearning:** Localized slice rollbacks avoid full model retraining.
- **Sepsis Risk Prediction:** Uses TabNet architectures trained on ICU time-series data.
- **Extended Operations:** Supports record deletions, single-slice insertions, and label updates.
- **Verifiable Engine:** Built-in benchmarking to compare SISA unlearning against full retrains.
- **Comprehensive Audit Logs:** Tracks execution times, affected slices, and accuracy metrics.
- **Interactive Web Dashboard:** Built with a FastAPI backend for real-time monitoring.

---

## Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** PyTorch, `pytorch-tabnet`, Scikit-Learn
- **Backend API:** FastAPI, Uvicorn, Pydantic
- **Database:** SQLite, SQLAlchemy / DB-API
- **Frontend:** HTML5, CSS3, JavaScript

---

## Quick Start

### 1. Installation
```bash
git clone [https://github.com/your-username/obliviate-unlearning.git](https://github.com/your-username/obliviate-unlearning.git)
cd obliviate-unlearning
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
