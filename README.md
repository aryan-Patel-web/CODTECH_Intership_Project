# CODTECH Internship — Data Science Tasks
**Intern:** Aryan Patel  
**Intern ID:** CT08DR2597  
**Internship period:** Dec 4, 2025 — Feb 4, 2026. (From offer letter). :contentReference[oaicite:1]{index=1}

## Overview
This repository contains solutions for the four CODTECH data science internship tasks:
1. Data preprocessing ETL pipeline (Pandas & scikit-learn).
2. Deep learning image classification (PyTorch).
3. End-to-end data science project with a FastAPI deployment.
4. Optimization problem solved using PuLP.

Each task includes code, sample data or instructions to create sample data, and instructions to run.

## Repo structure
(see same tree as in the top-level README in the project root)

## Setup
1. Create virtual env:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt


Run ETL:

python src/task1_data_pipeline.py
# Checks sample data in data/


Train / run DL model:

Put images under data/images/train/<class>/ and data/images/val/<class>/.

python src/task2_dl_image_classification.py


Run API:

uvicorn src.task3_end_to_end_api:app --reload --port 8000
# Visit http://localhost:8000


Optimization notebook:

Open notebooks/task4_optimization_pulp.ipynb and run cells.

Notes & Best Practices

Commenting: All core files are commented for readability.

Git: Commit code and push to a GitHub repository named codtech-internship-CT08DR2597.

Submission: Zip the repo or share GitHub URL following CODTECH group updates. Include this README and offer letter (PDF) in the repo for verification.

Citation: Internship offer letter (for verification) is included in root directory: CT08DR2597.pdf. 



