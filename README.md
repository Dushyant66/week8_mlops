# IRIS Data Poisoning Experiment with MLflow

## Objective
Analyze the impact of **data poisoning (random noise)** at different levels — **5%**, **10%**, and **50%** — on the **IRIS dataset**, and track results using **MLflow**.

This experiment demonstrates how even small data corruption can alter model performance and explains how to mitigate such attacks.

---

## Setup Instructions

### 1️ Clone Repo
```bash
git clone https://github.com/<your-username>/iris_poisoning_project.git
cd iris_poisoning_project
```

### 2️ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # mac/linux
venv\Scripts\activate       # windows
```

### 3️ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️ Run MLflow Tracking Server
```bash
mlflow server --host 0.0.0.0 --port 8100
```

### 5️ Run Experiment Script
```bash
python scripts/train_poison_experiment.py
```

---

## Outputs

All experiment artifacts are stored in the `outputs/` directory:

| File | Description |
|------|--------------|
| `report_0.txt`, `report_5.txt`, `report_10.txt`, `report_50.txt` | Classification reports for each poisoning level |
| `poison_accuracy_plot.png` | Accuracy vs. Poisoning Level graph |
| `analysis_report.md` | Markdown summary of experiment outcomes |

---

## Experiment Summary

A **Decision Tree Classifier** was trained using the IRIS dataset with random Gaussian noise injected into feature values.  
Noise was applied to 0%, 5%, 10%, and 50% of the samples to simulate data poisoning.

- At **5% poisoning**, model accuracy dropped slightly — showing mild robustness.  
- At **10%**, misclassification began to appear across all classes.  
- At **50%**, the model lost generalization ability, with random-like predictions.

All experiments were logged in **MLflow**, tracking parameters, accuracy, F1-score, and confusion matrices.

---

##  Mitigation Strategies

1. **Data Validation Pipelines:** Implement pre-ingestion checks to detect outliers or anomalies.  
2. **Robust Models:** Use ensemble or robust loss functions less sensitive to noisy samples.  
3. **Data Provenance:** Track source and lineage of datasets with versioning tools (like DVC).  
4. **Adversarial Training:** Retrain models on mixed clean + perturbed data to increase resilience.

---

##  Data Quality vs Quantity

When **data quality decreases**, the model needs **significantly more data** to reach the same performance levels.  
Hence, simply increasing dataset size cannot compensate for severe poisoning — **quality always outweighs quantity**.

---

## Author
**Dushyant Singh Bhadauriya**  

