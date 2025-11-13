# 📊 Validation Outcomes and Analysis — IRIS Data Poisoning

### Objective
To analyze how **data poisoning** (injected random noise) at different levels (5%, 10%, 50%) affects model validation outcomes when trained on the IRIS dataset, and to log experiments using **MLflow**.

| Poisoning Level | Accuracy | Observation |
|-----------------|-----------|-------------|
| 0% | 0.983 | Clean data gives best accuracy |
| 5% | 0.917 | Small poisoning still stable |
| 10% | 0.917 | Small poisoning still stable |
| 50% | 0.883 | Heavy poisoning distorts decision boundaries |

### Key Insights
- Increasing poisoning level reduces accuracy.
- Minor noise (<10%) is tolerable; beyond that, generalization drops.
- MLflow logging enables full reproducibility and visualization.

### Mitigation
- Validate input data (outlier and anomaly detection).
- Use robust training (ensembles, regularization).
- Track dataset versions with MLflow.
- Secure and audit all data access.
