import os
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from mlflow.models import infer_signature
from mlflow import MlflowClient

# === CONFIGURATION ===
OUTPUT_DIR = "outputs"
DATA_PATH = "data/iris.csv"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
data = pd.read_csv(DATA_PATH)

# Encode categorical label
if data["species"].dtype == "object":
    data["species"] = data["species"].astype("category").cat.codes

X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = data["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
)

# === DATA POISONING FUNCTION ===
def poison_data(X, level=0.05, random_state=None):
    """Add Gaussian noise to feature matrix at specified poison level."""
    X_poisoned = X.copy()
    n_samples = int(len(X) * level)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.index, n_samples, replace=False)
    noise = rng.normal(0, 0.5, size=(n_samples, X.shape[1]))
    X_poisoned.loc[idx] += noise
    # Clip values column-wise to original min/max to avoid unrealistic features
    X_poisoned = X_poisoned.clip(lower=X.min(), upper=X.max(), axis=1)
    return X_poisoned


# === MLFlow EXPERIMENT SETUP ===
mlflow.set_tracking_uri("http://127.0.0.1:8100")
mlflow.set_experiment("IRIS Data Poisoning Experimentssssss")
client = MlflowClient(mlflow.get_tracking_uri())

results = []
poison_levels = [0.0, 0.05, 0.10, 0.50]

for level in poison_levels:
    print(f"\n🔹 Training with {int(level*100)}% poisoned data")

    X_train_poisoned = poison_data(X_train, level, random_state=RANDOM_STATE)

    # Save poisoned dataset as artifact
    poisoned_csv = os.path.join(OUTPUT_DIR, f"X_train_poisoned_{int(level*100)}.csv")
    X_train_poisoned.to_csv(poisoned_csv, index=False)

    with mlflow.start_run(run_name=f"Poison_{int(level*100)}%"):
        # Model setup
        params = {"max_depth": 3, "criterion": "entropy", "random_state": RANDOM_STATE}
        model = DecisionTreeClassifier(**params)
        model.fit(X_train_poisoned, y_train)

        # Predictions & metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Logging
        mlflow.log_params(params)
        mlflow.log_param("poison_level", level)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("experiment_type", "data_poisoning")
        mlflow.log_artifact(poisoned_csv)

        # Log model
        signature = infer_signature(X_train_poisoned, model.predict(X_train_poisoned))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"IRIS-classifier-dt-{int(level*100)}",
            signature=signature,
            input_example=X_train_poisoned.head()
        )

        # Save classification report
        report_path = os.path.join(OUTPUT_DIR, f"report_{int(level*100)}.txt")
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, preds))
        mlflow.log_artifact(report_path)

        print(f"✅ Accuracy with {int(level*100)}% poison: {acc:.3f}")
        results.append((level, acc))

# === PLOT RESULTS ===
levels = [int(l * 100) for l, _ in results]
accuracies = [a for _, a in results]

plt.figure(figsize=(6, 4))
plt.plot(levels, accuracies, marker="o", linewidth=2)
plt.axhline(y=accuracies[0], color='r', linestyle='--', label="Clean Accuracy")
plt.title("Impact of Data Poisoning on Model Accuracy")
plt.xlabel("Poisoning Level (%)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plot_path = os.path.join(OUTPUT_DIR, "poison_accuracy_plot.png")
plt.savefig(plot_path, bbox_inches="tight")
mlflow.log_artifact(plot_path)
