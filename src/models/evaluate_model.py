import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    pipeline = joblib.load(
        PROJECT_ROOT / "models" / "best_pipeline.joblib"
    )
    test_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "test.csv",
        keep_default_na=False,
    )

    predictions = pipeline.predict(test_df["text"])
    accuracy = accuracy_score(test_df["label"], predictions)

    report = classification_report(
        test_df["label"],
        predictions,
        labels=[-1, 0, 1],
        target_names=["Negatif", "Neutre", "Positif"],
        output_dict=True,
        zero_division=0,
    )

    print(f"Accuracy : {accuracy:.2%}")
    print("\nRapport par classe :")
    print(classification_report(
        test_df["label"],
        predictions,
        labels=[-1, 0, 1],
        target_names=["Negatif", "Neutre", "Positif"],
        digits=3,
        zero_division=0,
    ))

    # Mesurer TF-IDF + prédiction des probabilités sur 50 textes.
    batch = test_df["text"].iloc[:50].tolist()
    pipeline.predict_proba(batch)  # Premier passage de chauffe.

    durations = []
    for _ in range(50):
        start = perf_counter()
        pipeline.predict_proba(batch)
        durations.append((perf_counter() - start) * 1000)

    median_ms = float(np.median(durations))
    p95_ms = float(np.percentile(durations, 95))

    print(f"Batch mesuré : {len(batch)} commentaires")
    print(f"Temps médian : {median_ms:.2f} ms")
    print(f"Temps au 95e percentile : {p95_ms:.2f} ms")

    accuracy_ok = accuracy >= 0.80
    f1_ok = all(
        report[label]["f1-score"] > 0.75
        for label in ["Negatif", "Neutre", "Positif"]
    )
    speed_ok = p95_ms < 100

    print("\nVérification des objectifs :")
    print(f"Accuracy >= 80 % : {accuracy_ok}")
    print(f"F1 > 0.75 pour chaque classe : {f1_ok}")
    print(f"Inférence locale p95 < 100 ms : {speed_ok}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "benchmark": {
            "batch_size": len(batch),
            "repetitions": 50,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
        },
    }

    with (REPORT_DIR / "metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    ConfusionMatrixDisplay.from_predictions(
        test_df["label"],
        predictions,
        labels=[-1, 0, 1],
        display_labels=["Negatif", "Neutre", "Positif"],
        cmap="Blues",
        values_format="d",
    )
    plt.title("Matrice de confusion — modèle optimisé")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print("\nRésultats enregistrés dans reports/")


if __name__ == "__main__":
    main()