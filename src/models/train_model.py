from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


def main():
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Exécute d'abord : python src/data/split_data.py"
        )

    train_df = pd.read_csv(train_path, keep_default_na=False)
    test_df = pd.read_csv(test_path, keep_default_na=False)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    print("Entraînement en cours...")
    start = perf_counter()
    pipeline.fit(train_df["text"], train_df["label"])
    print(f"Entraînement terminé en {perf_counter() - start:.1f} secondes")

    predictions = pipeline.predict(test_df["text"])

    print(f"\nAccuracy : {accuracy_score(test_df['label'], predictions):.2%}")

    print("\nRapport par classe :")
    print(classification_report(
        test_df["label"],
        predictions,
        labels=[-1, 0, 1],
        target_names=["Négatif", "Neutre", "Positif"],
        digits=3,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        test_df["label"], predictions, labels=[-1, 0, 1]
    )
    print("Matrice de confusion :")
    print(pd.DataFrame(
        matrix,
        index=["Vrai négatif", "Vrai neutre", "Vrai positif"],
        columns=["Prédit négatif", "Prédit neutre", "Prédit positif"],
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "baseline_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModèle et vectoriseur enregistrés : {model_path}")


if __name__ == "__main__":
    main()
