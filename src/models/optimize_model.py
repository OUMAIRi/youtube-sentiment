from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    train_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "train.csv",
        keep_default_na=False,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("classifier", LogisticRegression()),
    ])

    parameters = [
        {
            "classifier": [
                LogisticRegression(max_iter=1000, random_state=42)
            ],
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__class_weight": [None, "balanced"],
        },
        {
            "classifier": [MultinomialNB()],
            "classifier__alpha": [0.5, 1.0],
        },
    ]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=2,
        verbose=1,
        error_score="raise",
    )

    print("Comparaison et optimisation en cours...")
    search.fit(train_df["text"], train_df["label"])

    print(f"\nMeilleur F1 macro en validation : {search.best_score_:.3f}")
    print("\nMeilleurs paramètres :")
    for name, value in search.best_params_.items():
        print(f"{name} : {value}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    best_pipeline = search.best_estimator_

    joblib.dump(best_pipeline, MODEL_DIR / "best_pipeline.joblib")
    joblib.dump(
        best_pipeline.named_steps["tfidf"],
        MODEL_DIR / "vectorizer.joblib",
    )
    joblib.dump(
        best_pipeline.named_steps["classifier"],
        MODEL_DIR / "model.joblib",
    )

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(REPORT_DIR / "grid_search_results.csv", index=False)

    print("\nComparaison des configurations :")
    print(results[
        ["rank_test_score", "mean_test_score", "params"]
    ].sort_values("rank_test_score").to_string(index=False))

    print("\nMeilleur modèle enregistré dans models/best_pipeline.joblib")


if __name__ == "__main__":
    main()