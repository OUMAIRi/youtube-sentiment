from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    df = pd.read_csv(DATA_DIR / "reddit_clean.csv")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    # Vérifier qu'aucun texte n'apparaît dans les deux ensembles.
    overlap = set(train_df["text"]) & set(test_df["text"])
    if overlap:
        raise ValueError("Des commentaires sont présents dans les deux ensembles.")

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Entraînement : {len(train_df)} commentaires")
    print(f"Test : {len(test_df)} commentaires")

    distribution = pd.DataFrame({
        "Train (%)": train_df["label"].value_counts(normalize=True) * 100,
        "Test (%)": test_df["label"].value_counts(normalize=True) * 100,
    }).sort_index()

    print("\nRépartition des sentiments :")
    print(distribution.round(2))
    print("\nDécoupage enregistré : train.csv et test.csv")


if __name__ == "__main__":
    main()