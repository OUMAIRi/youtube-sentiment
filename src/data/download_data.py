from pathlib import Path

import pandas as pd
import requests


# Retrouver le dossier du projet, quel que soit le dossier de lancement.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "reddit.csv"

URL = (
    "https://raw.githubusercontent.com/Himanshu-1703/"
    "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Téléchargement du dataset...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()

    OUTPUT_PATH.write_bytes(response.content)
    print(f"Fichier enregistré : {OUTPUT_PATH}")

    df = pd.read_csv(OUTPUT_PATH)

    print(f"\nNombre de lignes : {len(df)}")
    print(f"Colonnes : {df.columns.tolist()}")

    print("\nTrois premiers exemples :")
    print(df.head(3).to_string(index=False))

    print("\nValeurs manquantes par colonne :")
    print(df.isna().sum())


if __name__ == "__main__":
    main()