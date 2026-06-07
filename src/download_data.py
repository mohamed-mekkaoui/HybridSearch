"""Telechargement du dataset Enron via kagglehub.

kagglehub telecharge et met en cache le dataset (~1.4 Go) dans
~/.cache/kagglehub/. On NE recopie PAS le fichier dans data/ : on pointe
directement vers le `emails.csv` du cache pour economiser l'espace disque.

Pre-requis : authentification Kaggle. kagglehub utilise soit :
  - le fichier ~/.kaggle/kaggle.json (token API Kaggle), soit
  - les variables d'environnement KAGGLE_USERNAME / KAGGLE_KEY.

Usage direct :
    python src/download_data.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import config

KAGGLE_DATASET = "wcukierski/enron-email-dataset"


def download_enron() -> Path:
    """Telecharge le dataset Enron et renvoie le chemin du dossier en cache."""
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    return Path(path)


def find_emails_csv(dataset_dir: Path) -> Optional[Path]:
    """Localise emails.csv dans le dossier telecharge."""
    direct = dataset_dir / "emails.csv"
    if direct.exists():
        return direct
    matches = list(dataset_dir.rglob("emails.csv"))
    return matches[0] if matches else None


def resolve_enron_csv() -> Path:
    """Renvoie le chemin du emails.csv a utiliser.

    Ordre de priorite :
      1. data/emails.csv (place manuellement)
      2. cache kagglehub (telechargement automatique si kagglehub est installe)
    """
    if config.RAW_ENRON_CSV.exists():
        return config.RAW_ENRON_CSV

    try:
        dataset_dir = download_enron()
    except ImportError as exc:
        raise FileNotFoundError(
            "emails.csv introuvable dans data/ et kagglehub n'est pas installe.\n"
            "Installez-le (`pip install kagglehub`) ou placez emails.csv dans data/."
        ) from exc

    csv_path = find_emails_csv(dataset_dir)
    if csv_path is None:
        raise FileNotFoundError(
            f"emails.csv introuvable dans le dataset telecharge ({dataset_dir})."
        )
    return csv_path


if __name__ == "__main__":
    path = resolve_enron_csv()
    print("Path to dataset file:", path)
