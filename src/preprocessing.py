"""Chargement et pretraitement du corpus d'emails Enron.

Pipeline :
  1. Lecture du CSV Kaggle (`emails.csv`, colonnes `file`, `message`).
  2. Parsing RFC822 -> extraction du sujet et du corps principal.
  3. Nettoyage (suppression des en-tetes de transfert, lignes citees, etc.).
  4. Filtrage des emails vides/trop courts + deduplication.
  5. Echantillonnage a MAX_EMAILS et sauvegarde en parquet.

Le CSV Enron est resolu via download_data.resolve_enron_csv (data/emails.csv
en priorite, sinon telechargement kagglehub).
"""
from __future__ import annotations

import email
import re
from email import policy
from typing import Optional

import pandas as pd

import config

# --------------------------------------------------------------------------
# Expressions regulieres de nettoyage
# --------------------------------------------------------------------------
_RE_FORWARD_HEADER = re.compile(
    r"-{2,}.*?(forwarded|original message).*?-{2,}", re.IGNORECASE | re.DOTALL
)
# Lignes d'en-tete residuelles (To:, From:, Sent:, Cc:, Subject:...)
_RE_HEADER_LINE = re.compile(
    r"^\s*(to|from|sent|cc|bcc|subject|date|re|fw|fwd)\s*:.*$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_QUOTED = re.compile(r"^\s*>.*$", re.MULTILINE)        # lignes citees ">"
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL_ADDR = re.compile(r"\S+@\S+\.\S+")
_RE_MULTISPACE = re.compile(r"[ \t]+")
_RE_MULTINEWLINE = re.compile(r"\n{3,}")


def extract_subject_body(raw_message: str) -> tuple[str, str]:
    """Parse un message RFC822 et renvoie (sujet, corps texte brut)."""
    try:
        msg = email.message_from_string(raw_message, policy=policy.default)
    except Exception:
        return "", raw_message

    subject = msg.get("subject", "") or ""

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    body = part.get_content()
                except Exception:
                    body = part.get_payload(decode=False) or ""
                break
    else:
        try:
            body = msg.get_content()
        except Exception:
            body = msg.get_payload(decode=False) or ""

    if not isinstance(body, str):
        body = str(body)
    return str(subject), body


def clean_text(text: str) -> str:
    """Nettoie le corps d'un email (transferts, citations, urls, espaces)."""
    if not text:
        return ""
    text = _RE_FORWARD_HEADER.sub(" ", text)
    text = _RE_QUOTED.sub(" ", text)
    text = _RE_HEADER_LINE.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL_ADDR.sub(" ", text)
    text = _RE_MULTISPACE.sub(" ", text)
    text = _RE_MULTINEWLINE.sub("\n\n", text)
    return text.strip()


def build_corpus(
    raw_csv: Optional[str] = None,
    max_emails: int = config.MAX_EMAILS,
) -> pd.DataFrame:
    """Construit le corpus pretraite a partir du CSV Enron.

    Renvoie un DataFrame avec les colonnes :
        doc_id, subject, body, text (sujet + corps, sert a l'indexation).
    """
    import download_data

    csv_path = raw_csv or download_data.resolve_enron_csv()
    df = pd.read_csv(csv_path)

    if "message" not in df.columns:
        raise ValueError(
            f"Colonne 'message' absente de {csv_path}. "
            "Le CSV Kaggle Enron doit contenir les colonnes 'file' et 'message'."
        )

    # Echantillonnage AVANT le parsing couteux pour gagner du temps.
    if max_emails and len(df) > max_emails:
        # On sur-echantillonne (x3) car beaucoup seront filtres ensuite.
        df = df.sample(
            n=min(len(df), max_emails * 3), random_state=config.RANDOM_SEED
        ).reset_index(drop=True)

    records = []
    for raw in df["message"].astype(str):
        subject, body = extract_subject_body(raw)
        body = clean_text(body)
        subject = clean_text(subject)
        if len(body) < config.MIN_BODY_CHARS:
            continue
        body = body[: config.MAX_BODY_CHARS]
        text = (subject + ". " + body).strip()
        records.append({"subject": subject, "body": body, "text": text})

    corpus = pd.DataFrame(records)

    # Deduplication (threads, forwards multiples du meme contenu)
    corpus = corpus.drop_duplicates(subset="text").reset_index(drop=True)

    if max_emails and len(corpus) > max_emails:
        corpus = corpus.head(max_emails).reset_index(drop=True)

    corpus.insert(0, "doc_id", range(len(corpus)))
    return corpus


def load_or_build_corpus(force: bool = False) -> pd.DataFrame:
    """Charge le corpus depuis le cache parquet ou le reconstruit."""
    if config.CORPUS_PARQUET.exists() and not force:
        return pd.read_parquet(config.CORPUS_PARQUET)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # build_corpus() resout automatiquement la source : data/emails.csv en
    # priorite, sinon telechargement via kagglehub (cf. download_data.py).
    corpus = build_corpus()
    corpus.to_parquet(config.CORPUS_PARQUET, index=False)
    return corpus


if __name__ == "__main__":
    c = load_or_build_corpus(force=True)
    print(f"Corpus construit : {len(c)} emails")
    print(c.head(3)[["doc_id", "subject"]])
