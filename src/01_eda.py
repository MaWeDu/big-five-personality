"""Download, clean, and save the personality dataset."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

FILE_ID = "1lgycXDQapixkoWFxIg48qaYcrNl6uSZM"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "data.csv"
CLEAN_FILE = DATA_DIR / "data_clean.csv"
MIN_AGE, MAX_AGE, REFERENCE_YEAR = 13, 100, 2017
MIN_BIRTH_YEAR = REFERENCE_YEAR - MAX_AGE + 1
MAX_BIRTH_YEAR = REFERENCE_YEAR - MIN_AGE
QUESTION_COLUMNS = [
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10",
    "E1", "E3", "E4", "E5", "E7", "E9", "E10", "C4", "A4",
]


def dataset_is_valid(file_path):
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return False
    try:
        pd.read_csv(file_path, nrows=5)
        return True
    except (pd.errors.ParserError, UnicodeDecodeError, OSError):
        return False


def download_dataset():
    if dataset_is_valid(DATA_FILE):
        print("Dataset found.")
        return
    try:
        import gdown
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "gdown"], check=True, timeout=120)
        import gdown
    temporary_file = DATA_DIR / "data_download.csv"
    print("Downloading dataset...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", output=str(temporary_file), quiet=True)
    if not dataset_is_valid(temporary_file):
        raise ValueError("The downloaded file is not a valid CSV file.")
    temporary_file.replace(DATA_FILE)
    print("Dataset downloaded.")


def missing_or_zero(series):
    normalized = series.astype("string").str.strip().str.lower()
    return series.isna() | normalized.isin({"0", "none", "null", "nan", ""})


DATA_DIR.mkdir(exist_ok=True)
download_dataset()
df = pd.read_csv(DATA_FILE)
age_raw = df["age"]

birth_year = age_raw.between(MIN_BIRTH_YEAR, MAX_BIRTH_YEAR)
three_digit_age = age_raw.between(MAX_AGE + 1, 999)
missing_age = age_raw.isna()
valid_age = age_raw.between(MIN_AGE, MAX_AGE)
invalid_age = ~(valid_age | birth_year) & ~missing_age
age_clean = age_raw.copy()
age_clean.loc[birth_year] = REFERENCE_YEAR - age_raw.loc[birth_year]

criteria = {
    "missing age": missing_age,
    "three-digit age": three_digit_age,
    "invalid age": invalid_age,
    "missing gender": missing_or_zero(df["gender"]),
    "missing hand": missing_or_zero(df["hand"]),
    "zero questionnaire answer": (df[QUESTION_COLUMNS] == 0).any(axis=1),
}
keep = pd.Series(True, index=df.index)
for criterion in criteria.values():
    keep &= ~criterion

df_clean = df.loc[keep].copy()
df_clean["age"] = age_clean.loc[keep].astype("Int64")
df_clean.to_csv(CLEAN_FILE, index=False)

print(f"Raw data: {len(df):,} rows, {df.shape[1]} columns")
print(f"Converted birth years: {int(birth_year.sum()):,}")
print(f"Removed rows: {len(df) - len(df_clean):,}")
print(f"Clean data: {len(df_clean):,} rows, {df_clean.shape[1]} columns")
print(f"Saved: {CLEAN_FILE}")
