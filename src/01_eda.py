from pathlib import Path
import gdown
import numpy as np
import pandas as pd

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    FILE_ID = "1lgycXDQapixkoWFxIg48qaYcrNl6uSZM"
    DATA_DIR = BASE_DIR / "DATA"
    DATA_DIR.mkdir(exist_ok=True)
    DATA_FILE = DATA_DIR / "data.csv"

    print("=== EDA & Data Cleaning ===")
    if DATA_FILE.exists():
        print("✓ Dataset already available.")
    else:
        print("Downloading dataset from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, output=str(DATA_FILE), quiet=False)
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Download failed. File not found: {DATA_FILE}")
        print("✓ Download complete.")

    df = pd.read_csv(DATA_FILE)

    MIN_AGE = 1
    MAX_AGE = 100
    REFERENCE_YEAR = 2017
    MIN_BIRTH_YEAR = REFERENCE_YEAR - MAX_AGE + 1

    question_columns = [
        'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10',
        'E1', 'E3', 'E4', 'E5', 'E7', 'E9', 'E10', 'C4', 'A4'
    ]

    age_num = pd.to_numeric(df['age'], errors='coerce')
    missing_age = age_num.isna()
    three_digit_age = age_num.between(MAX_AGE + 1, 999, inclusive="both")
    valid_age = age_num.between(MIN_AGE, MAX_AGE, inclusive="both")
    birth_year = age_num.between(MIN_BIRTH_YEAR, REFERENCE_YEAR, inclusive='both')
    invalid_age = ~(birth_year | valid_age) & ~missing_age

    age_clean = age_num.copy()
    age_clean.loc[birth_year] = REFERENCE_YEAR - age_num.loc[birth_year]

    def missing_or_zero(series):
        normalized = series.astype('string').str.strip().str.lower()
        return series.isna() | normalized.isin({'0', 'none', 'null', 'nan', ''})

    missing_gender = missing_or_zero(df['gender'])
    missing_hand = missing_or_zero(df['hand'])

    question_zero = pd.Series(False, index=df.index)
    for col in question_columns:
        if col in df.columns:
            question_zero |= pd.to_numeric(df[col], errors='coerce').eq(0)

    df_work = df.copy()
    df_work["age"] = age_clean

    keep = ~(missing_age | three_digit_age | invalid_age | missing_gender | missing_hand | question_zero)

    df_clean = df_work.loc[keep].copy()
    df_clean["age"] = df_clean["age"].astype("Int64")

    cols = [c for c in df_clean.columns if c != "age"]
    cols.insert(cols.index("gender"), "age")
    df_clean = df_clean[cols]

    print(f"Initial rows: {len(df):,} | Cleaned rows: {len(df_clean):,}")
    
    clean_file = DATA_DIR / "data_clean.csv"
    df_clean.to_csv(clean_file, index=False)
    print(f"✓ Clean dataset saved successfully to '{clean_file}'.")

if __name__ == "__main__":
    main()