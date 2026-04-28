import os
import pandas as pd

from src.config import DATA_RAW_DIR

COMPOUND_HARDNESS_2022 = {
    "C1": 0,  # Hard
    "C2": 2,  # Medium-Hard
    "C3": 3,  # Medium
    "C4": 4,  # Soft
    "C5": 5,  # Extra Soft
}

COMPOUND_HARDNESS_2023_2024 = {
    "C0": 0,  # Ultra Hard (never used in races, old C1 renamed)
    "C1": 1,  # Hard
    "C2": 2,  # Medium-Hard
    "C3": 3,  # Medium
    "C4": 4,  # Soft
    "C5": 5,  # Extra Soft
}

COMPOUND_HARDNESS_2025 = {
    "C1": 1,  # Hard
    "C2": 2,  # Medium-Hard
    "C3": 3,  # Medium
    "C4": 4,  # Soft
    "C5": 5,  # Extra Soft
    "C6": 6,  # Hyper Soft
}

HARDNESS_BY_YEAR = {
    2022: COMPOUND_HARDNESS_2022,
    2023: COMPOUND_HARDNESS_2023_2024,
    2024: COMPOUND_HARDNESS_2023_2024,
    2025: COMPOUND_HARDNESS_2025,
}

FILES = {
    year: os.path.join(DATA_RAW_DIR, "track_parameters_years", f"track_parameters_{year}.csv")
    for year in (2022, 2023, 2024, 2025)
}

OUTPUT_FILE = os.path.join(DATA_RAW_DIR, "track_parameters.csv")


def load_files(year: int, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Year"] = year

    mapping = HARDNESS_BY_YEAR[year]
    for type_c in ("Hard", "Medium", "Soft"):
        col = f"Compound_{type_c}"
        df[f"{col}_Hardness"] = df[col].map(mapping)
        missing = df[df[f"{col}_Hardness"].isna()][col].unique()
        if len(missing):
            print(f"Unmapped compounds in year {year} for {col}: {missing}")

    return df
