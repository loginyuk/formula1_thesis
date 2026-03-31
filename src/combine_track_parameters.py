import pandas as pd

COMPOUND_HARDNESS_2022 = {
    "C1": 0,  # Ultra Hard
    "C2": 2,  # Medium-Hard
    "C3": 3,  # Medium
    "C4": 4,  # Soft
    "C5": 5,  # Extra Soft
}

COMPOUND_HARDNESS_2023_2024 = {
    "C0": 0,  # Ultra Hard (never in races)
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
    2022: "data/track_parameters_years/track_parameters_2022.csv",
    2023: "data/track_parameters_years/track_parameters_2023.csv",
    2024: "data/track_parameters_years/track_parameters_2024.csv",
    2025: "data/track_parameters_years/track_parameters_2025.csv",
}

OUTPUT_FILE = "data/track_parameters.csv"


def load_and_annotate(year: int, path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Year"] = year

    mapping = HARDNESS_BY_YEAR[year]
    for role in ("Hard", "Medium", "Soft"):
        col = f"Compound_{role}"
        df[f"{col}_Hardness"] = df[col].map(mapping)
        missing = df[df[f"{col}_Hardness"].isna()][col].unique()
        if len(missing):
            print(f"WARNING ({year}): unmapped compounds in {col}: {missing}")

    return df


if __name__ == "__main__":
    frames = []
    for year, path in FILES.items():
        df = load_and_annotate(year, path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    cols = combined.columns.tolist()
    cols.remove("Year")
    loc_idx = cols.index("Location")
    cols.insert(loc_idx + 1, "Year")
    combined = combined[cols]

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(combined)} rows to {OUTPUT_FILE}")

