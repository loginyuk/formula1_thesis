"""
run_combine_tracks.py
─────────────────────
Merges per-year Pirelli track parameter CSVs into a single combined file.

Run from project root:
    python scripts/run_combine_tracks.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preparation.combine_tracks import FILES, OUTPUT_FILE, load_and_annotate

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

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(combined)} rows to {OUTPUT_FILE}")
