from pathlib import Path

import numpy as np
import pandas as pd
from source.analysis.base import BaseDataLoader
from source.core import logger

CSV_DATA_PATH = Path("/home/janezla/data/crp")


def load_dataframe(date: str, gerk: str) -> pd.DataFrame:
    csv_files = list(CSV_DATA_PATH.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_DATA_PATH}")

    data_frames = []
    for file in csv_files:
        df = pd.read_csv(file)
        df["date"] = file.stem
        df[["gerk", "id", "st"]] = df["GERK_ID_St"].str.split("_", expand=True)
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True)

    # Remove Band_X_NIR (X: 1-5 and 156-160)
    # Remove Band_X_SWIR (X: 1-5 and 284-288)
    # Remove Band_X_SWIR (X: 75-91 and 158-187)
    cols_to_remove = (
        [f"Band_{i}_NIR" for i in range(1, 6)]
        + [f"Band_{i}_NIR" for i in range(141, 161)]
        + [f"Band_{i}_SWIR" for i in range(1, 6)]
        + [f"Band_{i}_SWIR" for i in range(281, 289)]
        + [f"Band_{i}_SWIR" for i in range(75, 92)]
        + [f"Band_{i}_SWIR" for i in range(158, 188)]
    )
    combined_df.drop(columns=cols_to_remove, inplace=True)
    combined_df.dropna(inplace=True)
    # Remove negative values
    combined_df = combined_df[
        (combined_df.drop(columns=["date", "GERK_ID_St", "gerk", "id", "st"]) >= 0).all(
            axis=1
        )
    ]
    # Remove date with zero values
    # combined_df = combined_df[combined_df["date"] != "2023_09_20"]
    combined_df = combined_df[combined_df["date"] == date]
    combined_df = combined_df[combined_df["gerk"] == gerk]
    return combined_df


def load_data_balanced(date: str, gerk: str) -> tuple[np.ndarray, np.ndarray]:
    combined_df = load_dataframe(date, gerk)
    # Create date_gerk_st grouping column
    combined_df["date_gerk_st"] = combined_df[["date", "gerk", "st"]].apply(
        lambda x: f"{x['date']}_{x['gerk']}_{x['st']}", axis=1
    )
    # Get unique groups and their counts
    groups, counts = np.unique(combined_df["date_gerk_st"], return_counts=True)

    # Balance classes by sampling equal amounts from each group
    min_count = np.min(counts)
    balanced_dfs = []

    for group in groups:
        group_df = combined_df[combined_df["date_gerk_st"] == group]
        # Sample min_count samples from each group (or all if less than min_count)
        sample_size = min(len(group_df), min_count)
        sampled_df = group_df.sample(n=sample_size, random_state=42)
        balanced_dfs.append(sampled_df)

    # Combine balanced samples
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Extract features and labels
    X = balanced_df.drop(
        columns=["date", "GERK_ID_St", "gerk", "id", "st", "date_gerk_st"]
    ).values
    y = np.array(balanced_df["st"].values)
    logger.info(
        f"Balanced dataset: {len(X)} samples from {len(groups)} groups (min count per group: {min_count})"
    )
    return X, y


class CrpLoader2023_06_26_1528147(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="1528147")
        return self._shuffle_data(X, y)


class CrpLoader2023_06_26_174223(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="174223")
        return self._shuffle_data(X, y)


class CrpLoader2023_06_26_2119192(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="2119192")
        return self._shuffle_data(X, y)


class CrpLoader2023_06_26_3266572(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="3266572")
        return self._shuffle_data(X, y)


class CrpLoader2023_06_26_4606283(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="4606283")
        return self._shuffle_data(X, y)


class CrpLoader2023_06_26_5099541(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="5099541")
        return self._shuffle_data(X, y)


class CrpLoader2023_06_26_6006(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_06_26", gerk="6006")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_1528147(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="1528147")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_174223(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="174223")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_2119192(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="2119192")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_3266572(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="3266572")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_4606283(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="4606283")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_5099541(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="5099541")
        return self._shuffle_data(X, y)


class CrpLoader2023_09_20_6006(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2023_09_20", gerk="6006")
        return self._shuffle_data(X, y)


class CrpLoader2024_07_10_1528147(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_07_10", gerk="1528147")
        return self._shuffle_data(X, y)


class CrpLoader2024_07_10_2119192(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_07_10", gerk="2119192")
        return self._shuffle_data(X, y)


class CrpLoader2024_07_10_4606283(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_07_10", gerk="4606283")
        return self._shuffle_data(X, y)


class CrpLoader2024_07_10_5099541(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_07_10", gerk="5099541")
        return self._shuffle_data(X, y)


class CrpLoader2024_07_10_6006(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_07_10", gerk="6006")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_1528147(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="1528147")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_174223(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="174223")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_2119192(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="2119192")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_3266572(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="3266572")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_4606283(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="4606283")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_5099541(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="5099541")
        return self._shuffle_data(X, y)


class CrpLoader2024_09_23_6006(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = load_data_balanced(date="2024_09_23", gerk="6006")
        return self._shuffle_data(X, y)
