"""
Preprocess higgs-activity_time.txt.gz into a clean Parquet file.

Usage:
    python src/preprocess.py \
        --raw-path data/raw/higgs-activity_time.txt.gz \
        --output-path data/processed/events.parquet \
        [--max-events 1000000] \
        [--max-users 50000]
"""
import argparse
from pathlib import Path

import pandas as pd
from utils import get_logger

log = get_logger(__name__)

INTERACTION_MAP = {"RT": 0, "MT": 1, "RE": 2}


def load_events(raw_path: str, max_events: int | None, max_users: int | None) -> pd.DataFrame:
    log.info(f"Loading {raw_path}")
    df = pd.read_csv(
        raw_path,
        sep=" ",
        names=["src", "dst", "timestamp", "interaction"],
        compression="gzip",
    )
    log.info(f"Loaded {len(df):,} raw events")

    if max_events:
        df = df.head(max_events)
        log.info(f"Capped at {max_events:,} events")

    df["interaction_type"] = df["interaction"].map(INTERACTION_MAP)
    df = df.dropna(subset=["interaction_type"])
    df["interaction_type"] = df["interaction_type"].astype(int)
    df = df.drop(columns=["interaction"])
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    if max_users:
        activity = pd.concat([
            df["src"].value_counts(),
            df["dst"].value_counts(),
        ]).groupby(level=0).sum()
        top_users = set(activity.nlargest(max_users).index.tolist())
        df = df[df["src"].isin(top_users) & df["dst"].isin(top_users)].copy()
        log.info(f"Filtered to {max_users:,} most active users: {len(df):,} events remain")

        # Remap IDs to contiguous 0-indexed integers
        all_users = sorted(set(df["src"].tolist()) | set(df["dst"].tolist()))
        id_map = {u: i for i, u in enumerate(all_users)}
        df["src"] = df["src"].map(id_map)
        df["dst"] = df["dst"].map(id_map)
        log.info(f"Remapped {len(all_users):,} user IDs to [0, {len(all_users)-1}]")

    df = df.sort_values("timestamp").reset_index(drop=True)
    log.info(
        f"Final dataset: {len(df):,} events, "
        f"{pd.concat([df['src'], df['dst']]).nunique():,} unique users, "
        f"time range [{df['timestamp'].min()}, {df['timestamp'].max()}]"
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess Higgs Twitter events")
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    args = parser.parse_args()

    df = load_events(args.raw_path, args.max_events, args.max_users)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    log.info(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
