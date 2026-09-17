import pandas as pd
import subprocess
from pathlib import Path
import os

# ---- Config --------------------------------------------------------------
FASTA_PATH = "My_Data/mmseq2_processed_data/_test_proteins.fasta"
CLUSTER_OUT_PREFIX = "My_Data/mmseq2_processed_data/_test_clusterRes"
CLUSTER_TMP_DIR = "/home/sahil/mmseqs_tmp"
OUPUT_DIR = Path("My_Data/mmseq2_processed_data/_test")
MMSEQS_BIN = "/home/sahil/mmseqs/bin/mmseqs"  # from `echo ~/mmseqs/bin/mmseqs`
 
MIN_SEQ_ID = 0.3   # sequence identity threshold for clustering (0.0-1.0)
COVERAGE = 0.8     # required alignment coverage
COV_MODE = 1
 
TRAIN_FRAC = 0.6
VALID_FRAC = 0.2   # remainder goes to test
RANDOM_SEED = 42
# ---------------------------------------------------------------------------


def load_protein_df() -> pd.DataFrame:
    '''
    Loads full, pre-split protein data table.
    '''
    aa_df = pd.read_csv(r'My_Data\processed_csvs(all)\aa_df_all_species.csv')
    return aa_df


def write_fasta(aa_df: pd.DataFrame, path: Path = FASTA_PATH) -> None:
    '''
    Writes one FASTA record per unique EntryID.
    '''
    unique_proteins = aa_df[["EntryID", "Sequence"]].drop_duplicates(subset="EntryID")

    with open(path, "w") as f:
        for _, row in unique_proteins.iterrows():
            f.write(f">{row['EntryID']}\n{row['Sequence']}\n")


def run_mmseqs_clustering(
    fasta_path: Path = FASTA_PATH,
    out_prefix: Path = CLUSTER_OUT_PREFIX,
    tmp_dir: Path = CLUSTER_TMP_DIR,
    min_seq_id: float = MIN_SEQ_ID,
    coverage: float = COVERAGE,
    cov_mode: int = COV_MODE,):
    '''
    Runs MMseqs2 clustering inside WSL2 via subprocess
    '''

    subprocess.run(["wsl", "mkdir", "-p", tmp_dir], check=True)

    db_path = f"{out_prefix}_DB"
    clu_path = f"{out_prefix}_DB_clu"
    cluster_tsv = f"{out_prefix}_cluster.tsv"

    subprocess.run(["wsl", MMSEQS_BIN, "createdb", str(fasta_path), db_path], check=True)

    subprocess.run(["wsl", MMSEQS_BIN, "cluster", db_path, clu_path, tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(coverage),
        "--cov-mode", str(cov_mode)], check=True)

    subprocess.run(["wsl", MMSEQS_BIN, "createtsv", db_path, db_path, clu_path, str(cluster_tsv)],
        check=True)

    if not os.path.exists(cluster_tsv):
        raise FileNotFoundError(
            f"Expected {cluster_tsv} after clustering -- check the WSL/MMseqs2 "
            "output above for errors."
        )
    return cluster_tsv


def build_cluster_splits(
    cluster_tsv: Path,
    train_frac: float = TRAIN_FRAC,
    valid_frac: float = VALID_FRAC,
    seed: int = RANDOM_SEED,
) -> dict:
    """
    Assign whole clusters (not individual proteins) to train/valid/test,
    greedily filling each split's target size from a shuffled cluster order.
    A whole cluster always lands entirely on one side of the split.
    """
    clusters = pd.read_csv(
        cluster_tsv, sep="\t", header=None, names=["cluster_rep", "EntryID"]
    )
    cluster_groups = (
        clusters.groupby("cluster_rep")["EntryID"].apply(list).reset_index()
    )
    cluster_groups["size"] = cluster_groups["EntryID"].apply(len)
    cluster_groups = cluster_groups.sample(frac=1, random_state=seed).reset_index(drop=True)
 
    total = cluster_groups["size"].sum()
    train_target = train_frac * total
    valid_target = valid_frac * total
 
    splits = {"train": [], "valid": [], "test": []}
    train_count = valid_count = 0
 
    for _, row in cluster_groups.iterrows():
        if train_count < train_target:
            splits["train"].extend(row["EntryID"])
            train_count += row["size"]
        elif valid_count < valid_target:
            splits["valid"].extend(row["EntryID"])
            valid_count += row["size"]
        else:
            splits["test"].extend(row["EntryID"])
 
    print(
        f"Clusters: {len(cluster_groups):,} | "
        f"Train: {len(splits['train']):,} proteins | "
        f"Valid: {len(splits['valid']):,} proteins | "
        f"Test: {len(splits['test']):,} proteins"
    )
    return splits
 
 
def apply_splits(aa_df: pd.DataFrame, splits: dict) -> dict:
    """Filter the full protein table into train/valid/test DataFrames by EntryID."""
    return {
        name: aa_df[aa_df["EntryID"].isin(ids)].copy()
        for name, ids in splits.items()
    }


def main():
    aa_df = load_protein_df()
    write_fasta(aa_df, FASTA_PATH)
    cluster_tsv = run_mmseqs_clustering()
    splits = build_cluster_splits(cluster_tsv)
    splits_df = apply_splits(aa_df, splits)

    # Sanity check: confirm no EntryID appears in more than one split
    train_ids, valid_ids, test_ids = (set(splits[k]) for k in ("train", "valid", "test"))
    assert not (train_ids & valid_ids), "Leakage: overlap between train and valid!"
    assert not (train_ids & test_ids), "Leakage: overlap between train and test!"
    assert not (valid_ids & test_ids), "Leakage: overlap between valid and test!"
    print("No EntryID overlap between splits -- confirmed.")

    OUPUT_DIR.mkdir(exist_ok=True)
    for name, df in splits_df.items():
        out_path = OUPUT_DIR/ f"{name}_aa_data.parquet"
        df.to_parquet(out_path)
        print(f"Saved {name} split ({len(df):,} rows) to {out_path}")

if __name__ == "__main__":
    main()