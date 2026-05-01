from dlfb.proteins.dataset import store_sequence_embeddings

from transformers import AutoTokenizer, EsmModel
from dlfb.utils.context import assets
import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

# Load in the model
model_checkpoint = "facebook/esm2_t30_150M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = EsmModel.from_pretrained(model_checkpoint)

# Ensure model is ready on Mac GPU
device = torch.device("mps")
model.to(device).eval()

# Directories of importance
csv_dir = "processed_csvs/"
embeddings_dir = "protein_embeddings/"
os.makedirs(embeddings_dir, exist_ok=True)

CHUNK_SIZE = 5000

for split in ["train", "valid", "test"]:

    file_path = os.path.join(csv_dir, f"{split}_sequenced_df.csv")

    if not os.path.exists(file_path):
        print(f"Skipping {split} data, file not found")

    df = pd.read_csv(file_path)
    num_chunks = int(np.ceil(len(df) / CHUNK_SIZE))

    print(f"🎬 Processing {split} ({len(df)} total sequences)...")

    # Create a safe string of the model name for the filename
    model_name_suffix = model_checkpoint.replace("/", "_")

    for i in tqdm(range(num_chunks), desc=f"Chunks of {split}"):
        chunk_prefix = os.path.join(embeddings_dir, f"{split}_chunk_{i:04d}")

        # Construct the ACTUAL filename the library creates
        # It adds an underscore, the model name, and the extension
        actual_file_path = f"{chunk_prefix}_{model_name_suffix}.feather"
        
        # SKIP if files already exist (Resume Logic) AND is not an empty file
        if os.path.exists(actual_file_path) and os.path.getsize(actual_file_path) > 0:
            print(f"Skipping chunk #{i} as it has been processed before!")
            continue 

        # Slice the dataframe
        start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, len(df))
        chunk_df = df.iloc[start:end]
        
        # Generate and Store
        store_sequence_embeddings(
            sequence_df=chunk_df,
            store_prefix=chunk_prefix,
            tokenizer=tokenizer,
            model=model,
        )
        
    # Periodic Memory Flush
    del df
    torch.mps.empty_cache()

print("🏁 ALL DONE! Your Mac deserves a break.")