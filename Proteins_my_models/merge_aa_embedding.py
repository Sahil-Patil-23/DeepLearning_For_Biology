import pandas as pd
import glob
import os
from tqdm.auto import tqdm

# Update this to your actual folder
embeddings_dir = "protein_embeddings/"
output_dir = "processed_data_final_merge/"
os.makedirs(output_dir, exist_ok=True)

# The suffix the library added to your files
model_suffix = "facebook_esm2_t30_150M_UR50D"

def merge_to_master(split_name):
    pattern = os.path.join(embeddings_dir, f"{split_name}_chunk_*_{model_suffix}.feather")
    files = sorted(glob.glob(pattern))
    
    if not files: return
    
    print(f"🔗 Consolidating {len(files)} chunks for {split_name}...")
    # Load all chunks into a list and concat
    df = pd.concat([pd.read_feather(f) for f in tqdm(files)], ignore_index=True)
    
    # Save as one master file
    save_path = os.path.join(output_dir, f"{split_name}_master.parquet")
    df.to_parquet(save_path)
    print(f"✅ Created {save_path} (Shape: {df.shape})")
    del df # Free memory immediately

for s in ["train", "valid", "test"]:
    merge_to_master(s)