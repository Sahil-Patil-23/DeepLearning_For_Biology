from dlfb.proteins.dataset import store_sequence_embeddings
import dlfb.proteins.dataset
from transformers import AutoTokenizer, EsmModel
from dlfb.utils.context import assets
import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig


# def esmc_get_mean_embeddings(batch_seqs, tokenizer, model, device):
#     """Replacement function for the book's ESM-2 embedding extractor"""

#     # 1. Convert raw string sequence to ESMProtein object
#     proteins = [ESMProtein(sequence=seq) for seq in batch_seqs]
        
#     # 2. Encode protein structure/tokens using ESMC model
#     protein_tensor = model.encode(proteins).to(device)
        
#     # 3. Perform forward pass requesting embeddings
#     with torch.no_grad():
#         logits_output = model.logits(
#             protein_tensor, 
#             LogitsConfig(sequence=True, return_embeddings=True)
#         )
            
#     # 4. Extract embeddings tensor
#     # Shape: (1, sequence_length, embedding_dim)
#     embeddings = logits_output.embeddings 

#     mean_embeddings = []
        
#     # 5. Extract true sequence lengths to accurately average (ignoring padding tokens)
#     for idx, protein in enumerate(proteins):
#         seq_len = len(protein.sequence)
            
#         # Slice from index 1 (skip <bos>) up to seq_len + 1 (skip <eos> and padding)
#         individual_emb = embeddings[idx, 1 : seq_len + 1, :]
            
#         # Compute mean across the sequence length axis
#         mean_emb = individual_emb.mean(dim=0)
            
#         # Convert to CPU numpy array for the book's framework
#         mean_embeddings.append(mean_emb.cpu().numpy())
            
#     return mean_embeddings  

def esmc_get_mean_embeddings(batch_seqs, tokenizer, model, device):
    """Replacement function for the book's ESM-2 embedding extractor.
    Processes sequences individually to strictly match the ESMC SDK design.
    """
    mean_embeddings = []
    
    for seq in batch_seqs:
        # 1. Convert single raw string sequence to an ESMProtein object
        protein = ESMProtein(sequence=seq)
        
        # 2. Encode single protein structure using the ESMC model
        protein_tensor = model.encode(protein).to(device)
        
        # 3. Perform forward pass requesting embeddings
        with torch.no_grad():
            logits_output = model.logits(
                protein_tensor, 
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            
        # 4. Extract embeddings tensor (Shape: 1, sequence_length, 960)
        embeddings = logits_output.embeddings 
        
        # 5. Average across sequence length (ignoring special start <bos> and end <eos> tokens)
        # Slicing [:, 1:-1, :] cleanly strips out control tokens without padding issues
        mean_emb = embeddings[:, 1:-1, :].mean(dim=1)
        
        # 6. Flatten and append as a CPU numpy array for the book's framework
        mean_embeddings.append(mean_emb.squeeze(0).cpu().numpy())
        
    return mean_embeddings


dlfb.proteins.dataset.get_mean_embeddings = esmc_get_mean_embeddings


print("Loading ESMC-300M Model...")

# Ensure model is ready on Mac GPU
device = torch.device("mps")
# device = "cpu"
model_name = "esmc_300m"
model_client = ESMC.from_pretrained(model_name=model_name).to(device)
model_client.name_or_path = "esmc_300m"
tokenizer = model_client.tokenizer

# Directories of importance
csv_dir = "processed_csvs(all)/"
embeddings_dir = "protein_embeddings(all)/"
os.makedirs(embeddings_dir, exist_ok=True)

CHUNK_SIZE = 5000

for split in ["train", "valid", "test"]:

    file_path = os.path.join(csv_dir, f"{split}_sequenced_df.csv")

    if not os.path.exists(file_path):
        print(f"Skipping {split} data, file not found")
        continue

    df = pd.read_csv(file_path, chunksize=CHUNK_SIZE)
    # num_chunks = int(np.ceil(len(df) / CHUNK_SIZE))

    print(f"🎬 Processing {split} ...")


    for i, chunk_df in enumerate(df):
        chunk_prefix = os.path.join(embeddings_dir, f"{split}_chunk_{i:04d}")

        # Construct the ACTUAL filename the library creates
        # It adds an underscore, the model name, and the extension
        actual_file_path = f"{chunk_prefix}_{model_name}.feather"
        
        # SKIP if files already exist (Resume Logic) AND is not an empty file
        if os.path.exists(actual_file_path) and os.path.getsize(actual_file_path) > 0:
            print(f"Skipping chunk #{i} as it has been processed before!")
            continue 
        
        # Generate and Store
        store_sequence_embeddings(
            sequence_df=chunk_df,
            store_prefix=chunk_prefix,
            tokenizer=tokenizer,
            model=model_client,
        )
        
        if device == "mps":
            torch.mps.empty_cache()

print("🏁 ALL DONE! Your Mac deserves a break.")