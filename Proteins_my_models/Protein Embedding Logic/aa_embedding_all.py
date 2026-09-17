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
            
        # 4. Extract embeddings tensor (Shape: 1, sequence_length, 1152)
        embeddings = logits_output.embeddings 
        
        # 5. Average across sequence length (ignoring special start <bos> and end <eos> tokens)
        # Slicing [:, 1:-1, :] cleanly strips out control tokens without padding issues
        mean_emb = embeddings[:, 1:-1, :].mean(dim=1)
        
        # 6. Flatten and append as a CPU numpy array for the book's framework
        mean_embeddings.append(mean_emb.squeeze(0).cpu().numpy())
        
    return mean_embeddings


dlfb.proteins.dataset.get_mean_embeddings = esmc_get_mean_embeddings


print("Loading ESMC-600M Model...")

# Ensure model is ready on Nvidia RTX
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "esmc_600m"
model_client = ESMC.from_pretrained(model_name=model_name).to(device)
model_client.name_or_path = "esmc_600m"
tokenizer = model_client.tokenizer

# Directories of importance
data_dir = "../mmseq2_processed_data/"
embeddings_dir = "../protein_embeddings(all)/"
os.makedirs(embeddings_dir, exist_ok=True)

CHUNK_SIZE = 5000

for split in ["train", "valid", "test"]:

    file_path = os.path.join(data_dir, f"{split}_aa_data.parquet")

    if not os.path.exists(file_path):
        print(f"Skipping {split} data, file not found")
        continue


    df = pd.read_parquet(file_path)
    df = df.drop_duplicates(subset="EntryID").reset_index(drop=True)
    num_chunks = int(np.ceil(len(df) / CHUNK_SIZE))

    print(f"🎬 Processing {split}")


    for i in tqdm(range(num_chunks), desc=f"Chunks of {split}"):
        chunk_prefix = os.path.join(embeddings_dir, f"{split}_chunk_{i:04d}")

        # Construct the ACTUAL filename the library creates
        # It adds an underscore, the model name, and the extension
        actual_file_path = f"{chunk_prefix}_{model_name}.feather"
        
        # SKIP if files already exist (Resume Logic) AND is not an empty file
        if os.path.exists(actual_file_path) and os.path.getsize(actual_file_path) > 0:
            print(f"Skipping chunk #{i} as it has been processed before!")
            continue 

        start, end = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, len(df))
        chunk_df = df.iloc[start : end ]
        
        # Generate and Store
        store_sequence_embeddings(
            sequence_df=chunk_df,
            store_prefix=chunk_prefix,
            tokenizer=tokenizer,
            model=model_client,
        )
        
        if device == "cuda":
            torch.cuda.empty_cache()

print("🏁 ALL DONE! Your machine deserves a break.")