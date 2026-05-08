import pandas as pd
import numpy as np
from ast import literal_eval  # Safer string-to-list conversion
import os

CHUNKSIZE = 100_000  # Adjust based on your RAM (100K rows per chunk)
EMBEDDING_COL = "embedding:STRING"  # Column name with embeddings
# DIMENSION = 32  # Update with your embedding dimension
ENTITY_ONLY = True
def parse_embedding(embed_str):
    """Convert embedding string to numpy array"""
    # Remove brackets and convert to list
    return np.array(literal_eval(embed_str), dtype=np.float32)

# Create memory-mapped numpy file
def convert_csv_to_npy(csv_path, npy_path):
    total_embeddings = 0
    # check dir exist, if not then create it
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    
    with open(npy_path, "wb") as f:
        pass  # Initialize empty file

    # Process CSV in chunks
    for chunk_idx, df_chunk in enumerate(
        pd.read_csv(csv_path, chunksize=CHUNKSIZE, usecols=[EMBEDDING_COL])
    ):  
        
        
        # Parse embeddings
        embeddings = np.stack(
            df_chunk[EMBEDDING_COL].apply(parse_embedding).values
        )
        
        # Verify dimensions
        # assert embeddings.shape[1] == DIMENSION, \
        #     f"Dimension mismatch at chunk {chunk_idx}"
        total_embeddings += embeddings.shape[0]
        # Append to .npy file
        with open(npy_path, "ab") as f:
            np.save(f, embeddings.astype(np.float32))
        
        print(f"Processed chunk {chunk_idx} ({CHUNKSIZE*(chunk_idx+1)} rows)")
    print(f"Total number of embeddings: {total_embeddings}")
    print("Conversion complete!")
    
if __name__ == "__main__":
    keyword = 'cc_en'  # Change this to your desired keyword
    csv_dir="./import" # Change this to your CSV directory
    keyword_to_paths ={
        'cc_en':{
            'node_csv': f"{csv_dir}/triple_nodes_cc_en_from_json_2.csv",
            # 'edge_csv': f"{csv_dir}/triple_edges_cc_en_from_json_2.csv",
            'text_csv': f"{csv_dir}/text_nodes_cc_en_from_json_with_emb.csv",
        },
        'pes2o_abstract':{
            'node_csv': f"{csv_dir}/triple_nodes_pes2o_abstract_from_json.csv",
            # 'edge_csv': f"{csv_dir}/triple_edges_pes2o_abstract_from_json.csv",
            'text_csv': f"{csv_dir}/text_nodes_pes2o_abstract_from_json_with_emb.csv",
        },
        'en_simple_wiki_v0':{
            'node_csv': f"{csv_dir}/triple_nodes_en_simple_wiki_v0_from_json.csv",
            # 'edge_csv': f"{csv_dir}/triple_edges_en_simple_wiki_v0_from_json.csv",
            'text_csv': f"{csv_dir}/text_nodes_en_simple_wiki_v0_from_json_with_emb.csv",
        },
    }
    for key, path in keyword_to_paths[keyword].items():
        npy_path = path.replace(".csv", ".npy")
        convert_csv_to_npy(path, npy_path)
        print(f"Converted {path} to {npy_path}")