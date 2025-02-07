# utils.py

import os
import torch

def load_pt_file_from_s3(s3_client, bucket_name, file_key):
    """
    Load a .pt file from S3, flatten the embeddings, and return as a NumPy array.
    """
    local_file = f"/tmp/{os.path.basename(file_key)}"
    s3_client.download_file(bucket_name, file_key, local_file)
    
    # Load and flatten embeddings
    embeddings = torch.load(local_file).numpy()  # Shape: (N, 256, 4400)
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Shape: (N, 256*4400)
    
    # Delete the temporary file
    os.remove(local_file)
    return flattened_embeddings