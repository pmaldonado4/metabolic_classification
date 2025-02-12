# %%
import os
import torch
import boto3
import pandas as pd
from io import StringIO
from botocore.config import Config

# %%
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
read_access_key = os.getenv("AWS_ACCESS_KEY_ID")
read_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("AWS_BUCKET_NAME")
endpoint_url = os.getenv("AWS_ENDPOINT_URL")

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=read_access_key,
    aws_secret_access_key=read_secret_key,
    endpoint_url=endpoint_url,
    config=Config(signature_version="s3v4"),
)

# %%
def load_and_process_embedding(bucket_name, file_key):
    """
    Load a .pt file from S3, apply mean pooling, and return as a NumPy array.
    """
    print(f"Downloading {file_key}...")
    local_file = f"/tmp/{os.path.basename(file_key)}"
    s3.download_file(bucket_name, file_key, local_file)

    # Load embeddings (N, 256, 4400) with weights_only=True to avoid warnings
    embeddings = torch.load(local_file, weights_only=True).numpy()
    
    # Apply mean pooling across the sequence length (axis=2) -> (N, 256)
    pooled_embeddings = embeddings.mean(axis=2)
    
    print(f"Processed {file_key}: shape {pooled_embeddings.shape}")
    
    # Delete local file to free disk space
    os.remove(local_file)
    
    return pooled_embeddings

# %%
def save_embeddings_to_s3(embeddings, file_name):
    """
    Save pooled embeddings to S3 as a CSV file.
    """
    df = pd.DataFrame(embeddings, columns=[f"feature_{i}" for i in range(embeddings.shape[1])])
    
    s3_key = f"pooled_embeddings/{file_name}"
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"Saved {file_name} to S3: {s3_key}")

# %%
# List all embedding `.pt` files in the S3 bucket
folder_name = "embeddings/"
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
pt_files = [content["Key"] for content in response.get("Contents", []) if content["Key"].endswith(".pt")]

print(f"Found {len(pt_files)} embedding files.")

# Ensure the files are sorted correctly before selecting chunk 12
import re
pt_files = sorted(pt_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

# Select starting from chunk 12 (zero-based index 11)
embedding_files = pt_files

# Process and save the files from chunk 12 onward
for i, file_key in enumerate(embedding_files, start=0):  # Start numbering from chunk 12
    pooled_embeddings = load_and_process_embedding(bucket_name, file_key)
    save_embeddings_to_s3(pooled_embeddings, f"pooled_embeddings_chunk_{i}.csv")
    
    # Explicitly free memory
    del pooled_embeddings  

print("Pooled embeddings saved to S3 starting from chunk 12.")