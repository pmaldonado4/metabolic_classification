# %%
import os
import torch
import dask.array as da
from dask import delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import boto3
import pandas as pd
from io import StringIO
from botocore.config import Config

# %%
# Slurm Cluster Setup
cluster = SLURMCluster(
    queue='gpuA40x4',
    account='bdhi-delta-gpu',
    cores=35,
    memory='100GB',
    job_extra_directives=["--gpus=1"],
    walltime='01:00:00',
)
cluster.scale(jobs=2)
client = Client(cluster)

print("Cluster Dashboard:", cluster.dashboard_link)

# %%
# AWS S3 Configuration
read_access_key = "L7J5V9NECMPRCRFLCAD7"
read_secret_key = "AhcamdaEP7pHAJkCiklALCOh4lKd6ZcxT8HtqLuV"
bucket_name = "metabolic-atac-peaks"
endpoint_url = "https://rice1.osn.mghpcc.org"

# Delayed function to load and process PT files
@delayed
def load_pt_file_from_s3(bucket_name, file_key):
    """
    Load a .pt file from S3, flatten the embeddings, and return as a NumPy array.
    """
    print(f"Starting download: {file_key}")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=read_access_key,
        aws_secret_access_key=read_secret_key,
        endpoint_url=endpoint_url,
        config=Config(signature_version="s3v4"),
    )

    # Download the .pt file to a temporary location
    local_file = f"/tmp/{os.path.basename(file_key)}"
    s3.download_file(bucket_name, file_key, local_file)
    print(f"Downloaded {file_key} to {local_file}")
    
    # Load and flatten embeddings
    embeddings = torch.load(local_file).numpy()  # Shape: (N, 256, 4400)
    flattened_embeddings = embeddings.reshape(embeddings.shape[0], -1)  # Shape: (N, 256*4400)
    print(f"Processed {file_key}: shape {flattened_embeddings.shape}")
    
    # Delete the temporary file
    os.remove(local_file)
    return flattened_embeddings

# %%
# List all embedding `.pt` files in the S3 bucket
folder_name = "embeddings/"
s3 = boto3.client(
    "s3",
    aws_access_key_id=read_access_key,
    aws_secret_access_key=read_secret_key,
    endpoint_url=endpoint_url,
    config=Config(signature_version="s3v4"),
)
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
pt_files = [content["Key"] for content in response.get("Contents", []) if content["Key"].endswith(".pt")]

print(f"Found {len(pt_files)} embedding files.")

# Process only the first two files for testing
embedding_files = pt_files[:2]

# %%
# Create delayed tasks for each file
embedding_tasks = [load_pt_file_from_s3(bucket_name, file_key) for file_key in embedding_files]

# Stack all embeddings into a single Dask array
example_embedding = embedding_tasks[0].compute()  # Test loading first file
embedding_shape = example_embedding.shape
embedding_dtype = example_embedding.dtype
print(f"Example embedding shape: {embedding_shape}")

dask_embeddings = da.concatenate([
    da.from_delayed(task, shape=(None, embedding_shape[1]), dtype=embedding_dtype)
    for task in embedding_tasks
])
print(f"Dask array shape: {dask_embeddings.shape}")

# %%
# Save flattened embeddings to S3 in chunks
def save_embeddings_chunk_to_s3(chunk, chunk_idx, bucket_name, folder_name):
    """
    Save a chunk of flattened embeddings to S3 as a CSV file.
    """
    print(f"Saving chunk {chunk_idx} to S3...")
    chunk_df = pd.DataFrame(chunk, columns=[f"feature_{i}" for i in range(chunk.shape[1])])
    s3_key = f"{folder_name}/flattened_embeddings_chunk_{chunk_idx}.csv"

    # Convert DataFrame to CSV in memory and upload to S3
    csv_buffer = StringIO()
    chunk_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"Saved chunk {chunk_idx} to S3: {s3_key}")

# %%
# Process and save each chunk
for i, delayed_chunk in enumerate(dask_embeddings.to_delayed()):
    print(f"Processing chunk {i + 1}...")
    chunk = delayed_chunk.compute()
    save_embeddings_chunk_to_s3(chunk, i + 1, bucket_name, "flattened_embeddings")

print("Flattened embeddings saved to S3.")