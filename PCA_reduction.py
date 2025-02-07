import os
import boto3
import pandas as pd
import torch
import numpy as np
from io import StringIO
from botocore.config import Config
from sklearn.decomposition import PCA
import joblib  # For saving PCA models

# AWS S3 Configuration
read_access_key = "L7J5V9NECMPRCRFLCAD7"
read_secret_key = "AhcamdaEP7pHAJkCiklALCOh4lKd6ZcxT8HtqLuV"
bucket_name = "metabolic-atac-peaks"
endpoint_url = "https://rice1.osn.mghpcc.org"

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=read_access_key,
    aws_secret_access_key=read_secret_key,
    endpoint_url=endpoint_url,
    config=Config(signature_version="s3v4"),
)

# S3 paths
embeddings_folder = "pooled_embeddings/"
metadata_file_key = "dataset/features.csv"
output_folder = "processed_embeddings/"

# Load Metadata (features file, small enough to fit in memory)
def load_metadata_from_s3(bucket_name, file_key):
    print(f"Loading metadata from {file_key}...")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    metadata_df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    print(f"Metadata loaded: {metadata_df.shape}")
    return metadata_df

metadata_df = load_metadata_from_s3(bucket_name, metadata_file_key)

# Get list of embedding files
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=embeddings_folder)
embedding_files = sorted(
    [content["Key"] for content in response.get("Contents", []) if content["Key"].endswith(".csv")]
)

print(f"Found {len(embedding_files)} embedding chunks.")

# Define PCA model
pca = PCA(n_components=50)  # Reduce to 50 dimensions for visualization

# Sequentially process each chunk
for i, file_key in enumerate(embedding_files):
    print(f"Processing chunk {i+1}/{len(embedding_files)}: {file_key}")

    # Load chunk
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    embeddings_df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    # Ensure number of rows matches metadata
    if embeddings_df.shape[0] != metadata_df.shape[0]:
        print(f"Warning: Row mismatch! ({embeddings_df.shape[0]} != {metadata_df.shape[0]})")
        embeddings_df = embeddings_df.iloc[: metadata_df.shape[0]]  # Trim if necessary

    # Merge embeddings with metadata
    merged_df = pd.concat([embeddings_df, metadata_df], axis=1)

    # Apply PCA
    reduced_embeddings = pca.fit_transform(merged_df.iloc[:, :-len(metadata_df.columns)].values)

    # Convert to DataFrame
    reduced_df = pd.DataFrame(reduced_embeddings, columns=[f"PCA_{i}" for i in range(reduced_embeddings.shape[1])])
    reduced_df = pd.concat([reduced_df, metadata_df], axis=1)

    # Save reduced embeddings to S3
    output_key = f"{output_folder}/reduced_embeddings_chunk_{i+1}.csv"
    csv_buffer = StringIO()
    reduced_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())

    print(f"Saved reduced embeddings: {output_key}")

# Save PCA model for later use
pca_model_path = "pca_model.joblib"
joblib.dump(pca, pca_model_path)
s3.upload_file(pca_model_path, bucket_name, f"{output_folder}/pca_model.joblib")
print(f"PCA model saved to S3.")

print("All chunks processed and reduced.")