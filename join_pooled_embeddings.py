import os
import boto3
import pandas as pd
import dask.dataframe as dd
from io import StringIO
from botocore.config import Config

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

# Folder where pooled embeddings are stored
embeddings_folder = "pooled_embeddings/"
metadata_file_key = "dataset/features.csv"
output_folder = "processed_data/"


def load_metadata_from_s3(bucket_name, file_key):
    """
    Load metadata file from S3 and convert it to a Dask DataFrame.
    """
    print(f"Loading metadata from {file_key}...")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    
    # First, load it as a pandas DataFrame
    metadata_pandas_df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
    
    # Convert to Dask DataFrame for large-scale processing
    metadata_df = dd.from_pandas(metadata_pandas_df, npartitions=10)  # Adjust partitions if needed
    
    print(f"Metadata shape: {metadata_pandas_df.shape}")  # Keep this with pandas for clarity

    return metadata_df

def process_and_merge_embeddings(bucket_name, embeddings_folder, metadata_df, output_folder):
    """
    Process embeddings in chunks, merging with metadata, and saving each merged chunk to S3.
    """
    print("Fetching list of pooled embedding files from S3...")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=embeddings_folder)
    embedding_files = sorted([content["Key"] for content in response.get("Contents", []) if content["Key"].endswith(".csv")])

    print(f"Found {len(embedding_files)} embedding files.")

    for idx, file_key in enumerate(embedding_files):
        print(f"Processing {file_key} ({idx+1}/{len(embedding_files)})...")
        
        # Load embeddings as Dask DataFrame (efficient for large files)
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        embeddings_df = dd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

        # Ensure the metadata and embeddings match in size
        assert len(embeddings_df) == len(metadata_df), f"Mismatch in rows: {len(embeddings_df)} vs {len(metadata_df)}"

        # Merge embeddings with metadata
        merged_df = dd.concat([embeddings_df, metadata_df], axis=1)

        # Convert to CSV in memory and upload to S3
        output_file_key = f"{output_folder}/merged_chunk_{idx+1}.csv"
        csv_buffer = StringIO()
        merged_df.compute().to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=csv_buffer.getvalue())

        print(f"Saved merged chunk to S3: {output_file_key}")

# Load metadata
metadata_df = load_metadata_from_s3(bucket_name, metadata_file_key)

# Process embeddings in chunks and merge with metadata
process_and_merge_embeddings(bucket_name, embeddings_folder, metadata_df, output_folder)

print("All merged chunks saved to S3 successfully.")