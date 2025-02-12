import os
import boto3
import pandas as pd
import pyarrow.parquet as pq
from io import BytesIO  # <-- Fix: Use BytesIO instead of StringIO
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

# Define folders
embeddings_folder = "pooled_embeddings/"
parquet_folder = "parquet_embeddings/"  # New folder for Parquet files

# List all CSV embedding chunks
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=embeddings_folder)
csv_files = sorted(
    [content["Key"] for content in response.get("Contents", []) if content["Key"].endswith(".csv")]
)

print(f"Found {len(csv_files)} CSV embedding chunks.")

# Convert each CSV to Parquet
for i, file_key in enumerate(csv_files):
    print(f"Processing {i+1}/{len(csv_files)}: {file_key}")

    # Load CSV from S3
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(BytesIO(obj["Body"].read()))  # Fix: Read as binary

    # Convert to Parquet
    parquet_buffer = BytesIO()  # Fix: Use binary file buffer
    parquet_key = file_key.replace("pooled_embeddings/", "parquet_embeddings/").replace(".csv", ".parquet")

    df.to_parquet(parquet_buffer, index=False, engine="pyarrow", compression="snappy")

    # Upload Parquet to S3
    s3.put_object(Bucket=bucket_name, Key=parquet_key, Body=parquet_buffer.getvalue())
    print(f"Saved to S3: {parquet_key}")

print("All CSVs converted to Parquet.")