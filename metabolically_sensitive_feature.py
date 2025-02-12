import boto3
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
from io import BytesIO, StringIO
from botocore.config import Config

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
read_access_key = os.getenv("AWS_ACCESS_KEY_ID")
read_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("AWS_BUCKET_NAME")
endpoint_url = os.getenv("AWS_ENDPOINT_URL")

# **Initialize S3 client**
s3 = boto3.client(
    "s3",
    aws_access_key_id=read_access_key,
    aws_secret_access_key=read_secret_key,
    endpoint_url=endpoint_url,
    config=Config(signature_version="s3v4"),
)

# **File locations in S3**
merged_file_key = "processed_data/final_merged_dataset.parquet"
output_file_key = "processed_data/final_merged_with_sensitivity.parquet"

### **Step 1: Load the Merged Dataset from S3**
print(f"Loading merged dataset from {merged_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=merged_file_key)
parquet_buffer = BytesIO(obj["Body"].read())
df = pq.read_table(parquet_buffer).to_pandas()
print(f"Dataset loaded. Shape: {df.shape}")

### **Step 2: Create the `metabolically_sensitive` Feature**
df["metabolically_sensitive"] = (~((df["Fed"] == 1) & (df["Fasted"] == 1) & (df["Refed"] == 1))).astype(int)

# Check distribution of new feature
print(df["metabolically_sensitive"].value_counts())

### **Step 3: Save the Updated Dataset Back to S3**
final_parquet_buffer = BytesIO()
merged_table = pa.Table.from_pandas(df)
pq.write_table(merged_table, final_parquet_buffer, compression="snappy")
final_parquet_buffer.seek(0)

print(f"Uploading {output_file_key} to S3...")
s3.upload_fileobj(final_parquet_buffer, bucket_name, output_file_key)

print(f"Successfully uploaded {output_file_key} to S3.")