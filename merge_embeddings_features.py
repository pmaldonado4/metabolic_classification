import boto3
import pyarrow.parquet as pq
import pandas as pd
from io import BytesIO, StringIO
from botocore.config import Config

# AWS S3 Configuration
read_access_key = "TXZ5TA2AZQIO2UPPL7LS"
read_secret_key = "GcQMOd2U1NS4FIXcez6mBI4Fx8xzULi2rcfcW18I"
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

# File locations in S3
parquet_file_key = "master_embeddings/merged_embeddings.parquet"
features_file_key = "dataset/features.csv"

### **Step 1: Load the Master Parquet File**
print(f"Loading embeddings from {parquet_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=parquet_file_key)
parquet_buffer = BytesIO(obj["Body"].read())
embeddings_df = pq.read_table(parquet_buffer).to_pandas()
print(f"Embeddings loaded. Shape: {embeddings_df.shape}")

### **Step 2: Load the Features CSV File**
print(f"Loading metadata from {features_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=features_file_key)
features_df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
print(f"Metadata loaded. Shape: {features_df.shape}")

### **Step 3: Ensure Matching Row Counts Before Merging**
if embeddings_df.shape[0] != features_df.shape[0]:
    print("⚠️ Warning: Row counts do not match! Ensure they are properly aligned.")
    print(f"Embeddings Rows: {embeddings_df.shape[0]}, Metadata Rows: {features_df.shape[0]}")
else:
    print("Row counts match. Proceeding with merging.")

### **Step 4: Concatenate Features and Embeddings**
final_df = pd.concat([embeddings_df, features_df], axis=1)
print(f"Final dataset shape: {final_df.shape}")

### **Step 5: Save the Final Dataset to S3**
output_file_key = "processed_data/final_dataset.parquet"

with BytesIO() as final_parquet_buffer:
    final_df.to_parquet(final_parquet_buffer, engine="pyarrow", compression="snappy", index=False)
    final_parquet_buffer.seek(0)
    s3.put_object(Bucket=bucket_name, Key=output_file_key, Body=final_parquet_buffer.getvalue())

print(f"Final dataset saved to S3: {output_file_key}")