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
parquet_file_key = "master_embeddings/merged_embeddings.parquet"
features_file_key = "dataset/features.csv"
output_file_key = "processed_data/final_merged_dataset.parquet"

### **Step 1: Load the Master Parquet File**
print(f"Loading embeddings from {parquet_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=parquet_file_key)
parquet_buffer = BytesIO(obj["Body"].read())
embeddings_df = pq.read_table(parquet_buffer).to_pandas()
print(f"Embeddings loaded. Shape: {embeddings_df.shape}")

### **Step 2: Load the Features CSV File**
print(f"Loading metadata from {features_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=features_file_key)
csv_content = obj["Body"].read().decode("utf-8")

# Try different delimiters if needed
features_df = pd.read_csv(StringIO(csv_content), sep="\t")
print(f"Metadata loaded. Shape: {features_df.shape}")

### **Step 3: Ensure Matching Row Counts Before Merging**
if embeddings_df.shape[0] != features_df.shape[0]:
    print(f"⚠️ Warning: Row mismatch! Embeddings: {embeddings_df.shape[0]}, Metadata: {features_df.shape[0]}")
else:
    print("Row counts match. Proceeding with merging.")

### **Step 4: Merge Features with Embeddings**
merged_df = pd.concat([embeddings_df, features_df], axis=1)
print(f"Final merged dataset shape: {merged_df.shape}")

### **Step 5: Save the Final Dataset in Memory**
final_parquet_buffer = BytesIO()
merged_table = pa.Table.from_pandas(merged_df)
pq.write_table(merged_table, final_parquet_buffer, compression="snappy")
final_parquet_buffer.seek(0)

### **Step 6: Upload to S3**
print(f"Uploading {output_file_key} to S3...")
s3.upload_fileobj(final_parquet_buffer, bucket_name, output_file_key)

print(f"Successfully uploaded {output_file_key} to S3.")