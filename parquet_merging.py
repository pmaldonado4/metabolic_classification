# %%
import boto3
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from io import BytesIO
from botocore.config import Config


# %%
# Access keys
read_access_key = "TXZ5TA2AZQIO2UPPL7LS"
read_secret_key = "GcQMOd2U1NS4FIXcez6mBI4Fx8xzULi2rcfcW18I"
bucket_name = "metabolic-atac-peaks"
endpoint_url = "https://rice1.osn.mghpcc.org"

# Create S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=read_access_key,
    aws_secret_access_key=read_secret_key,
    endpoint_url=endpoint_url,  # Custom endpoint
    config=Config(signature_version="s3v4"),
)

folder_path = 'parquet_embeddings/'
# List objects in the S3 bucket under the specified folder
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
parquet_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".parquet")]

print(f"Found {len(parquet_files)} Parquet files.")



# %%
# Initialize an empty list to store dataframes
all_dataframes = []

# Load and merge each Parquet file
for i, file_key in enumerate(parquet_files):
    print(f"Loading {i+1}/{len(parquet_files)}: {file_key}")

    # Download file from S3 into memory
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    parquet_buffer = BytesIO(obj["Body"].read())

    # Read the Parquet file
    table = pq.read_table(parquet_buffer)
    df = table.to_pandas()

    print(f"Loaded {file_key} - Shape: {df.shape}")
    all_dataframes.append(df)  # Store in list

# Merge all DataFrames
print("Merging all Parquet files into one DataFrame...")
merged_df = pd.concat(all_dataframes, ignore_index=True)
print(f"Final merged dataset shape: {merged_df.shape}")

# Save merged dataset locally (optional)
merged_df.to_parquet("merged_embeddings.parquet", index=False, engine="pyarrow", compression="snappy")
print("Merged Parquet file saved locally as 'merged_embeddings.parquet'.")

# Upload merged dataset back to S3
merged_s3_key = "processed_data/master_embeddings.parquet"
with BytesIO() as final_parquet_buffer:
    table = pa.Table.from_pandas(merged_df)
    pq.write_table(table, final_parquet_buffer, compression="snappy")
    final_parquet_buffer.seek(0)
    s3.put_object(Bucket=bucket_name, Key=merged_s3_key, Body=final_parquet_buffer.getvalue())

print(f"Saved merged dataset to S3: {merged_s3_key}")


