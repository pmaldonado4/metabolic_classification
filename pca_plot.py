import boto3
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pyarrow as pa
from io import BytesIO
from botocore.config import Config
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
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
parquet_file_key = "processed_data/final_merged_with_sensitivity.parquet"
output_pca_file_key = "processed_data/pca_embeddings.parquet"
output_variance_file_key = "processed_data/pca_explained_variance.parquet"
output_tsne_file_key = "processed_data/tsne_embeddings.parquet"
output_umap_file_key = "processed_data/umap_embeddings.parquet"

### **Step 1: Load the Parquet File in Chunks**
print(f"Loading embeddings from {parquet_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=parquet_file_key)
parquet_buffer = BytesIO(obj["Body"].read())
df = pq.read_table(parquet_buffer).to_pandas()
print(f"Embeddings loaded. Shape: {df.shape}")

# **Extract only the embedding columns**
embedding_columns = [col for col in df.columns if col.startswith("feature_")]
embeddings = df[embedding_columns]

# **Normalize the Data**
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# **Chunking Parameters**
batch_size = 10000  # Adjust for memory constraints
n_components = 50  # Keep more components for t-SNE/UMAP later
ipca = IncrementalPCA(n_components=n_components)

# **Step 2: Fit Incremental PCA in Chunks**
print("Fitting Incremental PCA...")
for i in range(0, embeddings_scaled.shape[0], batch_size):
    batch = embeddings_scaled[i:i+batch_size]
    ipca.partial_fit(batch)

# **Step 3: Transform the Data in Chunks**
print("Transforming data using Incremental PCA...")
pca_embeddings = np.vstack([
    ipca.transform(embeddings_scaled[i:i+batch_size]) 
    for i in range(0, embeddings_scaled.shape[0], batch_size)
])

print(f"PCA transformation complete. Shape: {pca_embeddings.shape}")

# **Save PCA-transformed data to S3**
pca_df = pd.DataFrame(pca_embeddings, columns=[f"PCA_{i+1}" for i in range(n_components)])
pca_df["Metabolic_Sensitive"] = df["metabolically_sensitive"]  # Keep labels

# Convert DataFrame to Parquet
final_parquet_buffer = BytesIO()
pq.write_table(pa.Table.from_pandas(pca_df), final_parquet_buffer, compression="snappy")
final_parquet_buffer.seek(0)

print(f"Uploading PCA-transformed dataset to S3: {output_pca_file_key}...")
s3.upload_fileobj(final_parquet_buffer, bucket_name, output_pca_file_key)

# **Step 4: Save Explained Variance**
explained_variance = pd.DataFrame({
    "Component": [f"PCA_{i+1}" for i in range(n_components)],
    "Explained_Variance": ipca.explained_variance_ratio_
})

variance_parquet_buffer = BytesIO()
pq.write_table(pa.Table.from_pandas(explained_variance), variance_parquet_buffer, compression="snappy")
variance_parquet_buffer.seek(0)

print(f"Uploading PCA explained variance to S3: {output_variance_file_key}...")
s3.upload_fileobj(variance_parquet_buffer, bucket_name, output_variance_file_key)

# **Step 5: Compute t-SNE on the first 50 PCA components**
print("Computing t-SNE...")
tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
tsne_embeddings = tsne_model.fit_transform(pca_embeddings[:, :50])  # Reduce from 50 to 2D

# Save t-SNE embeddings
tsne_df = pd.DataFrame(tsne_embeddings, columns=["TSNE_1", "TSNE_2"])
tsne_df["Metabolic_Sensitive"] = df["metabolically_sensitive"]

tsne_parquet_buffer = BytesIO()
pq.write_table(pa.Table.from_pandas(tsne_df), tsne_parquet_buffer, compression="snappy")
tsne_parquet_buffer.seek(0)

print(f"Uploading t-SNE embeddings to S3: {output_tsne_file_key}...")
s3.upload_fileobj(tsne_parquet_buffer, bucket_name, output_tsne_file_key)

# **Step 6: Compute UMAP**
print("Computing UMAP...")
umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_embeddings = umap_model.fit_transform(pca_embeddings[:, :50])  # Reduce from 50 to 2D

# Save UMAP embeddings
umap_df = pd.DataFrame(umap_embeddings, columns=["UMAP_1", "UMAP_2"])
umap_df["Metabolic_Sensitive"] = df["metabolically_sensitive"]

umap_parquet_buffer = BytesIO()
pq.write_table(pa.Table.from_pandas(umap_df), umap_parquet_buffer, compression="snappy")
umap_parquet_buffer.seek(0)

print(f"Uploading UMAP embeddings to S3: {output_umap_file_key}...")
s3.upload_fileobj(umap_parquet_buffer, bucket_name, output_umap_file_key)

print("PCA, t-SNE, and UMAP computations completed successfully. Data saved to S3.")