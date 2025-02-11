import boto3
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import pyarrow as pa
from io import BytesIO
from botocore.config import Config
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt

# **AWS S3 Configuration**
read_access_key = "TXZ5TA2AZQIO2UPPL7LS"
read_secret_key = "GcQMOd2U1NS4FIXcez6mBI4Fx8xzULi2rcfcW18I"
bucket_name = "metabolic-atac-peaks"
endpoint_url = "https://rice1.osn.mghpcc.org"

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
figure_file_key = "figures/pca_plot.pdf"

### **Step 1: Load the Parquet File in Chunks**
print(f"Loading embeddings from {parquet_file_key}...")
obj = s3.get_object(Bucket=bucket_name, Key=parquet_file_key)
parquet_buffer = BytesIO(obj["Body"].read())
df = pq.read_table(parquet_buffer).to_pandas()
print(f"Embeddings loaded. Shape: {df.shape}")

# **Extract only the embedding columns**
embedding_columns = [col for col in df.columns if col.startswith("feature_")]
embeddings = df[embedding_columns]

# **Normalize the Data (Optional)**
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# **Chunking Parameters**
batch_size = 10000  # Adjust based on memory constraints
n_components = 2  # PCA to 2D for visualization
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

# **Step 4: Save PCA-transformed data to S3**
pca_df = pd.DataFrame(pca_embeddings, columns=["PCA1", "PCA2"])
pca_df["Metabolic_Sensitive"] = df["metabolically_sensitive"]  # Keep labels

# Convert DataFrame to Parquet in Memory
final_parquet_buffer = BytesIO()
pq.write_table(pa.Table.from_pandas(pca_df), final_parquet_buffer, compression="snappy")
final_parquet_buffer.seek(0)

print(f"Uploading PCA-transformed dataset to S3: {output_pca_file_key}...")
s3.upload_fileobj(final_parquet_buffer, bucket_name, output_pca_file_key)

# **Step 5: Visualize the PCA Results**
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_df["PCA1"], pca_df["PCA2"], c=pca_df["Metabolic_Sensitive"], cmap="coolwarm", alpha=0.6)
plt.colorbar(scatter, label="Metabolically Sensitive (1) vs. Insensitive (0)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Visualization of DNA Embeddings")

# Save Plot as PDF
plt.savefig("pca_plot.pdf")
plt.close()

# **Upload Plot to S3**
print(f"Uploading PCA plot to S3: {figure_file_key}...")
s3.upload_file("pca_plot.pdf", bucket_name, figure_file_key)

print("✅ PCA computation and visualization completed successfully.")