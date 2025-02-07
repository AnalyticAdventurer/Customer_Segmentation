# Purpose: Run K-Means clustering and save customer segment assignments.

import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(input_path, output_path, num_clusters=4):
    """Apply K-Means clustering on the dataset and save cluster assignments."""

    # Load processed dataset
    df = pd.read_csv(input_path)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(df)

    # Save clustered dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Clustering completed. Results saved at: {output_path}")

# Run script
if __name__ == "__main__":
    run_kmeans("../data/processed_data.csv", "../results/cluster_assignments.csv")