# Purpose: Generate and save visualizations of customer segmentation results.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_clusters(input_path, output_path):
    """Generate visualizations for cluster distributions and feature analysis."""

    # Load clustered dataset
    df = pd.read_csv(input_path)

    # Countplot of clusters
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Cluster", data=df, palette="viridis")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.title("Customer Distribution Across Clusters")
    plt.savefig(f"{output_path}/cluster_distribution.png")  # Save image
    plt.close()

    # Feature distribution per cluster
    num_features = df.select_dtypes(include=[np.number]).columns.drop("Cluster")

    plt.figure(figsize=(12, 6))
    for i, feature in enumerate(num_features[:4]):  # Limit to 4 features
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x="Cluster", y=feature, data=df, palette="coolwarm")
        plt.title(f"{feature} Distribution by Cluster")

    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_distribution.png")  # Save image
    plt.close()

    print(f"✅ Visualizations saved in: {output_path}")

# Run script
if __name__ == "__main__":
    visualize_clusters("../results/cluster_assignments.csv", "../results/")