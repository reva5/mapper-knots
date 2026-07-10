import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import kmapper as km
import os

OUTPUT_DIR = 'Documents/knot_output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load & Clean Data ----------
df = pd.read_csv('C:/Users/ellio/Documents/Ultra File/Math/1 PCMI/XML Group 8 TDA/code/knot_data/knotdataFinal_cleaned.csv')

# Clean column names
df.columns = df.columns.str.replace(r'<.*?>', '', regex=True).str.strip()

# Final_cleaned already has numeric L-space and split Nu columns,
# so skip Yes/No mapping and Nu string-splitting
feature_df = df.drop(columns=['Name'], errors='ignore').apply(pd.to_numeric, errors='coerce')

print("Rows loaded:", len(df))
print("NaNs per column:\n", feature_df.isna().sum())

# Impute NaNs with median
feature_df = feature_df.fillna(feature_df.median())

# ---------- Scale & PCA ----------
X = StandardScaler().fit_transform(feature_df.values)
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

# ---------- PCA Loadings ----------
loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=feature_df.columns)
print("\nPCA loadings (feature influence on each axis):")
print(loadings)

# ---------- Option A: color by each invariant ----------
for col in feature_df.columns:
    plt.figure()
    sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=feature_df[col],
                     cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label=col)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"Knot Invariants PCA — colored by {col}")
    safe = col.replace(' ', '_').replace('-', '_')
    plt.savefig(os.path.join(OUTPUT_DIR, f'pca_colored_by_{safe}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

# ---------- Option B: KMeans clustering ----------
k = 4
labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(X)
plt.figure()
sc = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(sc, label='Cluster')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"Knot Invariants PCA — KMeans ({k} clusters)")
plt.savefig(os.path.join(OUTPUT_DIR, 'pca_kmeans_clusters.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# ---------- Cluster Profile ----------
cluster_profile = feature_df.copy()
cluster_profile['cluster'] = labels
print(cluster_profile.groupby('cluster').mean())

# =====================================================
# ---------- MAPPER ALGORITHM (kmapper) ----------
# =====================================================

# Step 1: Initialize the Mapper object
mapper = km.KeplerMapper(verbose=1)

# Step 2: Create the lens using PCA Components 1 & 2 (2D lens)
# 'reduced' already contains PCA Component 1 & 2 from above
lens = reduced  # shape (n_samples, 2)

# Step 3: Check that Genus-4D exists as a column for coloring
color_col = 'Genus-4D'
if color_col not in feature_df.columns:
    # Try common alternative names
    possible = [c for c in feature_df.columns if '4d' in c.lower() or 'genus' in c.lower()]
    if possible:
        color_col = possible[0]
        print(f"Note: 'Genus-4D' not found, using '{color_col}' instead.")
    else:
        color_col = feature_df.columns[0]
        print(f"Warning: No Genus/4D column found. Defaulting to '{color_col}'.")

color_values = feature_df[color_col].values

# Step 4: Build the Mapper graph
#   - cover: 15 intervals with 50% overlap (tune as needed)
#   - clusterer: KMeans with k=2 within each cover element
graph = mapper.map(
    lens,
    X,                                          # use full scaled feature matrix
    cover=km.Cover(n_cubes=15, perc_overlap=0.5),
    clusterer=KMeans(n_clusters=2, random_state=0, n_init=10)
)

# Step 5: Visualize and save as interactive HTML
html_path = os.path.join(OUTPUT_DIR, 'mapper_graph_genus4D.html')

mapper.visualize(
    graph,
    path_html=html_path,
    title="Knot Invariants — Mapper Graph (Lens: PCA 1 & 2, Color: Genus-4D)",
    color_values=color_values,
    color_function_name=color_col,
)

print(f"\nMapper HTML saved to: {html_path}")
print(f"  Nodes in graph : {len(graph['nodes'])}")
print(f"  Edges in graph : {len(graph['links'])}")