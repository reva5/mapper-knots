import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

OUTPUT_DIR = 'Documents/knot_output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv('code/knot_data/knotinfoBIG.csv')

# --- clean the header column names (remove HTML tags like <i>s</i>) ---
df.columns = df.columns.str.replace(r'<.*?>', '', regex=True).str.strip()

if 'L-space' in df.columns:
    df['L-space'] = df['L-space'].map({'Yes': 1, 'No': 0})

if 'Nu' in df.columns:
    nu = df['Nu'].astype(str).str.strip('[]').str.split(';', expand=True)
    df['Nu_1'] = pd.to_numeric(nu[0], errors='coerce')
    df['Nu_2'] = pd.to_numeric(nu[1], errors='coerce')
    df = df.drop(columns=['Nu'])

feature_df = df.drop(columns=['Name'], errors='ignore').apply(pd.to_numeric, errors='coerce')

print("Rows loaded:", len(df))
print("NaNs per column:\n", feature_df.isna().sum())

# keep points by imputing instead of dropping
feature_df = feature_df.fillna(feature_df.median())

X = StandardScaler().fit_transform(feature_df.values)
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

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
k = 4  # adjust as desired
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

# ---------- Interpret clusters: mean invariant per cluster ----------
cluster_profile = feature_df.copy()
cluster_profile['cluster'] = labels
print(cluster_profile.groupby('cluster').mean())

# ---------- Which invariants drive the axes: PCA loadings ----------
loadings = pd.DataFrame(pca.components_.T,
                        columns=['PC1', 'PC2'],
                        index=feature_df.columns)
print("\nPCA loadings (feature influence on each axis):")
print(loadings)