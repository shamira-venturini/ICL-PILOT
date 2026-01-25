import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# 1. Load the normalized data from the previous step
df_z = pd.read_csv('normalized_kideval_data.csv')

# Define the features (the Z-score columns)
features_z = [col for col in df_z.columns if col.endswith('rate_z')]

# Check the exact name in your CSV. It is likely 'MLU_Morphemes_z'
if 'MLU_Morphemes_z' in df_z.columns:
    features_z.append('MLU_Morphemes_z')
elif 'MLU_Morphemes_rate_z' in df_z.columns: # fallback in case of naming diff
    features_z.append('MLU_Morphemes_rate_z')

# 2. Run PCA
pca = PCA(n_components=2)
components = pca.fit_transform(df_z[features_z])

# Create a PCA dataframe for plotting
df_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
df_pca['Group'] = df_z['Group']

# 3. Visualization: PCA Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Group', style='Group',
                palette='viridis', s=100, alpha=0.7)

plt.title('PCA of Morphosyntactic Profiles\n(All Conditions vs. Real SLI/TD)', fontsize=14)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('kideval_pca_plot.png')
plt.show()

# 4. Distance Analysis: Finding the "Best" Condition
# We calculate the "Centroid" (average profile) for the Real SLI group
sli_centroid = df_z[df_z['Group'] == 'SLI'][features_z].mean()

# Calculate the distance from each group's average profile to the SLI average profile
results = []
groups = df_z['Group'].unique()

for group in groups:
    if group == 'TD' or group == 'SLI':
        continue  # We only want to rank the Synthetic conditions

    group_centroid = df_z[df_z['Group'] == group][features_z].mean()
    dist = euclidean(sli_centroid, group_centroid)

    results.append({'Condition': group, 'Distance_to_SLI': dist})

# Convert to DataFrame and rank
distance_df = pd.DataFrame(results).sort_values(by='Distance_to_SLI')

print("\n--- FIDELITY RANKING ---")
print("Condition with the lowest distance is most morphosyntactically similar to Real SLI.")
print(distance_df)

# 5. Interpret which features drive the difference (Loadings)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=features_z
)
print("\n--- TOP PCA LOADINGS (What defines the DLD profile?) ---")
print(loadings.abs().sort_values(by='PC1', ascending=False).head(5))