import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import kruskal

# 1. Load the ALREADY normalized data
df_z = pd.read_csv('normalized_kideval_data.csv')

# 2. FILTERING: Focus only on the 3 finalists and the real groups
finalist_groups = ['TD', 'SLI', 'DLD_SC2', 'DLD_SC4', 'DLD_SC6', 'DLD_SC10']
df_final = df_z[df_z['Group'].isin(finalist_groups)].copy()

# 3. DEFINE FEATURES: MLU + Morphological Rates (No Lexical, No Raw Counts)
# We select MLU_Morphemes_z and every column that ends with 'rate_z'
features = [col for col in df_z.columns if col.endswith('rate_z')]
if 'MLU_Morphemes_z' in df_z.columns:
    features.append('MLU_Morphemes_z')

print(f"Linguistic markers used for Finalist PCA: {features}")


# 6. STATISTICAL "TURING TEST"
print("\n" + "=" * 60)
print("STATISTICAL SIMILARITY TO REAL SLI")
print("Goal: p > 0.05 (No significant difference from real kids)")
print("=" * 60)

# We test key DLD markers
markers_to_test = ['MLU_Morphemes_z', '*-PAST_rate_z', '*-S3_rate_z', 'det:art_rate_z', 'u-cop_rate_z']

for marker in markers_to_test:
    if marker not in df_final.columns: continue
    print(f"\n>>> TESTING: {marker}")

    real_sli = df_final[df_final['Group'] == 'SLI'][marker]

    for condition in ['DLD_SC2', 'DLD_SC4', 'DLD_SC6', 'DLD_SC10']:
        synth = df_final[df_final['Group'] == condition][marker]
        _, p = kruskal(real_sli, synth)

        status = "MATCH (Excellent)" if p > 0.05 else "FAIL (Statistically Different)"
        print(f"  SLI vs {condition}: p={p:.3f} -> {status}")
