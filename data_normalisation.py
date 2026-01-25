import pandas as pd
import numpy as np

# 1. Load the data
# Note: skipfooter=3 removes those CLAN warning rows at the bottom
df = pd.read_csv('UD_kideval_NOSD.csv', skipfooter=3, engine='python')

# 2. Define the morphological columns (raw counts)
morph_cols = [
    '*-PRESP','*-PL', 'u-cop',
    'det:art', '*-PAST', '*-S3', 'u-aux'
]
# 3. Feature Engineering: Convert Raw Counts to Rates
# We divide by 'mor_Words' and multiply by 100 to get "Rate per 100 words"
for col in morph_cols:
    df[f'{col}_rate'] = (df[col] / df['mor_Words']) * 100


# 4. Select the final features for analysis
features = ['MLU_Morphemes','*-PRESP', '*-PL', 'u-cop',
    'det:art', '*-PAST', '*-S3', 'u-aux'] + [f'{c}_rate' for c in morph_cols]

# 5. Handle missing values (fill NAs with 0 for the rates, as NA usually means "did not occur")
df[features] = df[features].fillna(0)

# 6. TD-Baseline Normalization (Z-Scores)
# We calculate Mean/SD from the TD group only
td_group = df[df['Group'] == 'TD']

z_scored_data = df.copy()

for feature in features:
    td_mean = td_group[feature].mean()
    td_std = td_group[feature].std()

    # Avoid division by zero if a feature never appears in TD group
    if td_std == 0 or np.isnan(td_std):
        z_scored_data[f'{feature}_z'] = 0
    else:
        z_scored_data[f'{feature}_z'] = (df[feature] - td_mean) / td_std

# 7. Save the normalized dataframe for the next step (PCA)
z_cols = [f'{f}_z' for f in features]
final_df = z_scored_data[['File', 'Group'] + z_cols]
final_df.to_csv('normalized_kideval_data.csv', index=False)

print("Normalization complete. TD group mean is now 0.0.")