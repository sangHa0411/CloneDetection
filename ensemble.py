import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

input_name = input("Enter the name of the ensemble file: ")
submission_paths = [
    './output/submission_1.csv',
    './output/submission_2.csv',
    './output/submission_3.csv',
]
df_submissions = [pd.read_csv(path) for path in submission_paths]
val_submissions = [df['similar'].values for df in df_submissions]

# hardvoting ensemble based on submissions' labels
df_ensembled = pd.DataFrame({})

for row_index, row in tqdm(df_submissions[0].iterrows()):
    labels = []
    for value in val_submissions:
        labels.append(value[row_index])
    # hardvoting ensemble using Counter
    ensemble_label = Counter(labels).most_common(1)[0][0]
    # print(ensemble_label)
    df_ensembled.at[row_index, "similar"] = ensemble_label
print(df_ensembled.shape)

# apply int for "similar"
df_ensembled["pair_id"] = df_submissions[0]["pair_id"]
# apply int for similar
df_ensembled["similar"] = df_ensembled["similar"].astype(int)
# reorder column 
df_ensembled = df_ensembled[["pair_id", "similar"]]
df_ensembled.to_csv("./ensembled/{input_name}_ensembled", index=False)