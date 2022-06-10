import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter

def hard_vote_ensemble(list_df: list) -> pd.DataFrame:

    num_files = len(list_df)
    labels = [df_item["similar"].values for df_item in list_df]
    sum_labels = [sum(label) for label in zip(*labels)]
    ensembled_labels = []

    for label in tqdm(sum_labels):
        if label > num_files / 2:
            label = 1
        else:
            label = 0
        ensembled_labels.append(label)

    df_ensembled = list_df[0].copy()
    df_ensembled["similar"] = ensembled_labels
    return df_ensembled

prefix = "./submissions/SOTA/"


""" graphcode roberta kfold training with 900k datasets per fold """
fold_1 = pd.read_csv(
    os.path.join(prefix, "graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold1.csv")
)
fold_2 = pd.read_csv(
    os.path.join(prefix, "graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold2.csv")
)
fold_3 = pd.read_csv(
    os.path.join(prefix, "graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold3.csv")
)
fold_4 = pd.read_csv(
    os.path.join(prefix, "graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold4.csv")
)
fold_5 = pd.read_csv(
    os.path.join(prefix, "graphcodebert_rbert_EP:2_BS:32_WR:0.05_WD:1e-2_LR:2e-5_fold5.csv")
)

""" t5 kfold training with 900k datasets per fold """
t5_fold1 = pd.read_csv(
    os.path.join(prefix, "VHT5_EP_3_BS_48_WR_0.05_WD_1e-2_LR_3e-5_fold1_30000step.csv")
)
t5_fold2 = pd.read_csv(
    os.path.join(prefix, "VHT5_EP_3_BS_48_WR_0.05_WD_1e-2_LR_3e-5_fold2_31500step.csv")
)
t5_fold3 = pd.read_csv(
    os.path.join(prefix, "VHT5_EP_3_BS_48_WR_0.05_WD_1e-2_LR_3e-5_fold3_18000step.csv")
)
t5_fold4 = pd.read_csv(
    os.path.join(prefix, "VHT5_EP_3_BS_48_WR_0.05_WD_1e-2_LR_3e-5_fold4_28500step.csv")
)

""" plbert kfold training with 900k datasets per fold """

plbart_fold_1 = pd.read_csv(
    os.path.join(
        prefix, "VHPLBART_EP:3_BS:32_WR:0.00_WD:0.00_LR:3e-5_fold1_42000step.csv"
    )
)
plbart_fold_2 = pd.read_csv(
    os.path.join(
        prefix, "VHPLBART_EP:3_BS:32_WR:0.00_WD:0.00_LR:3e-5_fold2_20000step.csv"
    )
)

list_submissions = [
    fold_1,
    fold_2,
    fold_3,
    fold_4,
    fold_5,  # 5 graphcodeberts
    t5_fold1,
    t5_fold2,
    t5_fold3,
    t5_fold4,
    plbart_fold_1,
    plbart_fold_2,
]
df_ensembled = hard_vote_ensemble(list_submissions)
df_ensembled.head()