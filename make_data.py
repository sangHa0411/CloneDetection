import os
import shutil
import glob
from itertools import combinations
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets import Dataset, DatasetDict


# read .py documents
df_train = pd.DataFrame({}, columns=["file_name", "question_label", "code"])
TRAIN_PATH = "/home/ubuntu/plclassification/code"
list_questions = glob.glob(os.path.join(TRAIN_PATH, "*"))

list_index = []
list_problem_names = []
list_code_text = []

for index, question in enumerate(list_questions):
    problem_name = os.path.basename(question)
    # extract numerics only from problem_name
    problem_name = "".join(i for i in problem_name if i.isdigit())

    for file in glob.glob(os.path.join(question, "*.py")):
        file_base_name = os.path.basename(file)
        # read text from .py file
        with open(file, "r") as f:
            code_text = f.read()
            # append row with index, problem_name and code text
            list_index.append(file_base_name)
            list_problem_names.append(problem_name)
            list_code_text.append(code_text)

df_train["file_name"] = list_index
df_train["question_label"] = list_problem_names
# remove 0 in front of the number in question_label
df_train["question_label"] = df_train["question_label"].str.replace("^0+", "").astype(int)
df_train["code"] = list_code_text

# create similar pairs
df_similar = pd.DataFrame(
    {}, columns=["code1", "code2", "similar", "code1_group", "code2_group", "pair_id"]
)

for i in tqdm(range(1, 301)):
    list_code1 = []
    list_code2 = []
    list_question_pair_id = []
    list_group_id = []
    # get corresponding dataframe where question_label == i
    group_index = df_train[df_train["question_label"] == i].index
    df_group = df_train.iloc[group_index]

    # make unique pair combination from group_index list
    combinated_indices = list(combinations(group_index, 2))
    for index in combinated_indices:
        code1 = df_train.iloc[index[0]]["code"]
        code2 = df_train.iloc[index[1]]["code"]
        list_code1.append(code1)
        list_code2.append(code2)
        question_pair_id = index[0] ** 3 + index[1] ** 3 + index[0] + index[1]
        list_question_pair_id.append(question_pair_id)

    df_similar = df_similar.append(
        pd.DataFrame(
            {
                "code1": list_code1,
                "code2": list_code2,
                "similar": 1,
                "code1_group": i,
                "code2_group": i,
                "pair_id": list_question_pair_id,
            }
        )
    )

df_similar = df_similar.reset_index(drop=True)
df_similar["code1_group"] = df_similar["code1_group"].astype(int)
df_similar["code2_group"] = df_similar["code2_group"].astype(int)
print(df_similar.shape)

# Create different pairs
NUM = 120000
df_different = pd.DataFrame({}, columns=["code1", "code2", "similar"])

for i in tqdm(range(1, 301)):
    # get corresponding dataframe where question_label == i
    group_index = df_train[df_train["question_label"] == i].index
    other_index = df_train[df_train["question_label"] != i].index
    sample_index = np.random.choice(group_index, NUM, replace=True)
    other_sample_index = np.random.choice(other_index, NUM, replace=True)

    # make unique pair id: multiplication - sum
    question_pair_id = (
        sample_index ** 3 + other_sample_index ** 3 + (sample_index + other_sample_index)
    )
    # make group pair id
    other_group_index = df_train.iloc[other_sample_index]["question_label"].values

    # append df_train.iloc[sample_index]["code"] to df_different["code1"]
    code1 = df_train.iloc[sample_index]["code"].values
    code2 = df_train.iloc[other_sample_index]["code"].values
    df_different = df_different.append(
        pd.DataFrame(
            {
                "code1": code1,
                "code2": code2,
                "similar": 0,
                "code1_group": i,
                "code2_group": other_group_index.astype(int),
                "question_pair_id": question_pair_id,
            }
        )
    )

df_different = df_different.drop_duplicates(subset=["question_pair_id"])
df_different["code1_group"] = df_different["code1_group"].astype(int)
df_different["code2_group"] = df_different["code2_group"].astype(int)
print(df_different.shape)


# import KFold from sklearn
list_groups = list([i for i in range(1, 301)])
kf = KFold(n_splits=5, shuffle=True)
folds_10 = kf.get_n_splits(list_groups)

for fold_index, (train_index, test_index) in enumerate(kf.split(list_groups)):
    # print("TRAIN:", train_index, "TEST:", test_index)

    df_similar_train_fold = df_similar[df_similar["code1_group"].isin(train_index)]
    df_similar_val_fold = df_similar[df_similar["code1_group"].isin(test_index)]
    # print(df_similar_train_fold.shape)
    # print(df_similar_val_fold.shape)

    df_different_train_fold = df_different[
        df_different["code1_group"].isin(train_index)
        & df_different["code2_group"].isin(train_index)
    ]
    df_different_train_fold = df_different_train_fold.sample(len(df_similar_train_fold))
    df_different_val_fold = df_different[
        df_different["code1_group"].isin(test_index) & df_different["code2_group"].isin(test_index)
    ]
    df_different_val_fold = df_different_val_fold.sample(len(df_similar_val_fold))
    # print(df_different_train_fold.shape)
    # print(df_different_val_fold.shape)

    df_train = pd.concat([df_similar_train_fold, df_different_train_fold])
    # shuffle df_train
    df_train = df_train.sample(frac=1)
    # reset index for df_train
    df_train = df_train.reset_index(drop=True)

    df_val = pd.concat([df_similar_val_fold, df_different_val_fold])
    # shuffle df_val
    df_val = df_val.sample(frac=1)
    # reset index for df_val
    df_val = df_val.reset_index(drop=True)

    # df_train.to_csv("/home/ubuntu/plclassification/data/df_train_fold_" + str(fold_index) + ".csv", encoding="utf-8", index=False)
    # df_val.to_csv("/home/ubuntu/plclassification/data/df_val_fold_" + str(fold_index) + ".csv", encoding="utf-8", index=False)
    print(df_train.shape, df_val.shape)
    assert df_train.columns == df_val.columns

    dataset_fold = DatasetDict()
    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    dataset_fold["train"] = dataset_train
    dataset_fold["val"] = dataset_val

    name = f"PoolC/{fold_index+1}-fold-clone-detection-600k-5fold"
    dataset_fold.push_to_hub(name, private=True)

# make all dataset
df_train_all = pd.concat([df_train, df_different])
# shuffle df_train_all
df_train_all = df_train_all.sample(frac=1)
# reset index for df_train_all
df_train_all = df_train_all.reset_index(drop=True)

df_val_all = pd.read_csv("/home/ubuntu/plclassification/sample_train.csv")
# shuffle df_val_all
df_val_all = df_val_all.sample(frac=1)
# reset index for df_val_all
df_val_all = df_val_all.reset_index(drop=True)

print(df_train_all.shape, df_val_all.shape)
assert df_train_all.columns == df_val_all.columns

dataset_all = DatasetDict()
dataset_train = Dataset.from_pandas(df_train_all)
dataset_val = Dataset.from_pandas(df_val_all)
dataset_all["train"] = dataset_train
dataset_all["val"] = dataset_val

name = f"PoolC/all-clone-detection"
dataset_all.push_to_hub(name, private=True)
