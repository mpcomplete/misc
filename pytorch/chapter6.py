import pandas as pd
from spamtrain_download import *

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

# Make every class the same size by dropping extras.
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df
balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())

# Rename labels into token IDs.
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

def random_split(df, train_frac, val_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df

train_df, val_df, test_df = random_split(balanced_df, .7, .1)
train_df.to_csv("train.csv", index=None)
val_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
