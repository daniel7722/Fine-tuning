import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42  # keep deterministic across runs

def load():
    df = pd.read_csv("./data/AVE_Dataset/annotations.txt", sep="&", engine="python")
    df.columns = ["Category", "VideoID", "Quality", "StartTime", "EndTime"]
    print("Total samples:", len(df), "Num classes:", df["Category"].nunique())
    return df

def split_four_way_with_valpre(df):
    # shuffle once (reproducible)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    y = df["Category"]

    # First: hold out TEST (20%)
    df_rest, df_test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Second: from rest, hold out VAL (20% of total -> 25% of the remaining 80%)
    y_rest = df_rest["Category"]
    df_traincombo, df_val = train_test_split(
        df_rest, test_size=0.25, random_state=RANDOM_STATE, stratify=y_rest
    )

    # Remaining is 60% total. Split evenly into TRAIN_PRE (30%) and TRAIN_FUSE (30%)
    y_traincombo = df_traincombo["Category"]
    df_train_pre, df_train_fuse = train_test_split(
        df_traincombo, test_size=0.50, random_state=RANDOM_STATE, stratify=y_traincombo
    )

    # Finally: carve out VAL_PRE (20% of TRAIN_PRE) to validate agent pretraining
    y_train_pre = df_train_pre["Category"]
    df_train_pre_main, df_val_pre = train_test_split(
        df_train_pre, test_size=0.20, random_state=RANDOM_STATE, stratify=y_train_pre
    )

    print(
        f"Splits -> train_pre:{len(df_train_pre_main)}  "
        f"val_pre:{len(df_val_pre)}  train_fuse:{len(df_train_fuse)}  "
        f"val:{len(df_val)}  test:{len(df_test)}"
    )

    return df_train_pre_main, df_val_pre, df_train_fuse, df_val, df_test

def save_splits(train_pre, val_pre, train_fuse, val_df, test_df):
    outdir = "./data/AVE_Dataset/splits"
    train_pre.to_csv(f"{outdir}/train_pre.csv", index=False)
    val_pre.to_csv(f"{outdir}/val_pre.csv", index=False)
    train_fuse.to_csv(f"{outdir}/train_fuse.csv", index=False)
    val_df.to_csv(f"{outdir}/val.csv", index=False)
    test_df.to_csv(f"{outdir}/test.csv", index=False)
    print("Saved: train_pre.csv, val_pre.csv, train_fuse.csv, val.csv, test.csv")

if __name__ == "__main__":
    df = load()
    train_pre, val_pre, train_fuse, val_df, test_df = split_four_way_with_valpre(df)
    for name, part in [("train_pre", train_pre), ("val_pre", val_pre), ("train_fuse", train_fuse), ("val", val_df), ("test", test_df)]:
        print(name, part["Category"].value_counts().sort_index().to_dict())
    save_splits(train_pre, val_pre, train_fuse, val_df, test_df)