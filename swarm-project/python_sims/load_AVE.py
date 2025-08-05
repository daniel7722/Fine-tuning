import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load(): 
    df = pd.read_csv("./data/AVE_Dataset/annotations.txt", sep="&", engine="python")
    df.columns = ["Category", "VideoID", "Quality", "StartTime", "EndTime"]
    print(df)
    return df

def split(df): 
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df): 
    train_df.to_csv("./data/AVE_Dataset/splits/train.csv", index=False)
    val_df.to_csv("./data/AVE_Dataset/splits/val.csv", index=False)
    test_df.to_csv("./data/AVE_Dataset/splits/test.csv", index=False)
    print("Saved train.csv, val.csv, and test.csv")

df = load()
train_df, val_df, test_df = split(df)
save_splits(train_df, val_df, test_df)