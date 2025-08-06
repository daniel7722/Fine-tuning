import random
import pickle
import orjson
import time
from pathlib import Path

def load_data(): 

    # Load preprocessed AVE dataset with pickle caching
    train_path = Path("./data/AVE_Dataset/processed/train.jsonl")
    val_path = Path("./data/AVE_Dataset/processed/val.jsonl")
    test_path = Path("./data/AVE_Dataset/processed/test.jsonl")
    cache_dir = Path("./data/AVE_Dataset/processed")
    cache_train = cache_dir / "train.pkl"
    cache_val = cache_dir / "val.pkl"
    cache_test = cache_dir / "test.pkl"

    # Function to load or cache JSONL data
    def load_or_cache_jsonl(jsonl_path, pkl_path):
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                data = [orjson.loads(line) for line in f]
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            return data
    start_time = time.time()
    print("Loading data (with cache)...")
    train_data = load_or_cache_jsonl(train_path, cache_train)
    val_data = load_or_cache_jsonl(val_path, cache_val)
    test_data = load_or_cache_jsonl(test_path, cache_test)
    print("Done loading data.")
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data