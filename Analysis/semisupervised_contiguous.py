# This script is for training linear regression models with contiguous data splits

import pandas as pd
import boto3
import io
import os
import numpy as np
from tqdm.auto import tqdm
import ast
from itertools import combinations
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

# Replace with your bucket name and prefix (if applicable) and file paths
bucket_name = "your-s3-bucket-name"
prefix = "your-prefix-to-data-files"
output_path = "path/to/Results/semisupervised_results/chunk_eval_120M_allyears_ridge.csv"


# Initialize S3 client
s3 = boto3.client("s3")

# List all CSV files in the S3 bucket
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
# Get the file keys (can reaplce the .endswith if you have a different siffix for processed files
file_keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith("_with_scoresandembeddings_allyears.csv")]

print(f"Found {len(file_keys)} CSV files in S3.")

# Helper to parse list-like strings into NumPy arrays
def parse_list_column(x):
    return np.array(ast.literal_eval(x))

# Splitting on the length of the protein -contiguous- rather than random splits

# First, we need to make a position column and also filter out multi-mutants
# Then train/test based on that range

def preprocess_and_evaluate_by_chunk(df, representation_col, model_label="model", alpha=1.0, n_chunks=5):
    # 1. Preprocessing: filter to single mutants
    df = df[~df["mutant"].str.contains(":")].copy()
    df["position"] = df["mutant"].str.extract(r'(\d+)').astype(int)
    #print(df.head)
    if df["position"].nunique() < 2:
        return {}

    # 2. Define protein range
    pos_min = df["position"].min()
    pos_max = df["position"].max()
    print(pos_min, "<- min and max ->", pos_max)
    protein_length = pos_max - pos_min + 1
    chunk_size = protein_length // n_chunks

    # 3. Assign chunk ID
    chunk_ids = np.zeros(len(df), dtype=int)
    for i in range(n_chunks):
        start = pos_min + i * chunk_size
        end = pos_min + (i + 1) * chunk_size - 1 if i < n_chunks - 1 else pos_max
        in_chunk = (df["position"] >= start) & (df["position"] <= end)
        chunk_ids[in_chunk] = i
    df["chunk_id"] = chunk_ids
    print(df["chunk_id"].value_counts())
    # 4. Prepare data
    X_all = np.stack(df[representation_col].values)
    y_all = df["DMS_score"].values

    grouped_results = {}

    # 5. Try all combinations of train/test splits
    for k in range(1, n_chunks):  # Train on 1 to 4 chunks
        print(f"k: {k}")
        spearman_scores = []

        for train_chunks in combinations(range(n_chunks), k):
            print(f"train_chunks: {train_chunks}")
            train_mask = df["chunk_id"].isin(train_chunks)
            test_mask = ~train_mask

            X_train, X_test = X_all[train_mask], X_all[test_mask]
            y_train, y_test = y_all[train_mask], y_all[test_mask]

            if len(y_train) == 0 or len(y_test) == 0:
                continue

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            print(y_train[0:5])
            print(y_pred[0:5])
            print(y_test[0:5])
            
            STD_THRESHOLD = 1e-5  # or 0.00005

            if np.std(y_pred) < STD_THRESHOLD or np.std(y_test) < STD_THRESHOLD:
                print(f"[WARNING] Nearly constant input for Spearman: train_chunks={train_chunks}, std_y_test={np.std(y_test):.5f}, std_y_pred={np.std(y_pred):.5f}")
                continue
            else:
                print(f"std y_test: {np.std(y_test):.4f}, std y_pred: {np.std(y_pred):.4f}")

                spearman, _ = spearmanr(y_test, y_pred)

            spearman_scores.append(spearman)

        if spearman_scores:
            mean_spearman = np.nanmean(spearman_scores)
            grouped_results[f"{k}chunk_train_{model_label}"] = mean_spearman

    return grouped_results


write_header = not os.path.exists(output_path)

# Collect all rows here to write at once
all_results = []

for key in tqdm(file_keys, desc="Processing all files"):
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)

        # Parse columns
        converters = {f"embedding_120M_{year}": parse_list_column for year in range(2011, 2025)}
        converters["one_hot"] = parse_list_column

        df = pd.read_csv(io.BytesIO(obj['Body'].read()), converters=converters)

        row_result = {"file": key.split("/")[-1]}

        # Evaluate one_hot
        results = preprocess_and_evaluate_by_chunk(df, "one_hot", model_label="onehot")
        row_result.update(results)

        # Evaluate each embedding column by year
        for year in range(2011, 2025):
            col_name = f"embedding_120M_{year}"
            if col_name in df.columns:
                results = preprocess_and_evaluate_by_chunk(df, col_name, model_label=str(year))
                row_result.update(results)
            else:
                print(f"Column {col_name} not found in {key}")

        # Add to master list
        all_results.append(row_result)

    except Exception as e:
        print(f"Error processing {key}: {e}")

# Save everything to one CSV
pd.DataFrame(all_results).to_csv(
    output_path,
    mode='a',
    header=write_header,
    index=False
)


