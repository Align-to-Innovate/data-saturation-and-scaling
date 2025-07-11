import pandas as pd
import boto3
import io
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import numpy as np
from tqdm.auto import tqdm
import ast

from itertools import combinations

#source activate amplify_env2
#cd aviv/scaling_laws/
#nohup python -u Semisupervised_curves_all_years_chunkies_modulo_simpleCV.py > logs/log_simplecv_modulo.txt 2>&1 &


# Define S3 path
pgfolder = "s3://research-model-checkpoints/DMS_ProteinGym_substitutions/"

# Initialize S3 client
s3 = boto3.client("s3")

# Extract bucket name and prefix (folder path in S3)
bucket_name = "research-model-checkpoints"
prefix = "DMS_ProteinGym_substitutions/"

# List all CSV files in the S3 bucket
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
file_keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith("_with_scoresandembeddings_allyears.csv")]

print(f"Found {len(file_keys)} CSV files in S3.")

# Helper to parse list-like strings into NumPy arrays
def parse_list_column(x):
    return np.array(ast.literal_eval(x))

# Okay now let's try splitting on the length of the protein rather than random splits

# First, we need to make a position column and also filter out multi-mutants
# Then train/test based on that range

def preprocess_and_evaluate_cv5(df,
                                representation_col,
                                model_label="model",
                                alpha=1.0,
                                n_groups=5,
                                random_state=42):
    """
    Random-group 5-fold CV:
      • Each position → random group (0–4) using `random_state`.
      • For each of the 5 groups:   train on the other 4, test on the held-out one.
      • Returns {'cv5_<model_label>': mean_spearman}.
    """
    # --- 1. keep only single mutants and extract position ---------------------
    df = df[~df["mutant"].str.contains(":")].copy()
    df["position"] = df["mutant"].str.extract(r"(\d+)").astype(int)

    if df["position"].nunique() < 2:
        return {}

    # --- 2. assign positions to random groups --------------------------------
    rng = np.random.default_rng(random_state)
    unique_pos = df["position"].unique()
    pos2grp = dict(zip(unique_pos, rng.integers(0, n_groups, len(unique_pos))))
    df["group_id"] = df["position"].map(pos2grp)

    # --- 3. prepare data ------------------------------------------------------
    X_all = np.stack(df[representation_col].values)
    y_all = df["DMS_score"].values

    spearman_scores = []
    STD_THRESHOLD = 1e-5

    # --- 4. 5-fold CV (train on 4, test on 1) --------------------------------
    for test_group in range(n_groups):
        train_mask = df["group_id"] != test_group
        test_mask  = ~train_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        if len(y_test) == 0:
            continue  # just in case an empty fold sneaks in

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #if np.std(y_pred) < STD_THRESHOLD or np.std(y_test) < STD_THRESHOLD:
            #print(f"[WARNING] Nearly constant input for Spearman (test_group={test_group}).")
            #continue

        spearman, _ = spearmanr(y_test, y_pred)
        spearman_scores.append(spearman)

    if not spearman_scores:
        return {}

    return {f"cv5_{model_label}": np.nanmean(spearman_scores)}


# Got this working now, though the one-hot often has y_pred that is all identical... 
# Need to dig into that!!
output_path = "semisup_results/modulo_eval_120M_allyears_ridge_simpleCV.csv"
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
        results = preprocess_and_evaluate_cv5(df, "one_hot", model_label="onehot")
        row_result.update(results)

        # Evaluate each embedding column by year
        for year in range(2011, 2025):
            col_name = f"embedding_120M_{year}"
            if col_name in df.columns:
                results = preprocess_and_evaluate_cv5(df, col_name, model_label=str(year))
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


