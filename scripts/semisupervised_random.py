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

# Initialize S3 client
s3 = boto3.client("s3")

# Replace with your bucket name and prefix (if applicable)
bucket_name = "your-s3-bucket-name"
prefix = "your-prefix-to-data-files"

# List all CSV files in the S3 bucket
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
# Get the file keys (can reaplce the .endswith if you have a different siffix for processed files
file_keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith("_with_scoresandembeddings_allyears.csv")]

print(f"Found {len(file_keys)} CSV files in S3.")

# Helper to parse list-like strings into NumPy arrays
def parse_list_column(x):
    return np.array(ast.literal_eval(x))

# Evaluation function
def evaluate_by_train_fraction(df, representation_col, alpha=1.0, train_sizes=np.arange(0.1, 1.0, 0.1), n_repeats=5, random_state=36):
    X_all = np.stack(df[representation_col].values)  # Combine array column into a 2D matrix
    y_all = df["DMS_score"].values

    results = {}
    rng = np.random.RandomState(random_state)

    for train_frac in train_sizes:
        spearman_scores = []

        for repeat in range(n_repeats):
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, train_size=train_frac, random_state=rng.randint(0, 1e6)
            )

            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if np.std(y_pred) == 0 or np.std(y_test) == 0:
                spearman = np.nan
            else:
                spearman, _ = spearmanr(y_test, y_pred)

            spearman_scores.append(spearman)

        mean_spearman = np.nanmean(spearman_scores)
        results[f"{int(train_frac * 100)}%_train"] = mean_spearman

    return results


output_path = "../semisupervised_results/onehot_120M_allyears_ridge.csv"
write_header = not os.path.exists(output_path)

# Main processing loop (using S3)
for key in tqdm(file_keys, desc="Processing all files"):
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=key)

        # Build converters for each year and one_hot
        converters = {f"embedding_120M_{year}": parse_list_column for year in range(2011, 2025)}
        converters["one_hot"] = parse_list_column

        df = pd.read_csv(
            io.BytesIO(obj['Body'].read()),
            converters=converters
        )

        row_result = {"file": key.split("/")[-1]}

        # Evaluate one_hot
        results = evaluate_by_train_fraction(df, "one_hot")
        for k, v in results.items():
            row_result[f"{k}_onehot"] = v

        # Evaluate each embedding column by year
        for year in range(2011, 2025):
            col_name = f"embedding_120M_{year}"
            if col_name in df.columns:
                results = evaluate_by_train_fraction(df, col_name)
                for k, v in results.items():
                    row_result[f"{k}_{year}"] = v
            else:
                print(f"Column {col_name} not found in {key}")

        # Append this row to CSV
        pd.DataFrame([row_result]).to_csv(
            output_path,
            mode='a',
            header=write_header,
            index=False
        )
        write_header = False  # Only write the header once

    except Exception as e:
        print(f"Error processing {key}: {e}")