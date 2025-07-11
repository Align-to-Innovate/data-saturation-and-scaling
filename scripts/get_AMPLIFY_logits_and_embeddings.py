# Get embeddings for all DMS substitutions in ProteinGym
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import pandas as pd
import numpy as np
import os
import datetime
import boto3 #for accessing s3
import s3fs
from concurrent.futures import ThreadPoolExecutor, as_completed


print(f"[{datetime.datetime.now()}] All packaged imported.")

from botocore.config import Config
my_config = Config(
    read_timeout=300,
    connect_timeout=120,
    retries={'max_attempts': 10}
)
print(f"[{datetime.datetime.now()}] S3 config intiialized.")


# Define S3 path
pgfolder = "s3://research-model-checkpoints/DMS_ProteinGym_substitutions/"

# Initialize S3 client
s3 = boto3.client('s3', config=my_config)

# Create a custom boto3 session with that config
session = boto3.session.Session()
s3_client = session.client('s3', config=my_config)

# Pass it to s3fs
fs = s3fs.S3FileSystem(session=session)


# Extract bucket name and prefix (folder path in S3)
bucket_name = "research-model-checkpoints"
prefix = "DMS_ProteinGym_substitutions/"

# List all CSV files in the S3 bucket
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
file_keys = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".csv")]

print(f"[{datetime.datetime.now()}] Found {len(file_keys)} CSV files in S3.")


# Load models and tokenizers as specified
print(f"[{datetime.datetime.now()}] Loading models...")

model120 = AutoModel.from_pretrained(f"chandar-lab/AMPLIFY_120M", trust_remote_code=True)
# Load tokenizerS
tokenizer120 = AutoTokenizer.from_pretrained(f"chandar-lab/AMPLIFY_120M", trust_remote_code=True)
# Move the model to GPU (required due to Flash Attention)
model120 = model120.to("cuda").eval() # Use eval mode to prevent dropout

model350 = AutoModel.from_pretrained(f"chandar-lab/AMPLIFY_350M", trust_remote_code=True)
# Load tokenizer
tokenizer350 = AutoTokenizer.from_pretrained(f"chandar-lab/AMPLIFY_350M", trust_remote_code=True)
# Move the model to GPU (required due to Flash Attention)
model350 = model350.to("cuda").eval()

print(f"[{datetime.datetime.now()}] Models loaded successfully.")

# Function to count mutations
def count_mutations(mutation_str):
    """Count the number of mutations in a sequence string like 'M1A:D2C'."""
    if pd.isna(mutation_str) or mutation_str.strip() == "":
        return 0  # Return 0 if mutation string is empty or NaN
    return len(mutation_str.split(":"))

# Function to get embeddings
def get_embedding(sequence, tokenizer, model):
    """Get sequence embeddings from the last layer of the model."""
    input_tokens = tokenizer.encode(sequence, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(input_tokens, output_hidden_states=True)  # Get hidden states
        embedding = output.hidden_states[-1].mean(dim=1).cpu().numpy().flatten()  # Mean-pool across sequence length
    
    return embedding.tolist()

# Function to get sequence log probability
def get_sequence_log_probability(sequence, tokenizer, model):
    """Compute the log probability of a sequence."""
    input_tokens = tokenizer.encode(sequence, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model(input_tokens)
        logits = output["logits"]
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    input_ids = input_tokens[0]  # Token IDs of amino acids
    token_probs = probabilities[0, torch.arange(len(input_ids)), input_ids]

    seq_log_prob = token_probs.log().sum().item()  # Sum log probabilities

    return seq_log_prob

# Function to generate one-hot encoding for sequences
def one_hot_encode_sequence(sequence, aa_alphabet="ACDEFGHIKLMNPQRSTVWY"):
    """Generate a one-hot encoded matrix for a given protein sequence."""
    aa_to_index = {aa: i for i, aa in enumerate(aa_alphabet)}
    one_hot_matrix = np.zeros((len(sequence), len(aa_alphabet)))

    for i, aa in enumerate(sequence):
        if aa in aa_to_index:  # Ignore unknown characters
            one_hot_matrix[i, aa_to_index[aa]] = 1
    
    return one_hot_matrix.flatten().tolist()  # Flatten for storing in CSV




# Length limit of AMPLIFY is 2048 amino acids 
# How many proteins are longer than this?
# This code chunk takes like 1min so be patient
prot_lens_dict = {}

for file_key in file_keys:
    s3_path = f"s3://{bucket_name}/{file_key}"
    # Only load in one row and just the column of interest beause this process is kind of slow
    df = pd.read_csv(s3_path, storage_options={"anon": False}, nrows=2, usecols=['mutated_sequence'])

    if len(df) > 1:  # Ensure there's a second row
        prot_lens_dict[file_key] = len(str(df['mutated_sequence'].iloc[1]))  # Get second row

sorted_lengths = sorted(prot_lens_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 longest sequences:")
for file, length in sorted_lengths:
    print(f"{file}: {length} amino acids")

# Only process DMSs with proteins of suitable length
# And also do not process any files that have already been processed! 
filtered_file_keys = [
    key for key in file_keys 
    if key.endswith(".csv") 
    and prot_lens_dict[key] <= 2048 
    #ignore what has already been processed
    and not key.endswith("_with_scoresandembeddings.csv")
    and not key.endswith("_with_scoresandembeddings_allyears.csv")
    and not key.endswith("_with_scoresandembeddings_allyearsTEST.csv")
]
len(filtered_file_keys)

#fs = s3fs.S3FileSystem()  # Enables pandas to read/write directly from S3

for file_key in filtered_file_keys:
    s3_path = f"s3://{bucket_name}/{file_key}"

    # Check if file has already been processed
    new_filename = file_key.replace(".csv", "_with_scoresandembeddings.csv")
    new_s3_path = f"s3://{bucket_name}/{new_filename}"

    if fs.exists(new_s3_path):
        #print(f"[{datetime.datetime.now()}] Skipping {s3_path} (Already Processed)")
        continue  # Skip if the file is already processed

    print(f"[{datetime.datetime.now()}] Processing {s3_path}")

    chunk_size = 5000  # Adjust based on available memory

    for chunk in pd.read_csv(s3_path, storage_options={"anon": False}, chunksize=chunk_size):  # ✅ FIXED LINE
        # Process chunk instead of full DataFrame
        chunk["num_mutations"] = chunk["mutant"].apply(count_mutations)
        chunk["one_hot"] = chunk["mutated_sequence"].apply(one_hot_encode_sequence)
    
        chunk["embedding_120M"] = chunk["mutated_sequence"].apply(lambda seq: get_embedding(seq, tokenizer120, model120))
        chunk["log_prob_120M"] = chunk["mutated_sequence"].apply(lambda seq: get_sequence_log_probability(seq, tokenizer120, model120))
    
        chunk["embedding_350M"] = chunk["mutated_sequence"].apply(lambda seq: get_embedding(seq, tokenizer350, model350))
        chunk["log_prob_350M"] = chunk["mutated_sequence"].apply(lambda seq: get_sequence_log_probability(seq, tokenizer350, model350))
    
        # Append processed chunk to S3 (avoid keeping everything in memory)
        with fs.open(new_s3_path, "a") as f:  # "a" mode appends to file
            chunk.to_csv(f, index=False, header=not fs.exists(new_s3_path))
    
    print(f"[{datetime.datetime.now()}] Saved {new_s3_path}")

print("Processing with 120M and 350M complete.")

# Load all of the other models now 
# Dictionary them to keep organization clean 
models = {}
tokenizers = {}

for year in range(2011, 2025):
    model_name = f"chandar-lab/AMPLIFY_120M"
    revision = f"AMPLIFY_120M_{year}"
    
    models[year] = AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=True).to("cuda").eval()
    tokenizers[year] = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)

print("All models loaded")

# Now, with those files completed also generate embeddings and score with the other yearly models 
# Still underconstruction!!! 

# Define your S3 bucket and get filtered file keys
filtered_file_keys = [key for key in file_keys if key.endswith("_with_scoresandembeddings.csv")]
print(f"[{datetime.datetime.now()}] Found {len(filtered_file_keys)} CSV files to process with all years.")

#reverse it
filtered_file_keys.reverse()
filtered_file_keys = filtered_file_keys[40:]

# PARALLELIZED FILE TEST WITH ALL MODELS IN S3

# Disable Hugging Face tokenizer parallelism to avoid conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# S3 Configuration
#fs = s3fs.S3FileSystem(anon=False)  # Ensure authentication

# Get all files from S3 that match the pattern

print(f"Found {len(filtered_file_keys)} files in S3 to process.")

chunk_size = 5000  # Adjust based on available memory
years = list(range(2011, 2025))  # Years to process


# Function to process a single sequence with a specific model year
def process_sequence(seq, year):
    try:
        model = models[year]
        tokenizer = tokenizers[year]
        embedding = get_embedding(seq, tokenizer, model)
        log_prob = get_sequence_log_probability(seq, tokenizer, model)
        return year, embedding, log_prob
    except Exception as e:
        print(f"Error processing sequence {seq} for year {year}: {e}")
        return year, None, None  # Ensure failures don't crash the pipeline


# Function to process a single file (S3 version)
def process_file(file_key):
    s3_path = f"s3://{bucket_name}/{file_key}"

    # Define the new output filename
    new_filename = file_key.replace("_with_scoresandembeddings.csv", "_with_scoresandembeddings_allyears.csv")
    new_s3_path = f"s3://{bucket_name}/{new_filename}"

    # Skip if already processed
    if fs.exists(new_s3_path):
        print(f"[{datetime.datetime.now()}] Skipping {new_s3_path} (Already Processed)")
        return

    print(f"[{datetime.datetime.now()}] Processing {s3_path}")

    # Read CSV in chunks from S3
    for chunk in pd.read_csv(s3_path, storage_options={"anon": False}, chunksize=chunk_size):
        all_results = []

        # Process each sequence in the chunk using parallelized model calculations
        with ThreadPoolExecutor() as executor:
            future_to_seq = {
                executor.submit(process_sequence, seq, year): (seq, year)
                for seq in chunk["mutated_sequence"] for year in years
            }

            for future in future_to_seq:
                seq, year = future_to_seq[future]
                try:
                    result = future.result()
                    if result and len(result) == 3:  # Ensure we received (year, embedding, log_prob)
                        all_results.append((seq, *result))  # Store sequence for merging later
                except Exception as e:
                    print(f"Error processing {seq} for year {year}: {e}")

        # Convert results into DataFrame for merging
        results_df = pd.DataFrame(all_results, columns=["mutated_sequence", "year", "embedding", "log_prob"])

        # Drop any failed computations before pivoting
        results_df.dropna(subset=["embedding", "log_prob"], inplace=True)

        if results_df.empty:
            print(f"Warning: No valid embeddings/log_probs for {s3_path}")
            continue  # Skip saving if nothing to process

        # Pivot the results so each year becomes a column
        results_pivot = results_df.pivot(index="mutated_sequence", columns="year", values=["embedding", "log_prob"])

        # Flatten column names
        results_pivot.columns = [f"{col[0]}_120M_{col[1]}" for col in results_pivot.columns]

        # Merge with the original chunk
        chunk = chunk.merge(results_pivot, on="mutated_sequence", how="left")

        # Append processed chunk to new file in S3
        with fs.open(new_s3_path, "a") as f:  # "a" mode appends to file
            chunk.to_csv(f, index=False, header=not fs.exists(new_s3_path))

    print(f"[{datetime.datetime.now()}] Saved {new_s3_path}")


# Process files sequentially (one at a time)
for file_key in filtered_file_keys:
    process_file(file_key)

print(f"[{datetime.datetime.now()}] Processing with all yearly models complete.")

