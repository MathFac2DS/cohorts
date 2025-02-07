import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os
os.makedirs("../data", exist_ok=True)
import csv

# set the min_score for boundary of cohorts
# for example, min_score = 75, then similarity = 75/50 - 1 and length = arccos(0.5) = pi/3
min_score = 75

# set the client, embedding_key, and paths to the embedding files
client = "aipp"
embedding_key = "alchemist_embedding_lodestone-v3-client-768"
query_file_path = "data/aipp_10segments_wEMB_updated.jsonl"
keyword_embeddings_file_path = "data/aipp_taxonomy_keywords_lodestone_v3_aipp.jsonl"
profile_embeddings_file_path = "data/aipp-lodestone-v3-aipp.jsonl"

# create the paths to store the cohorts and benchmarks
cohorts_csv_file_path = f"data/{client}_cohorts_single_keyword_v3_{min_score}.csv"
cohorts_json_file_path = f"data/{client}_cohorts_single_keyword_v3_{min_score}.jsonl"
benchmarks_file_path = f"data/{client}_benchmarks_single_keyword_v3_{min_score}.csv"

# define a function to convert cosine similarity to length along the hypersphere
def score_to_length(score: float) -> float:
    return np.arccos(score/50-1)

max_length = score_to_length(min_score)

# Helper function to convert similarity to score
def similarity_to_score(similarity: float) -> float:
    return (similarity + 1) * 50.0

# Helper function to ensure embeddings are 2D
def ensure_2d(array, name="embedding"):
    if len(array.shape) == 1:
        print(f"{name} array is 1D; reshaping to 2D...")
        array = array.reshape(1, -1)
    elif len(array.shape) > 2:
        raise ValueError(f"{name} array has an invalid shape: {array.shape}")
    return array

# helper function to load the queries
def load_query_embeddings(file_path):
    queries_df = pd.read_json(file_path, lines=True)
    queries_df = queries_df[['embedding', 'min_score', 'length']]
    return queries_df

print(f"Loading the query embeddings from file path {query_file_path}...")
query_embeddings_df = load_query_embeddings(query_file_path)
query_embeddings = np.array(query_embeddings_df["embedding"].tolist())
query_embeddings = ensure_2d(query_embeddings, name="query_embeddings")
query_embeddings_norm = normalize(query_embeddings, axis=1)
query_embeddings_lengths = query_embeddings_df['length']
print("Query embeddings loaded.")
print(query_embeddings_df.head())


# define a function to efficiently load embeddings from a json file to a dataframe
# arg1 file_path (str): path to the json file
# arg2 key (list): list of keys representing the hierarchy used to extract the embedding from each json object
# returns pd.DataFrame: DataFrame with embeddings in the 'embedding' column
def load_embeddings(file_path, key, id_key):
    # generator to extract the embeddings from the json file
    def extract_embeddings(file_path, key, id_key):
        with open(file_path, "r") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    embedding_id = data.get(id_key, None) # extract the id if present
                    # navigate through the hierarchy of keys
                    for k in key:
                        data = data.get(k, {})
                        if not data:
                            break
                    if isinstance(data, list):
                        yield {"id": embedding_id, "embedding": data}
                except json.JSONDecodeError:
                    # log or handle improperly formatted lines
                    pass

    # Use the generator to create a DataFrame
    return pd.DataFrame(extract_embeddings(file_path, key, id_key))

# load and process the keyword embeddings (normalize for cosine similarity)
print("Loading and processing the keyword embeddings...")
keyword_embeddings_df = load_embeddings(keyword_embeddings_file_path, [embedding_key], "id")
keyword_embeddings = np.array(keyword_embeddings_df["embedding"].tolist())
keyword_embeddings = ensure_2d(keyword_embeddings, name="keyword_embeddings")
keyword_embeddings_norm = normalize(keyword_embeddings, axis=1)
print(f"Keyword embeddings loaded and processed.")
print(f"The length of the list of keyword embeddings is {len(keyword_embeddings)}.")
print(keyword_embeddings_df.head())

# load and process the profile embeddings (normalize for cosine similarity)
print("Loading and processing the profile embeddings...")
profile_embeddings_df = load_embeddings(profile_embeddings_file_path, ["_source", embedding_key], "_id")
profile_embeddings = np.array(profile_embeddings_df["embedding"].tolist())
profile_embeddings = ensure_2d(profile_embeddings, name="profile_embeddings")
profile_embeddings_norm = normalize(profile_embeddings, axis=1)
print(f"Profile embeddings loaded and processed.")
print(f"The length of the list of profile embeddings is {len(profile_embeddings)}.")
print(profile_embeddings_df.head())



# compute the cosine similarities for the query and cohort embeddings
print("Computing the cosine similarities for the query and keyword embeddings...")
query_similarity_matrix = cosine_similarity(query_embeddings_norm, keyword_embeddings_norm)
query_similarity_matrix = np.clip(query_similarity_matrix, -1.0, 1.0) # ensures that rounding errors do not impede computation of the arccosine
query_lengths_between = np.arccos(query_similarity_matrix)


# define a function to find the keywords within the length of the query embeddings
def get_keyword_candidates_to_cohort(query_lengths, keywords_df, lengths_between_matrix):
    keyword_candidates = []
    valid_keyword_ids = set()
    for i, query_length in enumerate(query_lengths):
        valid_indices = np.where(lengths_between_matrix[i] <= query_length)[0]
        sorted_indices = valid_indices[np.argsort(lengths_between_matrix[i][valid_indices])][:100]  # Get top 100 closest keywords
        valid_keywords = keywords_df.iloc[sorted_indices]
        keyword_list = []
        for idx, row in valid_keywords.iterrows():
            keyword_list.append({"id": row["id"], "length_between": lengths_between_matrix[i][idx]})
            valid_keyword_ids.add(row["id"])
        keyword_candidates.append({
            "query_index": i,
            "count": len(valid_keywords),
            "keywords": keyword_list
        })

    # filter keyword embeddings dataframe to include only valid keyword IDs
    global keyword_embeddings_df
    keyword_embeddings_df = keyword_embeddings_df[keyword_embeddings_df["id"].isin(valid_keyword_ids)]

    return keyword_candidates

print(f"Getting keyword candidates...")
keyword_candidates_to_cohort = get_keyword_candidates_to_cohort(query_embeddings_lengths, keyword_embeddings_df, query_lengths_between)


# redefine the list of keyword embeddings used for subsequent similarity calculations
keyword_embeddings = np.array(keyword_embeddings_df["embedding"].tolist())
keyword_embeddings = ensure_2d(keyword_embeddings, name="keyword_embeddings")
keyword_embeddings_norm = normalize(keyword_embeddings, axis=1)
print(f"The length of the list of keyword embeddings is {len(keyword_embeddings)}.")
print(f"There are {len(keyword_embeddings)} valid keyword candidates to form cohorts.")
print(keyword_embeddings_df.head())

# compute the cosine similarities for the profile and cohort embeddings
print("Computing the cosine similarities for the profile and keyword embeddings...")
profile_similarity_matrix = cosine_similarity(profile_embeddings_norm, keyword_embeddings_norm)
profile_similarity_matrix = np.clip(profile_similarity_matrix, -1.0, 1.0) # ensures that rounding errors do not impede computation of the arccosine
profile_lengths_between = np.arccos(profile_similarity_matrix)



# starting with a uniform radius given by the max_length = score_to_length(min_score)
keyword_lengths = [max_length] * len(keyword_embeddings)

# initialize tagged profiles list
tagged_profiles = []

# for each profile find the cohort_embedding
# Iterate through profiles
for profile_idx, (profile_id, profile_row) in enumerate(profile_embeddings_df.iterrows()):
    profile_embedding = profile_row["embedding"]
    profile_length_between = profile_lengths_between[profile_idx, :]  # Extract row for the profile

    # Find valid keyword indices where profile length is within keyword length
    valid_keyword_indices = np.where(profile_length_between <= max_length)[0]

    if valid_keyword_indices.size > 0:
        best_keyword_idx = valid_keyword_indices[np.argmin(profile_length_between[valid_keyword_indices])]
        cohort_id = keyword_embeddings_df.iloc[best_keyword_idx]["id"]
        cohort_embedding = keyword_embeddings_df.iloc[best_keyword_idx]["embedding"]
        cohort_type = "keyword_matched"
        profile_length = profile_length_between[best_keyword_idx]
    else:
        cohort_id = profile_id
        cohort_embedding = profile_row["embedding"]
        cohort_type = "self_cohort"
        profile_length = 0

    tagged_profiles.append({
        "profile_id": profile_id,
        "profile_embedding": profile_embedding,
        "cohort_id": cohort_id,
        "cohort_embedding": cohort_embedding,
        "cohort_type": cohort_type,
        "profile_length": profile_length
    })

# Convert tagged profiles to DataFrame and save as JSONL
tagged_profiles_df = pd.DataFrame(tagged_profiles)
tagged_profiles_file_path = f"data/{client}_tagged_profiles_single_keyword_v3_{min_score}.jsonl"
tagged_profiles_df.to_json(tagged_profiles_file_path, orient="records", lines=True)

# # Convert to DataFrame and save as csv
# tagged_profiles_df = pd.DataFrame(tagged_profiles)
# tagged_profiles_file_path = f"data/{client}_tagged_profiles_single_keyword_v3.csv"
# tagged_profiles_df.to_csv(tagged_profiles_file_path, index = False)

print(f"Tagged profiles saved to {tagged_profiles_file_path}.")



# group profiles by closest keywords to create cohorts
print("Creating cohorts...")
cohorts = []

# Convert tagged_profiles_df to dictionary lookup for efficiency
tagged_profiles_dict = (
    tagged_profiles_df.groupby("cohort_id")
    .agg({
        "cohort_embedding": "first",  # Cohort embeddings should be identical within the group
        "cohort_type": "first",  # Cohort type should be the same
        "profile_id": list,  # Collect profile IDs in a list
        "profile_length": list  # Collect all profile lengths
    })
    .to_dict(orient="index")
)


cohorts = []

for cohort_id, cohort_data in tagged_profiles_dict.items():
    cohort_embedding = cohort_data["cohort_embedding"]
    cohort_type = cohort_data["cohort_type"]
    profiles_in_cohort = cohort_data["profile_id"]  # List of profile IDs
    profile_count = len(profiles_in_cohort)
    cohort_length = max(cohort_data["profile_length"])  # Get the max profile length

    cohorts.append({
        "cohort_id": cohort_id,
        "cohort_embedding": cohort_embedding,
        "cohort_type": cohort_type,
        "profile_count": profile_count,
        "cohort_length": cohort_length
    })

# convert cohorts list to DataFrame
cohorts_df = pd.DataFrame(cohorts)

# save cohorts_df as JSONL
print(f"Saving the cohorts to the path {cohorts_json_file_path}...")
cohorts_df.to_json(cohorts_json_file_path, orient="records", lines=True)

# create cohorts csv for analytics, drop embeddings,
cohorts_keyword_matched_df = cohorts_df[cohorts_df['cohort_type'] == "keyword_matched"].copy()
cohorts_keyword_matched_df.drop(["cohort_embedding", "cohort_type"], axis=1, inplace=True)

# save the cohorts_df to a csv file
print(f"Saving cohorts to {cohorts_csv_file_path}...")
cohorts_keyword_matched_df.to_csv(cohorts_csv_file_path, index=False)

print("Cohort generation complete.")
