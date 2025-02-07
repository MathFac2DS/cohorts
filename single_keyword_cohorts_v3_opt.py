import jsonlines
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os
from joblib import Parallel, delayed

os.makedirs("../data", exist_ok=True)

# Set configurations
client = "aipp"
min_score = 75
embedding_key = "alchemist_embedding_lodestone-v3-client-768"

query_file_path = "data/aipp_10segments_wEMB_updated.jsonl"
keyword_embeddings_file_path = "data/aipp_taxonomy_keywords_lodestone_v3_aipp.jsonl"
profile_embeddings_file_path = "data/aipp-lodestone-v3-aipp.jsonl"

cohorts_csv_file_path = f"data/{client}_cohorts_single_keyword_v3_{min_score}.csv"
cohorts_json_file_path = f"data/{client}_cohorts_single_keyword_v3_{min_score}.jsonl"

# Convert cosine similarity score to length
def score_to_length(score: float) -> float:
    return np.arccos(score / 50 - 1)

max_length = score_to_length(min_score)

# Ensure embeddings are always 2D
def ensure_2d(array, name="embedding"):
    array = np.atleast_2d(array)
    if array.shape[0] == 1:
        print(f"{name} reshaped to 2D.")
    return array

# Load JSONL using jsonlines (More Efficient)
def load_jsonl(file_path, columns):
    with jsonlines.open(file_path) as reader:
        return pd.DataFrame(reader)[columns]

# Load and normalize embeddings
def load_embeddings(file_path, key_path, id_key):
    def extract_embeddings(file_path, key_path, id_key):
        with jsonlines.open(file_path) as reader:
            for line in reader:
                embedding_id = line.get(id_key)
                embedding = line
                for key in key_path:
                    embedding = embedding.get(key, {})
                if isinstance(embedding, list):
                    yield {"id": embedding_id, "embedding": embedding}

    df = pd.DataFrame(extract_embeddings(file_path, key_path, id_key))
    df["embedding"] = normalize(np.stack(df["embedding"].values), axis=1)
    return df

# Load query embeddings
print("Loading query embeddings...")
query_embeddings_df = load_jsonl(query_file_path, ["embedding", "min_score", "length"])
query_embeddings = normalize(np.stack(query_embeddings_df["embedding"].values), axis=1)
query_lengths_between = np.arccos(np.clip(cosine_similarity(query_embeddings), -1.0, 1.0))
print("Query embeddings loaded.")

# Load keyword and profile embeddings
print("Loading keyword and profile embeddings...")
keyword_embeddings_df = load_embeddings(keyword_embeddings_file_path, [embedding_key], "id")
profile_embeddings_df = load_embeddings(profile_embeddings_file_path, ["_source", embedding_key], "_id")
print(f"Loaded {len(keyword_embeddings_df)} keywords and {len(profile_embeddings_df)} profiles.")

# Compute keyword similarities
print("Computing query-keyword similarities...")
query_similarity_matrix = cosine_similarity(query_embeddings, keyword_embeddings_df["embedding"].tolist())
query_similarity_matrix = np.clip(query_similarity_matrix, -1.0, 1.0)
query_lengths_between = np.arccos(query_similarity_matrix)

# Get keyword candidates efficiently
def get_keyword_candidates(query_lengths, keywords_df, lengths_between_matrix):
    keyword_candidates = []
    valid_keyword_ids = set()

    for i, query_length in enumerate(query_lengths):
        valid_indices = np.where(lengths_between_matrix[i] <= query_length)[0]
        sorted_indices = valid_indices[np.argsort(lengths_between_matrix[i][valid_indices])][:100]

        valid_keywords = keywords_df.iloc[sorted_indices]
        keyword_list = [{"id": row["id"], "length_between": lengths_between_matrix[i][idx]} 
                        for idx, row in valid_keywords.iterrows()]
        
        valid_keyword_ids.update(valid_keywords["id"])
        keyword_candidates.append({"query_index": i, "count": len(valid_keywords), "keywords": keyword_list})

    return keyword_candidates, keywords_df[keywords_df["id"].isin(valid_keyword_ids)]

print("Finding keyword candidates...")
keyword_candidates_to_cohort, keyword_embeddings_df = get_keyword_candidates(query_embeddings_df["length"], keyword_embeddings_df, query_lengths_between)

# Compute profile-keyword similarities
print("Computing profile-keyword similarities...")
profile_similarity_matrix = Parallel(n_jobs=-1)(
    delayed(cosine_similarity)(profile_embeddings_df.iloc[i:i+5000]["embedding"].tolist(), keyword_embeddings_df["embedding"].tolist())
    for i in range(0, len(profile_embeddings_df), 5000)
)
profile_similarity_matrix = np.vstack(profile_similarity_matrix)
profile_similarity_matrix = np.clip(profile_similarity_matrix, -1.0, 1.0)
profile_lengths_between = np.arccos(profile_similarity_matrix)

# Assign profiles to cohorts
print("Assigning profiles to cohorts...")
tagged_profiles = []

for profile_idx, row in profile_embeddings_df.itertuples(index=False, name=None):
    profile_id, profile_embedding = row
    profile_length_between = profile_lengths_between[profile_idx, :]

    valid_keyword_indices = np.where(profile_length_between <= max_length)[0]
    if valid_keyword_indices.size > 0:
        best_keyword_idx = valid_keyword_indices[np.argmin(profile_length_between[valid_keyword_indices])]
        cohort_id = keyword_embeddings_df.iloc[best_keyword_idx]["id"]
        cohort_embedding = keyword_embeddings_df.iloc[best_keyword_idx]["embedding"]
        cohort_type = "keyword_matched"
        profile_length = profile_length_between[best_keyword_idx]
    else:
        cohort_id, cohort_embedding, cohort_type, profile_length = profile_id, profile_embedding, "self_cohort", 0

    tagged_profiles.append({"profile_id": profile_id, "profile_embedding": profile_embedding,
                            "cohort_id": cohort_id, "cohort_embedding": cohort_embedding,
                            "cohort_type": cohort_type, "profile_length": profile_length})

tagged_profiles_df = pd.DataFrame(tagged_profiles)
tagged_profiles_df.to_json(cohorts_json_file_path, orient="records", lines=True)
print(f"Tagged profiles saved to {cohorts_json_file_path}")

# Group profiles into cohorts
print("Grouping profiles into cohorts...")
tagged_profiles_grouped = tagged_profiles_df.groupby("cohort_id", as_index=False).agg({
    "cohort_embedding": "first", "cohort_type": "first",
    "profile_id": lambda x: list(x), "profile_length": lambda x: list(x)
})
tagged_profiles_dict = tagged_profiles_grouped.set_index("cohort_id").to_dict(orient="index")

cohorts = [{"cohort_id": k, **v, "profile_count": len(v["profile_id"]),
            "cohort_length": max(v["profile_length"])} for k, v in tagged_profiles_dict.items()]
cohorts_df = pd.DataFrame(cohorts)
cohorts_df.to_json(cohorts_json_file_path, orient="records", lines=True)

# Save cohorts to CSV
cohorts_keyword_matched_df = cohorts_df[cohorts_df["cohort_type"] == "keyword_matched"].copy()
cohorts_keyword_matched_df.drop(["cohort_embedding", "cohort_type"], axis=1, inplace=True)
cohorts_keyword_matched_df.to_csv(cohorts_csv_file_path, index=False)

print("Cohort generation complete.")
