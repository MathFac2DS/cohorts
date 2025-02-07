import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import os
os.makedirs("../data", exist_ok=True)
import csv

# set the client, embedding_key, and paths to the embedding files
client = "aipp"
min_score = 75 # match min_score from the cohort generation process
embedding_key = "alchemist_embedding_lodestone-v3-client-768"
query_file_path = "data/aipp_10segments_wEMB_updated.jsonl"

# retrieve data from create_cohorts_single_keyword_v2 artifacts
tagged_profiles_file_path = f"data/{client}_tagged_profiles_single_keyword_v3_{min_score}.jsonl"
cohort_file_path = f"data/{client}_cohorts_single_keyword_v3_{min_score}.jsonl"
benchmarks_file_path = f"data/{client}_benchmarks_single_keyword_v3_{min_score}.csv"

# helper function to ensure embeddings are 2D
def ensure_2d(array, name):
    if len(array.shape) == 1:
        print(f"{name} array is 1D; reshaping to 2D...")
        array = array.reshape(1, -1)
    elif len(array.shape) > 2:
        raise ValueError(f"{name} array has an invalid shape: {array.shape}")
    return array

# helper function to load the profiles
# fields of the tagged profiles jsonl are profile_id, profile_embedding, cohort_id, cohort_embedding, cohort_type, and profile_length
def load_profile_embeddings(file_path):
    cohorts_df = pd.read_json(file_path, lines=True)
    cohorts_df = cohorts_df[['profile_id', 'profile_embedding', 'cohort_id', 'cohort_embedding', 'cohort_type', 'profile_length']]
    return cohorts_df

print(f"Loading the tagged profile embeddings from file path {tagged_profiles_file_path}...")
profile_embeddings_df = load_profile_embeddings(tagged_profiles_file_path)
profile_ids = profile_embeddings_df["profile_id"].tolist()
profile_embeddings = np.array(profile_embeddings_df["profile_embedding"].tolist())
profile_embeddings = ensure_2d(profile_embeddings, name="profile_embeddings")
profile_embeddings_norm = normalize(profile_embeddings, axis=1)
profile_embeddings_df['profile_embedding'] = profile_embeddings_norm
profile_lengths = profile_embeddings_df['profile_length']
profile_cohort_ids = profile_embeddings_df["cohort_id"].tolist()
profile_cohort_types = profile_embeddings_df["cohort_type"].tolist()
print(profile_embeddings_df.head())


# helper function to load the queries
# fields of the tagged profiles jsonl are client, id, embedding, min_score, and length
def load_query_embeddings(file_path):
    queries_df = pd.read_json(file_path, lines=True)
    queries_df = queries_df[['client', 'id', 'embedding', 'min_score', 'length']]
    return queries_df

print(f"Loading the query embeddings from file path {query_file_path}...")
query_embeddings_df = load_query_embeddings(query_file_path)
query_embeddings = np.array(query_embeddings_df["embedding"].tolist())
query_embeddings = ensure_2d(query_embeddings, name = "query_embedding")
query_embeddings_norm = normalize(query_embeddings, axis = 1)
query_embeddings_df["embedding"] = query_embeddings_norm
query_embeddings_lengths = query_embeddings_df['length']
print(query_embeddings_df.head())


# helper function to load the cohorts
# fields of the cohorts jsonl are cohort_id, cohort_embedding, cohort_type, profile_count, and cohort_length
def load_cohort_embeddings(file_path):
    cohorts_df = pd.read_json(file_path, lines=True)
    cohorts_df = cohorts_df[['cohort_id', 'cohort_embedding', 'cohort_type', 'profile_count', 'cohort_length']]
    return cohorts_df

print(f"Loading the cohort embeddings from file path {cohort_file_path}...")
cohort_embeddings_df = load_cohort_embeddings(cohort_file_path)
cohort_ids = cohort_embeddings_df["cohort_id"].tolist()
cohort_embeddings = np.array(cohort_embeddings_df["cohort_embedding"].tolist())
cohort_embeddings = ensure_2d(cohort_embeddings, name = "cohort_embedding")
cohort_embeddings_norm = normalize(cohort_embeddings, axis = 1)
cohort_embeddings_df['cohort_embedding'] = cohort_embeddings_norm
cohort_types = cohort_embeddings_df["cohort_type"].tolist()
profile_counts = np.array(cohort_embeddings_df["profile_count"].tolist())
cohort_lengths = np.array(cohort_embeddings_df["cohort_length"].tolist())
print(cohort_embeddings_df.head())



# compute the cosine similarities for the query and cohort embeddings
print("Computing the cosine similarities for the query and cohort embeddings...")
cohort_similarity_matrix = cosine_similarity(query_embeddings, cohort_embeddings)
cohort_similarity_matrix = np.clip(cohort_similarity_matrix, -1.0, 1.0) # ensures that rounding errors do not impede computation of the arccosine
query_to_cohort_lengths_between = np.arccos(cohort_similarity_matrix)


def tag_profiles_by_cohorts(query_df, cohort_df, profile_df, query_lengths_between):
    tagged_profiles_df = pd.DataFrame(profile_ids, columns=['profile_id'])

    for i, query in query_df.iterrows():
        query_name = f"q{i+1}_cohort"
        query_length = query['length']

        length_between = query_lengths_between[i]
        inclusion_mask = length_between <= query_length
        included_cohorts = cohort_df.loc[inclusion_mask, 'cohort_id']

        profile_df['tag'] = profile_df['cohort_id'].isin(included_cohorts).map({True: 1, False: 0})
        tagged_profiles_df[query_name] = profile_df['tag'].values

    return tagged_profiles_df


print("Tagging profiles based on query-cohort distances...")
tagged_profiles_cohort = tag_profiles_by_cohorts(query_embeddings_df, cohort_embeddings_df, profile_embeddings_df, query_to_cohort_lengths_between)
print(tagged_profiles_cohort.head())


# compute the cosine similarities for the query and profile embeddings
print("Computing the cosine similarities for the query and profile embeddings...")
profile_similarity_matrix = cosine_similarity(query_embeddings, profile_embeddings)
profile_similarity_matrix = np.clip(profile_similarity_matrix, -1.0, 1.0) # ensures that rounding errors do not impede computation of the arccosine
query_to_profile_lengths_between = np.arccos(profile_similarity_matrix)


def tag_profiles_brute(query_df, profile_df, query_lengths_between):
    tagged_profiles_df = pd.DataFrame(profile_ids, columns=['profile_id'])

    for i, query in query_df.iterrows():
        query_name = f"q{i+1}_brute"
        query_length = query['length']

        length_between = query_lengths_between[i]
        inclusion_mask = length_between <= query_length
        included_profiles = profile_df.loc[inclusion_mask, 'profile_id']

        profile_df['tag'] = profile_df['profile_id'].isin(included_profiles).map({True: 1, False: 0})
        tagged_profiles_df[query_name] = profile_df['tag'].values

    return tagged_profiles_df


print("Tagging profiles based on query-profile lengths...")
tagged_profiles_brute = tag_profiles_brute(query_embeddings_df, profile_embeddings_df, query_to_profile_lengths_between)
print(tagged_profiles_brute.head())

def evaluate_truth_table(tagged_cohort_df, tagged_brute_df):
    # set the 'profile_id' as index for both dataframes
    tagged_brute_df = tagged_brute_df.set_index('profile_id')
    tagged_cohort_df = tagged_cohort_df.set_index('profile_id')

    # initialize the truth_table_df
    truth_table_df = pd.DataFrame(profile_ids, columns=['profile_id'])

    # convert columns to arrays for quick lookup
    brute_values = tagged_brute_df.to_numpy()
    cohort_values = tagged_cohort_df.to_numpy()

    truths = []
    # iterate over the profile_ids and compute truth values
    for i, profile_id in enumerate(profile_ids):
        brute_row = brute_values[i]  # get the brute values for this profile
        cohort_row = cohort_values[i]  # get the cohort values for this profile

        # vectorized computation of total and difference for all queries
        total = brute_row + cohort_row
        difference = brute_row - cohort_row

        # Apply the truth table conditions
        profile_truths = np.select(
            [
                (total == 2) & (difference == 0),
                (total == 1) & (difference == -1),
                (total == 1) & (difference == 1),
                (total == 0) & (difference == 0)
            ],
            ['TP', 'FP', 'FN', 'TN'],
            default='Unknown'
        )

        truths.append(profile_truths)

    # convert the list of truths to a DataFrame and assign appropriate column names
    truth_values_df = pd.DataFrame(truths, columns=[f'q{i+1}' for i in range(len(query_embeddings))])

    # concatenate the profile_id column with the truth values
    truth_table_df = pd.concat([truth_table_df, truth_values_df], axis=1)

    return truth_table_df


print("Computing truth table...")
truth_table = evaluate_truth_table(tagged_profiles_cohort, tagged_profiles_brute)
print(truth_table.head())

print(f"Saving the truth table...")
truth_table_file_path = f"data/{client}_truth_table_single_keyword_v3_{min_score}.csv"
truth_table.to_csv(truth_table_file_path, index = False)
print(f"Truth table for {client} saved to path {truth_table_file_path}.")