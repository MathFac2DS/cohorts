# This file will contain the algorithms used to perform the cohort querying process.
# Each algorithm will take an array of query embeddings, a list of cohort objects, a list of tagged profile objects, a query radius, and a cohort radius as inputs and will output a list containing the number of cosine similarities calculated for each query along with an array whose length is equal to the number of queries and whose rows are each a binary list of 0s and 1s whose length is equal to the number of profiles (i.e. Q x P).

from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cohort_utils import distance_to_length, length_to_distance, distance_to_similarity
from cohort_distance_calculator import solve_for_d_by_coverage
from collections import OrderedDict, defaultdict
import pandas as pd


# this function performs the brute force baseline search and does not follow the input convention described above
def brute_force_search(queries, tagged_profiles, min_cosine_similarity):
    query_embeddings = np.array(queries["query_embedding"])
    profile_embeddings = np.stack([profile["profile_embedding"] for profile in tagged_profiles])    
    cosine_similarities = cosine_similarity(query_embeddings, profile_embeddings)    
    return [len(profile_embeddings)] * len(query_embeddings), (cosine_similarities >= min_cosine_similarity).astype(int)


def cohort_search(queries, tagged_cohorts, query_radius, cohort_radius, close_overlap):
    cohort_ids = [cohort["cohort_id"] for cohort in tagged_cohorts]
    cohort_embeddings = np.stack([cohort["cohort_embedding"] for cohort in tagged_cohorts]) if tagged_cohorts else np.array([])
    query_ids = queries["query_id"]
    query_embeddings = np.array(queries["query_embedding"])
        
    closely_matched_query_indices = []
    loosely_matched_query_indices = []
    query_to_cohort_distances = []
    
    if cohort_embeddings.size > 0:
        cohort_knn = NearestNeighbors(algorithm='brute', metric="euclidean", n_jobs=-1)
        cohort_knn.fit(cohort_embeddings)
        
        for query_embedding in query_embeddings:
            # Find cohorts within the search radius
            distances, cohort_indices = cohort_knn.radius_neighbors(
                [query_embedding],
                radius=length_to_distance(distance_to_length(query_radius) + distance_to_length(cohort_radius)),
                return_distance=True
            )
            cohort_indices = cohort_indices[0]
            distances = distances[0]
            
            # Partition the cohorts into closely matched and loosely matched based on a threshold
            close_threshold = solve_for_d_by_coverage(cohort_radius, query_radius, close_overlap)  # Adjust the close_overlap threshold based profile distributions
            
            closely_matched = []
            loosely_matched = []
            cohort_distances = {}

            for i, distance in zip(cohort_indices, distances):
                cohort_embedding = cohort_embeddings[i]
                cohort_distances[cohort_ids[i]] = distance
                
                if distance_to_length(distance) <= close_threshold:
                    closely_matched.append(cohort_ids[i])
                else:
                    loosely_matched.append(cohort_ids[i])
            
            closely_matched_query_indices.append(closely_matched)
            loosely_matched_query_indices.append(loosely_matched)
            query_to_cohort_distances.append(cohort_distances)
    else:
        closely_matched_query_indices = [[] for _ in range(len(query_embeddings))]
        loosely_matched_query_indices = [[] for _ in range(len(query_embeddings))]
        query_to_cohort_distances = [{} for _ in range(len(query_embeddings))]

    print(f"There are {[len(closely_matched) for closely_matched in closely_matched_query_indices]} cohorts closely matched to each query, respectively.")
    print(f"There are {[len(loosely_matched) for loosely_matched in loosely_matched_query_indices]} cohorts loosely matched to each query, respectively.")

    return cohort_ids, closely_matched_query_indices, loosely_matched_query_indices




def collect_profiles_to_query(tagged_profiles, cohort_ids, closely_matched_cohort_indices, loosely_matched_cohort_indices, query_count):
    profiles_to_query = {query_idx: {"profile_id": [], "profile_embedding": []} for query_idx in range(query_count)}
    profiles_to_include = {query_idx: {"profile_id": [], "profile_embedding": []} for query_idx in range(query_count)}

    cohort_to_profiles = defaultdict(list)
    for profile in tagged_profiles:
        for cohort_id in profile["cohort_ids"]:
            cohort_to_profiles[cohort_id].append((profile["profile_id"], profile["profile_embedding"]))

    for query_idx in range(query_count):
        closely_matched_profiles = OrderedDict()
        loosely_matched_profiles = OrderedDict()
      
        for cohort_id in closely_matched_cohort_indices[query_idx]:
            if cohort_id in cohort_to_profiles:
                for profile_id, embedding in cohort_to_profiles[cohort_id]:
                    if profile_id not in closely_matched_profiles:
                        closely_matched_profiles[profile_id] = embedding
                      
        profiles_to_include[query_idx]["profile_id"].extend(closely_matched_profiles.keys())
        profiles_to_include[query_idx]["profile_embedding"].extend(closely_matched_profiles.values())
      
        for cohort_id in loosely_matched_cohort_indices[query_idx]:
            if cohort_id in cohort_to_profiles:
                for profile_id, embedding in cohort_to_profiles[cohort_id]:
                    if profile_id not in closely_matched_profiles and profile_id not in loosely_matched_profiles:
                        loosely_matched_profiles[profile_id] = embedding

        profiles_to_query[query_idx]["profile_id"].extend(loosely_matched_profiles.keys())
        profiles_to_query[query_idx]["profile_embedding"].extend(loosely_matched_profiles.values())

    return profiles_to_include, profiles_to_query


def profile_search(queries, tagged_profiles, cohort_ids, closely_matched_cohort_indices, loosely_matched_cohort_indices, query_radius):
    query_ids = queries["query_id"]
    query_embeddings = np.array(queries["query_embedding"])
    
    print(f"Querying profiles...")

    # Collect profiles to query
    profiles_to_include, profiles_to_query = collect_profiles_to_query(tagged_profiles, cohort_ids, closely_matched_cohort_indices, loosely_matched_cohort_indices, len(query_ids))
    
    # Count of profiles in each cohort per query
    cohort_calculation_counts = [len(cohort_ids) + len(profiles_to_query[q_idx]["profile_id"]) for q_idx in range(len(query_ids))]
    print(f"Cohort calculation counts per query: {cohort_calculation_counts}")
    
    # Initialize the list to hold profile ids in range for each query
    in_range_profile_ids = [profiles_to_include[q_idx]["profile_id"][:] for q_idx in range(len(query_ids))]
    
    # Query processing: find profiles in range for each query
    for query_idx in range(len(query_ids)):
        profiles = profiles_to_query[query_idx]
        if profiles["profile_embedding"]:
            embeddings = np.vstack(profiles["profile_embedding"])
        else:
            embeddings = np.empty((0, query_embeddings.shape[1]))

        if embeddings.size > 0:
            profile_knn = NearestNeighbors(algorithm="brute", metric="euclidean", n_jobs=-1)
            profile_knn.fit(embeddings)
            all_inner_profile_indices = profile_knn.radius_neighbors([query_embeddings[query_idx]], radius=query_radius, return_distance=False)[0]
            in_range_profile_ids[query_idx].extend([profiles["profile_id"][i] for i in all_inner_profile_indices])

    
    # Initialize the result array (query x profiles)
    cohort_query_results = np.zeros((len(query_ids), len(tagged_profiles)), dtype=int)
    profile_id_array = np.array([p["profile_id"] for p in tagged_profiles])  # Convert once

    for query_idx in range(len(query_ids)):
        matched_profile_ids = np.array(in_range_profile_ids[query_idx])  # Convert to NumPy array
        mask = np.isin(profile_id_array, matched_profile_ids)  # Vectorized lookup
        cohort_query_results[query_idx, mask] = True
    cohort_query_results = cohort_query_results.astype(int)

    
    return cohort_calculation_counts, cohort_query_results