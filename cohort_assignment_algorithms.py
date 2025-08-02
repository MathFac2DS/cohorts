import math
import numpy as np
import pandas as pd
from cohort_utils import distance_to_length, length_to_distance, distance_to_similarity
from sklearn.neighbors import NearestNeighbors


def multiple_cohort_assignment(cohorts, profiles, cohort_radius):
    cohort_ids = np.array(cohorts["cohort_id"]) 
    cohort_embeddings = np.array(cohorts["cohort_embedding"])
    profile_ids = np.array(profiles["profile_id"]) 
    profile_embeddings = np.array(profiles["profile_embedding"])
    print(f"Assigning {len(profile_ids)} profiles to {len(cohort_ids)} cohorts...")

    knn = NearestNeighbors(algorithm='brute', metric="euclidean", n_jobs=-1)
    knn.fit(cohort_embeddings)
    cohort_distances, cohort_indices = knn.kneighbors(profile_embeddings, n_neighbors=20, return_distance=True)
    
    cohort_profile_ids = {cohort_id: [] for cohort_id in cohort_ids}
    cohort_profile_distances = {cohort_id: [] for cohort_id in cohort_ids}
    profile_cohort_ids = []
    
    for profile_idx, (distances, indices) in enumerate(zip(cohort_distances, cohort_indices)):
        assigned_cohort_ids = []
        for distance, cohort_idx in zip(distances, indices):
            if distance <= cohort_radius:
                cohort_id = cohort_ids[cohort_idx]
                assigned_cohort_ids.append(cohort_id)
                cohort_profile_ids[cohort_id].append(profile_ids[profile_idx])
                cohort_profile_distances[cohort_id].append(distance)
        profile_cohort_ids.append(assigned_cohort_ids)

    total_profiles = len(profile_ids)
    min_profile_count = total_profiles * 0.001  # 0.1% threshold
    
    tagged_profiles = []
    for pid, pembed, cohort_list in zip(profile_ids, profile_embeddings, profile_cohort_ids):
        filtered_cohort_list = [cid for cid in cohort_list if len(cohort_profile_ids[cid]) >= min_profile_count]
        tagged_profiles.append({"profile_id": pid, "profile_embedding": pembed, "cohort_ids": filtered_cohort_list})
    
    tagged_cohorts = []
    for cid, cembed in zip(cohort_ids, cohort_embeddings):
        profile_count = len(cohort_profile_ids[cid])
        if profile_count < min_profile_count:
            continue  # Skip cohorts with profile count below threshold

        distances = cohort_profile_distances[cid]
        avg_distance = np.mean(distances) if profile_count > 0 else float('nan')
        outer_tolerance = length_to_distance(distance_to_length(cohort_radius) - distance_to_length(avg_distance))
        min_distance = np.min(distances) if profile_count > 0 else float('nan')
        inner_tolerance = length_to_distance(distance_to_length(cohort_radius) - distance_to_length(min_distance))

        tagged_cohorts.append({
            "cohort_id": cid,
            "cohort_embedding": cembed,
            "profile_ids": cohort_profile_ids[cid],
            "profile_count": profile_count,
            "average_distance": avg_distance,
            "outer_tolerance": outer_tolerance,
            "minimum_distance": min_distance,
            "inner_tolerance": inner_tolerance
        })

    return tagged_profiles, tagged_cohorts
