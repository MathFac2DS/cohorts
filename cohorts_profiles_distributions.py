# This script produces distributions for cohorts and profiles after cohort assignment
# The inputs are lists of objects of the form cohorts: {"cohort_id": str, "cohort_embedding": list[float], "cohort_type": str}, and tagged_profiles: {"profile_id": str, "profile_embedding": list[float], "cohort_embeddings": list[list[float]], "cohort_ids": list[str]}
# The outputs are figures for the distribution of profile coutns, and the distribution of cohort counts

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from cohort_utils import write_objects_to_csv, get_settings


def cohorts_and_profiles_distributions(tagged_profiles, tagged_cohorts, settings_path):
    configs = get_settings(settings_path)
    cohort_profile_match_count = defaultdict(int) # Dictionary to track the number of profiles assigned to each cohort
    profile_cohort_list = [] # List to store profile-cohort count mappings

    # Iterate through each profile and count how many cohorts it is assigned to
    for profile in tagged_profiles:
        cohort_ids = profile["cohort_ids"]

        # Append profile's cohort count to the list
        profile_cohort_list.append({
            "profile_id": profile["profile_id"],
            "cohort_count": len(cohort_ids)
        })

        # Update cohort profile match counts
        for cohort_id in cohort_ids:
            cohort_profile_match_count[cohort_id] += 1

    # Convert cohort profile match counts into a list of dictionaries
    cohort_profile_list = [
        {"cohort_id": cohort["cohort_id"], "profile_count": cohort_profile_match_count.get(cohort["cohort_id"], 0)}
        for cohort in tagged_cohorts  # Extract "cohort_id" from each dictionary
    ]
    
    
    write_objects_to_csv(configs["profile_counts_file_path"], cohort_profile_list)
    write_objects_to_csv(configs["cohort_counts_file_path"], profile_cohort_list)
    
    return
   
    
def plot_distance_distribution(cohort_distance_distributions):
    cohort_distance_distributions = cohort_distance_distributions.to_dict(orient="list")
    all_distances = [distance for distances in cohort_distance_distributions.values() for distance in distances]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_distances, bins=50, range=(0, 1), alpha=0.75, color='b', edgecolor='black')
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of Profile-Cohort Distances")
    ax.grid(True)

    return fig

    
