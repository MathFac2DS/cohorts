from cohort_utils import distance_to_length, length_to_distance, distance_to_similarity

def calculate_tolerances(tagged_cohorts, cohort_stats, cohort_radius):
    print(f"Calculating cohort search tolerance for each cohort...")
    cohort_ids = [cohort["cohort_id"] for cohort in tagged_cohorts]
    cohort_avg_distances = {row["cohort_id"]: row["average_distance"] for _, row in cohort_stats.iterrows()}

    for cohort in tagged_cohorts:
        cohort_id = cohort["cohort_id"]
        avg_distance = cohort_avg_distances.get(cohort_id, 0)
        tolerance = length_to_distance(distance_to_length(cohort_radius) - distance_to_length(avg_distance))
        cohort["tolerance"] = tolerance

    return tagged_cohorts
