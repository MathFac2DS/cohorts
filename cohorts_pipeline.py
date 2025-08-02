import sys
import json
from load_embeddings import LoadEmbeddings
from cohort_utils import get_settings, score_to_length, score_to_similarity, score_to_distance, score_to_cosine_distance, read_objects_from_file, write_objects_to_file, write_objects_to_csv, score_results
from cohort_assignment_algorithms import multiple_cohort_assignment
from cohort_querying_algorithms import brute_force_search, cohort_search, profile_search
from cohorts_profiles_distributions import cohorts_and_profiles_distributions, plot_distance_distribution
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt


def main(settings_path):
    configs = get_settings(settings_path)
    client = configs["client"]
    version = configs["version"]
    embedding_key = configs["embedding_key"]   

    print("Loading embeddings...")
    cohorts_loader = LoadEmbeddings(configs["keyword_embeddings_file_path"], configs["embedding_key"])
    cohorts = cohorts_loader.load_cohorts()

    profiles_loader = LoadEmbeddings(configs["profile_embeddings_file_path"], configs["embedding_key"])
    profiles = profiles_loader.load_profiles()

    queries_loader = LoadEmbeddings(configs["query_embeddings_file_path"], "embedding")
    queries = queries_loader.load_queries()
    
    # COHORT ASSIGNMENT SECTION
    
#####     #####     #####     #####     ##### ASSIGNMENT ALGORITHM 
    
    print("Assigning profiles to cohorts...")
    tagged_profiles, tagged_cohorts = multiple_cohort_assignment(cohorts, profiles, score_to_distance(configs["min_cohort_score"]))
    
#####     #####     #####     #####     ##### ASSIGNMENT ALGORITHM

    print("Analyzing profile and cohort distributions...")
    cohorts_and_profiles_distributions(tagged_profiles, tagged_cohorts, settings_path)

#####     #####     #####     #####     ##### QUERYING ALGORITHM
    
    print("Performing audience search queries...")
    brute_force_calculation_counts, brute_force_query_results = brute_force_search(queries, tagged_profiles, score_to_similarity(configs["min_query_score"]))
    
    cohort_ids, closely_matched_cohort_indices, loosely_matched_cohort_indices = cohort_search(queries, tagged_cohorts,
                                                   score_to_distance(configs["min_query_score"]),
                                                   score_to_distance(configs["min_cohort_score"]),
                                                   configs["close_overlap"]
                                                  )

    cohort_calculation_counts, cohort_query_results = profile_search(queries, tagged_profiles,
                                                                     cohort_ids, closely_matched_cohort_indices, loosely_matched_cohort_indices,
                                                                     score_to_distance(configs["min_query_score"])
                                                                    )

#####     #####     #####     #####     ##### QUERYING ALGORITHM


    print("Scoring cohort results...")
    precisions, recalls = score_results(brute_force_query_results, cohort_query_results)
    query_ids = queries["query_id"]
    
    final_results = [{"query_id": query_ids[query_idx],
                      "calculation_count": cohort_calculation_counts[query_idx],
                      "precision": precisions[query_idx],
                      "recall": recalls[query_idx]}
                     for query_idx in range(len(query_ids))]
    
    print("Saving cohort results...")
    write_objects_to_csv(configs["results_file_path"], final_results)
    print(f"Final results saved to the file path {configs['results_file_path']}.")


if __name__ == "__main__":
    main(sys.argv[1])