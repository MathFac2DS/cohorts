To run the cohort assignment and search process, complete the following.

0. Process the required embeddings (keyword_embeddings, profile_embeddings, query_embeddings). 

0a. Filter keyword embeddings to include those with salience > 5.

0b. Reference for the project can be found in Drive in the Contractor Projects folder


1. In the root directory, edit the file "create_cohort_config.ipynb" by updating the parameters listed at the top (client, min_cohort_score, min_query_score, close_overlap, version). File paths will need adjusted when creating cohorts for new clients.

2. Assign and search cohorts by running "cohorts_pipeline.py" from Terminal.

2a. Load the embeddings: The LoadEmbeddings class (load_embeddings.py) creates structured dictionaries for profiles, cohorts, and queries. 

2b. Assign profiles to cohorts: The multiple_cohort_assignment (cohort_assignment_algorithms.py) assigns profiles to the 20 closest cohorts, drops any cohorts with fewer than 0.01% of the total profiles, and returns 
	tagged_profiles: [{"profile_id", "profile_embedding", "cohort_ids"} for each profile
	tagged_cohorts: [{"cohort_id", "cohort_embedding", "profile_ids", "profile_count", "average_distance", "outer_tolerance"}] for each cohort

2c. Record profile and cohort distributions: The function cohorts_and_profiles_distributions (cohorts_profiles_distributions.py) generates and stores csvs of profile counts per cohort (configs["profile_counts_file_path"]) and cohort counts per profile (configs["cohort_counts_file_path"]). These artifacts can be used a determine if the cohorts adequately cover the profiles, and if the profiles are adequately represented by the assigned cohort.

2d. Match cohorts: The cohort_search function (cohort_querying_algorithms.py) identifies cohorts in range and partitions into closely matched, loosely matched:
	cohorts in range (cohort_indices): distance < r + R
	closely matched cohorts (closely_matched_query_indices): distance < close_threshold
	loosely matched cohorts (loosely_matched_query_indices): distance > close_threshold

2e. Inner profile search: The profile_search function (cohort_querying_algorithms.py) returns all profiles in closely matched cohorts and queries profiles in loosely matched cohorts

2f. Score results: The score_results function compares brute force search to cohort search. Based on results, consider tuning parameters: min_cohort_score, min_query_score, close_overlap 
