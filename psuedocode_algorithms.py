

def assign_cohorts(keywords, profiles, radius):
    for profile in profiles:
        nearest_keyword, distance = find_nearest(profile, keywords)
      
        if distance < radius:
            cohort = get_or_create_cohort(nearest_keyword, "keyword_matched")
        else:
            cohort = create_cohort(profile, "self_cohort")
        
        assign_profile_to_cohort(profile, cohort)
    
    return cohorts, tagged_profiles



def query_cohorts(queries, cohorts, profiles, query_radius, cohort_radius):
    for query in queries:
        keyword_cohort_ids = find_cohorts(query, keyword_cohorts, query_radius + cohort_radius)
        self_cohort_ids = find_cohorts(query, self_cohorts, query_radius)
        
        result = [1 if profile_in_cohorts(profile, keyword_cohort_ids, self_cohort_ids) else 0
                  for profile in profiles]

        store_query_result(result)

    return count_cohorts(cohorts), query_results


#v8 pseudo code

def assign_cohorts(keywords, profiles, radius):
    for profile in profiles:
        cohorts, distance = find_nearest_neighbors(profile, keywords)
        if distance < radius:
            assign_profile_to_cohort(profile, cohort)
    tagged_profiles = {profile_id, profile_embedding, cohort_ids}
     
    for cohort in cohorts:
        avg_distance = mean(distance(cohort, profiles))
        tolerance = radius - avg_distance
    
    tagged_cohorts = {cohort_id, cohort_embedding, profile_ids, tolerance}
    
    return tagged_cohorts, tagged_profiles


# with tolerances
def cohort_search(queries, tagged_cohorts, query_radius, cohort_radius):
    avg_distances = map_cohort_avg_distances(tagged_cohorts)
    tolerances = map_cohort_tolerances(tagged_cohorts)

    cohort_knn = build_knn(cohort_embeddings)
    for query in query_embeddings:
        avg_radius = mean(cohort_avg_distances)
        cohort_indices, distances = search_cohorts(cohort_knn, query, query_radius, avg_radius)

        matched_cohorts = {cohort_ids[i]: distances[j] for j, i in enumerate(cohort_indices)
            if (distances[j] < query_radius + cohort_tolerances[i])
        }
        
        all_cohort_indices.append(matched_cohorts)
    
    return cohort_ids, all_cohort_indices


# without tolerances, baseline cohort query
def cohort_search(queries, tagged_cohorts, query_radius, cohort_radius):
    cohort_knn = build_knn(cohort_embeddings)
    
    for query in queries:        
        cohort_indices, distances = search_cohorts(cohort_knn, query, max_radius) 
        matched_cohorts = {cohort_ids[i]: distances[j] 
                           for j, i in enumerate(cohort_indices) 
                           if distances[j] <= threshold
                          }
        closely_matched, loosely_matched = partition_cohorts(matched_cohorts, close_threshold)
    
    return cohort_ids, closely_matched, loosely_matched

def cohort_search(queries, tagged_cohorts, query_radius, cohort_radius):
    cohort_knn = build_knn(cohort_embeddings)

    for query in query_embeddings:
        cohort_indices, distances = search_cohorts(cohort_knn, query, max_radius)

        matched_cohorts = {cohort_ids[i]: distances[j]
                           for j, i in enumerate(cohort_indices)}

        closely_matched, loosely_matched = partition_cohorts(matched_cohorts, close_threshold)

    return cohort_ids, closely_matched, loosely_matched



# inner profile search

def profile_search(queries, tagged_profiles, cohort_ids, closely_matched, loosely_matched, query_radius):
    
    profiles_to_include, profiles_to_query = collect_profiles_to_query(tagged_profiles, cohort_ids, 
                                                                       closely_matched, loosely_matched)

    for query in queries:
        profiles = profiles_to_query[query]
        profile_indices, distances = search_profiles(profiles, query, query_radius)

        in_range_profiles = profiles_to_include[query] + extract_matched_profiles(profiles, profile_indices)
        store_query_results(query, in_range_profiles)

    return cohort_calculation_counts, cohort_query_results

# partition cohorts pseudocode
def partition_cohorts(cohort_radius, query_radius, coverage):
    r = distance_to_length(cohort_radius)
    R = distance_to_length(query_radius)

    A_r = π * r²
    A_part = A_r * coverage
    d_initial = (r + R) / 2 # initial guess

    d_solution = fsolve(intersection_equation, d_initial, args=(r, R, A_part))

    return length_to_distance(d_solution[0])