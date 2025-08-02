import sys
from cohort_utils import get_settings
import json
import numpy as np
from sklearn.preprocessing import normalize

class LoadEmbeddings:
    def __init__(self, file_path, embedding_key):
        self.file_path = file_path
        self.embedding_key = embedding_key

    def _read_objects_from_file(self):
        """Generator to yield JSON objects from a file line-by-line."""
        with open(self.file_path, 'r') as file:
            for line in file:
                try:
                    yield json.loads(line.strip())  # Load each line as a dictionary
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

    def load_cohorts(self):
        """Load cohort embeddings into a structured dictionary."""
        cohorts = {
            "cohort_id": [],
            "cohort_embedding": []
        }

        for obj in self._read_objects_from_file():
            cohorts["cohort_id"].append(obj.get("id"))
            embedding = obj.get(self.embedding_key, [])
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            cohorts["cohort_embedding"].append(embedding)

        if cohorts["cohort_embedding"]:
            cohorts["cohort_embedding"] = normalize(np.array(cohorts["cohort_embedding"]), axis=1).tolist()

        print(f"Loaded {len(cohorts['cohort_id'])} cohorts.")

        return cohorts
    
    def load_profiles(self):
        """Load profile embeddings into a structured dictionary."""
        profiles = {
            "profile_id": [],
            "profile_embedding": []
        }

        for obj in self._read_objects_from_file():
            profiles["profile_id"].append(obj.get("id"))
            profiles["profile_embedding"].append(obj.get(self.embedding_key, []))

        profiles["profile_embedding"] = normalize(profiles["profile_embedding"], axis=1)
        
        print(f"Loaded {len(profiles['profile_id'])} profiles.") 
        
        return profiles
    
    def load_queries(self):
        """Load query embeddings into a structured dictionary."""
        queries = {
            "query_id": [],
            "query_embedding": []
        }

        for obj in self._read_objects_from_file():
            queries["query_id"].append(obj.get("id"))
            queries["query_embedding"].append(obj.get("embedding", []))

        queries["query_embedding"] = np.array(normalize(queries["query_embedding"], axis=1))
        
        print(f"Loaded {len(queries['query_id'])} queries.") 
        
        return queries