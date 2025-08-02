import json
import csv
from sklearn.metrics import precision_score, recall_score
import math
import numpy as np


def get_settings(path: str):
    with open(path, "r") as f:
        settings = json.load(f)
    return settings


def similarity_to_distance(similarity: float) -> float:
    return (2 - (2 * similarity)) ** .5

def distance_to_similarity(distance: float) -> float:
    return (2 - distance ** 2) / 2

def score_to_similarity(score: float) -> float:
    return score / 50.0 - 1


def score_to_distance(score: float) -> float:
    return similarity_to_distance(score_to_similarity(score))

def score_to_cosine_distance(score: float) -> float:
    return (1-score_to_similarity(score))

def cosine_distance_to_length(cosine_distance: float) -> float:
    return math.acos(1-cosine_distance)

def length_to_cosine_distance(length: float) -> float:
    return (1-math.cos(length))
    
def score_to_length(score: float) -> float:
    return similarity_to_length(score_to_similarity(score))

def similarity_to_length(similarity: float) -> float:
    return math.acos(similarity)

def length_to_similarity(length: float) -> float:
    return math.cos(length)

def length_to_distance(length: float) -> float:
    return similarity_to_distance(length_to_similarity(length))

def distance_to_length(distance: float) -> float:
    return similarity_to_length(distance_to_similarity(distance))

def score_to_radius(score: float) -> float:
    return similarity_to_radius(score_to_similarity(score))

def similarity_to_radius(similarity: float) -> float:
    return ((1 - similarity**2)**(1/2))


def read_objects_from_file(file_path: str):
    for line in open(file_path, 'r'):
        try:
            obj = json.loads(line.strip())
            yield obj
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")  


# def write_objects_to_file(file_path: str, objects):
#     with open(file_path, 'w') as f:
#         for obj in objects:
#             f.write(json.dumps(obj) + "\n")

def write_objects_to_file(file_path: str, objects):
    with open(file_path, 'w') as f:
        for obj in objects:
            serializable_obj = {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in obj.items()
            }
            f.write(json.dumps(serializable_obj) + "\n")
            
def write_objects_to_csv(file_path: str, objects):
    if not objects:
        return  # Avoid writing an empty file if the list is empty
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=objects[0].keys())
        writer.writeheader()
        writer.writerows(objects)
            

def score_results(y_trues: list[list[int]], y_preds: list[list[int]]):
    y_trues_counts = [len(y_trues[query_idx]) for query_idx in range(len(y_trues))]
    y_preds_counts = [len(y_preds[query_idx]) for query_idx in range(len(y_preds))]
    print(f"Counts of y_trues by query: {y_trues_counts}.")
    print(f"Counts of y_preds by query: {y_preds_counts}.")
    precisions = [precision_score(y_trues[query_idx], y_preds[query_idx]) for query_idx in range(len(y_trues))]
    recalls = [recall_score(y_trues[query_idx], y_preds[query_idx]) for query_idx in range(len(y_trues))]
    return precisions, recalls

def convert_profile_id(profile_id):
    """Ensure profile ID is always a string."""
    if isinstance(profile_id, list) and len(profile_id) == 1:
        return str(profile_id[0])  # Convert ["string"] to "string"
    return str(profile_id)  # Convert any other type to string