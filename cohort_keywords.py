class TaxonomyKeywordsPaths:
    """
    A class to store and keywords by client.
    """

    def __init__(self):
        # Dictionary to store client-to-key mappings
        self.client_path = {
            "aipp": "/home/jupyter/cohort/data/aipp/aipp_salient_keyword_embeddings.jsonl",
            "annex": "/home/jupyter/cohort/data/annex/annex_salient_keyword_embeddings.jsonl",
            # "oup": "cuebert_embedding_all-distilroberta-v1",
            "tandf": "/home/jupyter/cohort/data/tandf/tandf_salient_keyword_embeddings.jsonl"
        }


    def get_path_to_keywords(self, client_name: str) -> str:
        """
        Retrieve the taxonomy keywords for a given client name.

        Args:
            client_name (str): The name of the client.

        Returns:
            str: The path to taxonomy keywords associated with the client.

        Raises:
            ValueError: If the client name is not found in the dictionary.
        """
        if client_name in self.client_path:
            return self.client_path[client_name]
        else:
            raise ValueError(f"Client name '{client_name}' not found in thekeyword paths.")
