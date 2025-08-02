class EmbeddingKey:
    """
    A class to store and retrieve embedding keys by client.
    """

    def __init__(self):
        # Dictionary to store client-to-key mappings
        self.client_key = {
            "aipp": "alchemist_embedding_lodestone-v3-client-768",
            "annex": "alchemist_embedding_lodestone-v3-base-768", 
            # "oup": "cuebert_embedding_all-distilroberta-v1",
            "tandf": "alchemist_embedding_lodestone-v3-scholarly-768"
        }


    def get_key(self, client_name: str) -> str:
        """
        Retrieve the embedding key for a given client name.

        Args:
            client_name (str): The name of the client.

        Returns:
            str: The embedding key(s) associated with the client.

        Raises:
            ValueError: If the client name is not found in the dictionary.
        """
        if client_name in self.client_key:
            return self.client_key[client_name]
        else:
            raise ValueError(f"Client name '{client_name}' not found in the key mappings.")
