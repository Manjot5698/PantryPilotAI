from typing import List

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


class EmbeddingManager(Embeddings):
    """
    LangChain-compatible embedding model using SentenceTransformer.

    Responsibilities:
    - Load the embedding model.
    - Embed documents.
    - Embed user queries.
    """

    def __init__(self,model_name: str = "all-MiniLM-L6-v2",):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self,texts: List[str],) -> List[List[float]]:
        """
        Embed multiple documents.

        Args:
            texts: List of document texts.

        Returns:
            List of embedding vectors.
        """

        embeddings=  self.model.encode(texts,convert_to_numpy=True,show_progress_bar=True,normalize_embeddings=True,)
        return embeddings.tolist()
    
    def embed_query(self,text: str,) -> List[float]:
        """
        Embed a user query.

        Args:
            text: Query string.

        Returns:
            Query embedding.
        """

        query_vector =self.model.encode(text,convert_to_numpy=True,normalize_embeddings=True,)
        return query_vector.tolist()