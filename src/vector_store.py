from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS


class VectorStoreManager:
    """
    Manage the FAISS vector store.

    Responsibilities:
    - Create a FAISS vector store.
    - Save and load the vector store.
    - Create a retriever for similarity search.
    """

    def __init__(self, embedding_model):
        """
        Initialize the vector store manager.

        Args:
            embedding_model: LangChain embedding model.
        """

        self.embedding_model = embedding_model
        self.vector_store = None

    def create_vector_store(self,documents: list[Document],) -> FAISS:
        """
        Create a FAISS vector store from documents.

        Args:
            documents: Chunked LangChain documents.

        Returns:
            FAISS: Created vector store.
        """

        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        return self.vector_store

    def save(self,save_path: str | Path) -> None:
        """
        Save the FAISS index to disk.

        Args:
            save_path: Directory where the index will be saved.
        """

        if self.vector_store is None:
            raise ValueError("Vector store has not been created.")

        self.vector_store.save_local(str(save_path))

    def load(self,load_path: str | Path) -> FAISS:
        """
        Load a FAISS index from disk.

        Args:
            load_path: Directory containing the saved FAISS index.

        Returns:
            FAISS: Loaded vector store.
        """

        self.vector_store = FAISS.load_local(
            str(load_path),
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )

        return self.vector_store

    def as_retriever(
        self,
        k: int = 5
    ) -> VectorStoreRetriever:
        """
        Create a retriever from the vector store.

        Args:
            k: Number of documents to retrieve.

        Returns:
            VectorStoreRetriever
        """

        if self.vector_store is None:
            raise ValueError("Vector store has not been created.")

        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )