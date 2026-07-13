from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List



class TextSplitter:
    """
        Splits LangChain Documents into smaller overlapping chunks.

        Responsibilities:
        - Configure a RecursiveCharacterTextSplitter.
        - Split documents into chunks suitable for embedding.

        This class does NOT perform embedding or vector storage.
    """
    def __init__(self,chunk_size = 1000,chunk_overlap =200):
        """
        Initialize the text splitter.

        Args:
            chunk_size: Maximum size of each chunk.
            chunk_overlap: Number of overlapping characters
                between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter  = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            length_function = len,
            separators = ['\n\n','\n',' ',''])

    def split_documents(self,documents:list[Document]) -> list[Document]:

        """
        Split LangChain documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            list[Document]: Chunked LangChain documents.
        """

        
        return self.text_splitter.split_documents(documents)