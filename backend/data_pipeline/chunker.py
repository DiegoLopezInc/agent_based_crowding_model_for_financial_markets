"""Text chunking utilities for document processing"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.utils.config import get_config
from backend.utils.logger import setup_logger

logger = setup_logger("chunker")


class TextChunker:
    """Chunk text documents for embedding"""

    def __init__(self):
        self.config = get_config().data_pipeline
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_text(self, text: str) -> List[str]:
        """Chunk a single text

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        if not text or len(text) < self.config.min_chunk_size:
            return [text] if text else []

        chunks = self.splitter.split_text(text)

        # Filter out chunks that are too small
        chunks = [
            chunk for chunk in chunks
            if len(chunk) >= self.config.min_chunk_size
        ]

        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def chunk_document(
        self,
        document: Dict[str, Any],
        preserve_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Chunk a document and preserve metadata

        Args:
            document: Document dictionary with 'content' and 'metadata' keys
            preserve_metadata: Whether to preserve metadata in chunks

        Returns:
            List of chunked documents with metadata
        """
        content = document.get("content", "")
        chunks = self.chunk_text(content)

        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "content": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            if preserve_metadata:
                # Copy all original fields except content
                for key, value in document.items():
                    if key != "content":
                        chunk_doc[key] = value

                # Update metadata with chunk info
                if "metadata" not in chunk_doc:
                    chunk_doc["metadata"] = {}

                chunk_doc["metadata"].update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "original_length": len(content),
                })

            chunked_docs.append(chunk_doc)

        logger.debug(f"Created {len(chunked_docs)} chunked documents")
        return chunked_docs

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Chunk multiple documents

        Args:
            documents: List of documents

        Returns:
            List of all chunked documents
        """
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks"
        )
        return all_chunks


def get_chunker() -> TextChunker:
    """Get text chunker instance"""
    return TextChunker()
