"""
Semantic search implementation using ChromaDB for Napistu MCP server.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic search engine using ChromaDB and sentence transformers.

    Provides AI-powered search capabilities for text content using vector embeddings.
    Manages multiple collections for different content types and handles persistent
    storage of embeddings.

    Parameters
    ----------
    persist_directory : str, optional
        Directory path for persistent storage of ChromaDB data.
        Default is "./chroma_db". The directory will be created if it doesn't exist.

    Attributes
    ----------
    client : chromadb.PersistentClient
        ChromaDB client instance for managing collections and persistence.
    embedding_function : chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
        Embedding function using the 'all-MiniLM-L6-v2' model for text vectorization.
    collections : Dict[str, chromadb.Collection]
        Dictionary mapping collection names to ChromaDB collection objects.

    Examples
    --------
    Basic usage for semantic search:

    >>> search = SemanticSearch()
    >>> collection = search.get_or_create_collection("documents")
    >>> content = {"readme": {"install": "pip install package"}}
    >>> search.index_content("documents", content)
    >>> results = search.search("how to install", "documents")
    >>> print(f"Found {len(results)} results")

    Notes
    -----
    The class uses the 'all-MiniLM-L6-v2' sentence transformer model which:
    - Produces 384-dimensional embeddings
    - Is optimized for semantic similarity tasks
    - Has a good balance of speed and quality
    - Downloads automatically on first use (~90MB)

    ChromaDB stores embeddings persistently, so collections survive across
    sessions. The first indexing operation may be slower due to model download
    and embedding computation.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize SemanticSearch with persistent ChromaDB storage.

        Parameters
        ----------
        persist_directory : str, optional
            Path to directory for storing ChromaDB data. Created if doesn't exist.
            Default is "./chroma_db".

        Examples
        --------
        >>> search = SemanticSearch()  # Uses default ./chroma_db
        >>> search = SemanticSearch("/custom/path/db")  # Custom path
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        self.collections = {}

    def get_or_create_collection(self, name: str):
        """
        Get existing collection or create a new one with consistent configuration.

        Parameters
        ----------
        name : str
            Name of the collection. Will be prefixed with "napistu_" for namespacing.

        Returns
        -------
        chromadb.Collection
            ChromaDB collection object for storing and querying embeddings.

        Examples
        --------
        >>> search = SemanticSearch()
        >>> collection = search.get_or_create_collection("documentation")
        >>> print(collection.name)  # "napistu_documentation"

        Notes
        -----
        Collections are cached in self.collections for efficient reuse.
        If a collection already exists, it will be loaded with the same
        embedding function to ensure consistency.
        """
        collection_name = f"napistu_{name}"

        try:
            collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
        except (chromadb.errors.CollectionNotFoundError, ValueError):
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

        self.collections[name] = collection
        return collection

    def index_content(self, collection_name: str, content_dict: Dict[str, Any]):
        """
        Index content into a collection for semantic search.

        Processes nested content dictionaries and creates searchable embeddings.
        Handles different content types appropriately (issues/PRs vs regular content).
        Existing content in the collection is cleared before adding new content.

        Parameters
        ----------
        collection_name : str
            Name of the collection to index content into.
        content_dict : Dict[str, Any]
            Nested dictionary containing content to index. Structure should be:
            {content_type: {name: content_text, ...}, ...}

            Special handling for content types:
            - 'issues', 'prs': Expected to contain lists of dictionaries with 'title', 'body', 'number'
            - Other types: Expected to contain direct text content

        Examples
        --------
        Index documentation content with mixed types:

        >>> content = {
        ...     "readme": {
        ...         "install": "Installation instructions...",
        ...         "usage": "Usage examples..."
        ...     },
        ...     "issues": {
        ...         "repo1": [
        ...             {"title": "Bug report", "body": "Description...", "number": 123}
        ...         ]
        ...     }
        ... }
        >>> search.index_content("documentation", content)

        Notes
        -----
        - Content shorter than 50 characters is filtered out for regular content
        - Content shorter than 20 characters is filtered out for issues/PRs
        - Issues and PRs combine title and body for better searchability
        - Each piece of content gets metadata including type, name, and source
        - IDs are generated for uniqueness and later retrieval
        - The collection is cleared before indexing to ensure consistency

        Raises
        ------
        Exception
            If ChromaDB indexing fails (e.g., invalid content format)
        """
        collection = self.get_or_create_collection(collection_name)

        documents = []
        metadatas = []
        ids = []

        for content_type, items in content_dict.items():
            if content_type in ["issues", "prs"]:
                # Handle issues and PRs specially - they're lists of dictionaries
                for repo_name, item_list in items.items():
                    if isinstance(item_list, list):
                        for item in item_list:
                            if isinstance(item, dict) and item.get("title"):
                                # Combine title and body for better search
                                title = item.get("title", "")
                                body = item.get("body", "")
                                content = f"{title}\n\n{body}" if body else title

                                if len(content.strip()) > 20:  # Skip very short content
                                    documents.append(content)
                                    metadatas.append(
                                        {
                                            "type": content_type,
                                            "name": f"{repo_name}#{item.get('number', '')}",
                                            "source": f"{content_type}: {repo_name}#{item.get('number', '')}",
                                        }
                                    )
                                    ids.append(
                                        f"{content_type}_{repo_name}_{item.get('number', '')}"
                                    )

            elif isinstance(items, dict):
                # Handle regular content (readme, wiki, etc.)
                for name, content in items.items():
                    if content and len(str(content).strip()) > 50:
                        documents.append(str(content))
                        metadatas.append(
                            {
                                "type": content_type,
                                "name": name,
                                "source": f"{content_type}: {name}",
                            }
                        )
                        ids.append(f"{content_type}_{name}")

        if documents:
            # Clear existing content and reindex
            try:
                collection.delete(where={})
            except (chromadb.errors.NoIndexException, ValueError, RuntimeError):
                # Collection might be empty or not properly initialized
                pass

            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} items in {collection_name}")

            # Log breakdown by content type for debugging
            type_counts = {}
            for metadata in metadatas:
                content_type = metadata["type"]
                type_counts[content_type] = type_counts.get(content_type, 0) + 1

            logger.debug(f"Content type breakdown: {type_counts}")
        else:
            logger.warning(f"No content to index in {collection_name}")

    def search(
        self, query: str, collection_name: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on a collection with similarity scores.

        Uses AI embeddings to find content semantically similar to the query,
        even if exact keywords don't match. Returns results with similarity scores.

        Parameters
        ----------
        query : str
            Natural language search query. Can be keywords, phrases, or questions.
        collection_name : str
            Name of the collection to search in.
        n_results : int, optional
            Maximum number of results to return. Default is 5.

        Returns
        -------
        List[Dict[str, Any]]
            List of search results ordered by similarity (highest first), each containing:
            - 'content': The matched text content
            - 'metadata': Dictionary with type, name, source information
            - 'source': Human-readable source description
            - 'similarity_score': Float between 0 and 1 (1 = perfect match, 0 = no similarity)

        Examples
        --------
        Basic semantic search with scores:

        >>> results = search.search("how to install", "documentation")
        >>> for result in results:
        ...     score = result['similarity_score']
        ...     print(f"Score: {score:.3f} - {result['source']}")

        Notes
        -----
        Similarity scores help you understand result quality:
        - 0.8-1.0: Very relevant matches
        - 0.6-0.8: Good matches
        - 0.4-0.6: Moderate relevance
        - 0.0-0.4: Low relevance (may not be useful)

        ChromaDB uses cosine similarity between embeddings, where:
        - 1.0 = identical semantic meaning
        - 0.0 = completely unrelated content
        """
        if collection_name not in self.collections:
            return []

        collection = self.collections[collection_name]

        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],  # Include distances
        )

        formatted_results = []
        for i in range(len(results["documents"][0])):
            # Convert distance to similarity score
            # ChromaDB returns distances, but we want similarity (higher = better)
            distance = results["distances"][0][i] if "distances" in results else 0.0

            # For cosine distance, similarity = 1 - distance
            similarity_score = 1.0 - distance

            formatted_results.append(
                {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "source": results["metadatas"][0][i].get("source", "Unknown"),
                    "similarity_score": similarity_score,
                }
            )

        return formatted_results
