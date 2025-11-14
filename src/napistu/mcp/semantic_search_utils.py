"""
Utility functions for semantic search content processing and chunking.

These functions are designed to be pure, testable utilities that don't depend
on any specific class state. They can be easily tested, reused, and maintained
independently of the SemanticSearch class.
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from napistu.mcp.constants import (
    CODEBASE_DEFS,
    CODEBASE_RTD_DEFS,
    GITHUB_DEFS,
    SEMANTIC_SEARCH_DEFS,
    SEMANTIC_SEARCH_METADATA_DEFS,
)

logger = logging.getLogger(__name__)


def process_issues_and_prs(
    content_type: str, items: Dict[str, Any]
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Process issues and pull requests for indexing.

    Combines title and body for each issue/PR without chunking since they're
    typically reasonably sized.

    Parameters
    ----------
    content_type : str
        Type of content ('issues' or 'prs')
    items : Dict[str, Any]
        Dictionary mapping repo names to lists of issues/PRs

    Returns
    -------
    Tuple[List[str], List[Dict], List[str]]
        Tuple of (documents, metadatas, ids) ready for ChromaDB indexing
    """
    documents = []
    metadatas = []
    ids = []

    for repo_name, item_list in items.items():
        if isinstance(item_list, list):
            for item in item_list:
                if isinstance(item, dict) and item.get(GITHUB_DEFS.TITLE):
                    title = item.get(GITHUB_DEFS.TITLE, "")
                    body = item.get(GITHUB_DEFS.BODY, "")
                    content = f"{title}\n\n{body}" if body else title

                    if (
                        len(content.strip())
                        > SEMANTIC_SEARCH_DEFS.MIN_CONTENT_LENGTH_SHORT
                    ):
                        documents.append(content)
                        metadatas.append(
                            {
                                SEMANTIC_SEARCH_METADATA_DEFS.TYPE: content_type,
                                SEMANTIC_SEARCH_METADATA_DEFS.NAME: f"{repo_name}#{item.get(GITHUB_DEFS.NUMBER, '')}",
                                SEMANTIC_SEARCH_METADATA_DEFS.SOURCE: f"{content_type}: {repo_name}#{item.get(GITHUB_DEFS.NUMBER, '')}",
                                SEMANTIC_SEARCH_METADATA_DEFS.CHUNK: 0,
                                SEMANTIC_SEARCH_METADATA_DEFS.IS_CHUNKED: False,
                            }
                        )
                        ids.append(
                            f"{content_type}_{repo_name}_{item.get(GITHUB_DEFS.NUMBER, '')}"
                        )

    return documents, metadatas, ids


def process_chunkable_content(
    content_type: str,
    items: Dict[str, Any],
    chunk_threshold: int = SEMANTIC_SEARCH_DEFS.CHUNK_THRESHOLD,
    max_chunk_size: int = SEMANTIC_SEARCH_DEFS.MAX_CHUNK_SIZE,
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Process content that may need chunking (wiki, README, etc.).

    Uses smart chunking that preserves document structure and semantic boundaries.
    Automatically detects markdown headers and splits appropriately.

    Parameters
    ----------
    content_type : str
        Type of content ('wiki', 'readme', etc.)
    items : Dict[str, Any]
        Dictionary mapping content names to content text
    chunk_threshold : int, optional
        Content length threshold for chunking (default 1200 chars)
    max_chunk_size : int, optional
        Maximum size per chunk (default 1000 chars)

    Returns
    -------
    Tuple[List[str], List[Dict], List[str]]
        Tuple of (documents, metadatas, ids) ready for ChromaDB indexing
    """
    documents = []
    metadatas = []
    ids = []

    for name, content in items.items():
        if (
            content
            and len(str(content).strip()) > SEMANTIC_SEARCH_DEFS.MIN_CONTENT_LENGTH_LONG
        ):
            content_str = str(content)

            if len(content_str) > chunk_threshold:
                # Use smart chunking for long content
                chunks = _chunk_content_smart(content_str, name, max_chunk_size)

                for chunk_idx, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append(
                        {
                            SEMANTIC_SEARCH_METADATA_DEFS.TYPE: content_type,
                            SEMANTIC_SEARCH_METADATA_DEFS.NAME: name,
                            SEMANTIC_SEARCH_METADATA_DEFS.SOURCE: f"{content_type}: {name}{SEMANTIC_SEARCH_DEFS.CHUNK_PART_PREFIX}{chunk_idx + 1}{SEMANTIC_SEARCH_DEFS.CHUNK_PART_SUFFIX}",
                            SEMANTIC_SEARCH_METADATA_DEFS.CHUNK: chunk_idx,
                            SEMANTIC_SEARCH_METADATA_DEFS.TOTAL_CHUNKS: len(chunks),
                            SEMANTIC_SEARCH_METADATA_DEFS.IS_CHUNKED: True,
                        }
                    )
                    ids.append(f"{content_type}_{name}_chunk_{chunk_idx}")
            else:
                # Short content, index as-is
                documents.append(content_str)
                metadatas.append(
                    {
                        SEMANTIC_SEARCH_METADATA_DEFS.TYPE: content_type,
                        SEMANTIC_SEARCH_METADATA_DEFS.NAME: name,
                        SEMANTIC_SEARCH_METADATA_DEFS.SOURCE: f"{content_type}: {name}",
                        SEMANTIC_SEARCH_METADATA_DEFS.CHUNK: 0,
                        SEMANTIC_SEARCH_METADATA_DEFS.IS_CHUNKED: False,
                    }
                )
                ids.append(f"{content_type}_{name}")

    return documents, metadatas, ids


def process_regular_content(
    content_type: str, items: Dict[str, Any]
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Process regular content types without special handling.

    For content types that don't need chunking or special processing,
    indexes content directly if it meets minimum length requirements.

    Parameters
    ----------
    content_type : str
        Type of content
    items : Dict[str, Any]
        Dictionary mapping content names to content text

    Returns
    -------
    Tuple[List[str], List[Dict], List[str]]
        Tuple of (documents, metadatas, ids) ready for ChromaDB indexing
    """
    documents = []
    metadatas = []
    ids = []

    for name, content in items.items():
        if (
            content
            and len(str(content).strip()) > SEMANTIC_SEARCH_DEFS.MIN_CONTENT_LENGTH_LONG
        ):
            documents.append(str(content))
            metadatas.append(
                {
                    SEMANTIC_SEARCH_METADATA_DEFS.TYPE: content_type,
                    SEMANTIC_SEARCH_METADATA_DEFS.NAME: name,
                    SEMANTIC_SEARCH_METADATA_DEFS.SOURCE: f"{content_type}: {name}",
                    SEMANTIC_SEARCH_METADATA_DEFS.CHUNK: 0,
                    SEMANTIC_SEARCH_METADATA_DEFS.IS_CHUNKED: False,
                }
            )
            ids.append(f"{content_type}_{name}")

    return documents, metadatas, ids


def process_codebase_content(
    content_type: str, items: Dict[str, Any]
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Process codebase content (functions, classes) with proper formatting.

    For classes, extracts methods as separate searchable items.
    For functions, formats with full qualified names.
    """
    documents = []
    metadatas = []
    ids = []

    for name, info in items.items():
        if not isinstance(info, dict):
            continue

        # Process main class/function content
        # ReadTheDocs data structure uses CODEBASE_RTD_DEFS field names
        signature = info.get(CODEBASE_RTD_DEFS.SIGNATURE, "")
        doc = info.get(CODEBASE_RTD_DEFS.DOC, "")

        if doc or signature:
            # Format content for embedding
            if content_type == CODEBASE_DEFS.FUNCTIONS:
                # Use full qualified name: napistu.module.function_name()
                func_name = name.split(".")[-1]
                if signature and func_name in signature:
                    content = signature.replace(func_name, name, 1)
                else:
                    content = f"{name}()"
            elif content_type == CODEBASE_DEFS.CLASSES:
                # Use full qualified name: class napistu.module.ClassName
                content = f"class {name}"
            else:
                content = signature or name

            if doc:
                content = f"{content}\n\n{doc}"

            if len(content.strip()) > SEMANTIC_SEARCH_DEFS.MIN_CONTENT_LENGTH_SHORT:
                documents.append(content)
                metadatas.append(
                    {
                        SEMANTIC_SEARCH_METADATA_DEFS.TYPE: content_type,
                        SEMANTIC_SEARCH_METADATA_DEFS.NAME: name,
                        SEMANTIC_SEARCH_METADATA_DEFS.SOURCE: f"{content_type}: {name}",
                    }
                )
                ids.append(f"{content_type}_{name.replace('.', '_')}")

        # For classes, also process individual methods
        if content_type == CODEBASE_DEFS.CLASSES and CODEBASE_RTD_DEFS.METHODS in info:
            class_short_name = name.split(".")[-1]  # Just "SBML_dfs" not full path

            for method_name, method_info in info[CODEBASE_RTD_DEFS.METHODS].items():
                if isinstance(method_info, dict):
                    method_sig = method_info.get(CODEBASE_RTD_DEFS.SIGNATURE, "")
                    method_doc = method_info.get(CODEBASE_RTD_DEFS.DOC, "")

                    if method_doc or method_sig:
                        # Format as ClassName.method_name()
                        method_content = f"{class_short_name}.{method_name}()"
                        if method_doc:
                            method_content = f"{method_content}\n\n{method_doc}"

                        if (
                            len(method_content.strip())
                            > SEMANTIC_SEARCH_DEFS.MIN_CONTENT_LENGTH_SHORT
                        ):
                            documents.append(method_content)
                            metadatas.append(
                                {
                                    SEMANTIC_SEARCH_METADATA_DEFS.TYPE: CODEBASE_DEFS.METHODS,
                                    SEMANTIC_SEARCH_METADATA_DEFS.NAME: f"{name}.{method_name}",
                                    SEMANTIC_SEARCH_METADATA_DEFS.SOURCE: f"{SEMANTIC_SEARCH_DEFS.METHOD_SOURCE_PREFIX}{class_short_name}.{method_name}",
                                    SEMANTIC_SEARCH_METADATA_DEFS.CLASS_NAME: name,
                                }
                            )
                            ids.append(f"method_{name.replace('.', '_')}_{method_name}")

    return documents, metadatas, ids


# utility functions


def _chunk_content_smart(
    text: str,
    content_name: str,
    max_chunk_size: int = SEMANTIC_SEARCH_DEFS.MAX_CHUNK_SIZE,
) -> List[str]:
    """
    Smart chunking that works well for any structured content (wiki, README, etc.).

    Automatically detects document structure and chunks appropriately:
    1. Tries to split by markdown headers first
    2. Falls back to paragraph-based grouping if no headers or content is too long
    3. Ensures chunks don't exceed max_chunk_size

    Parameters
    ----------
    text : str
        Content to chunk
    content_name : str
        Name of the content for logging
    max_chunk_size : int, optional
        Maximum characters per chunk (default 1000)

    Returns
    -------
    List[str]
        List of well-structured chunks
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []

    # Step 1: Try to split by headers first (works for wiki, README, etc.)
    header_sections = _split_by_headers(text)

    logger.debug(
        f"Chunking '{content_name}': found {len(header_sections)} header sections"
    )

    # Check if we actually found meaningful header sections
    if len(header_sections) == 1 and header_sections[0] == text:
        # No headers found, use paragraph-based chunking for the entire text
        logger.info(
            f"No headers found in '{content_name}', using paragraph-based chunking"
        )
        paragraphs = text.split("\n\n")
        chunks = _group_paragraphs_semantically(paragraphs, max_chunk_size)
    else:
        # Process each header section
        for section in header_sections:
            if (
                len(section) <= max_chunk_size * 1.2
            ):  # Allow slightly larger for coherence
                chunks.append(section)
            else:
                # Step 2: Split long sections by paragraphs
                paragraphs = section.split("\n\n")
                logger.debug(
                    f"Splitting long section ({len(section)} chars) into {len(paragraphs)} paragraphs"
                )
                grouped_chunks = _group_paragraphs_semantically(
                    paragraphs, max_chunk_size
                )
                chunks.extend(grouped_chunks)

    logger.debug(
        f"Smart chunking '{content_name}': {len(text)} chars → {len(chunks)} chunks"
    )
    return chunks


def _find_best_break_point(text: str, start: int, end: int) -> int:
    """
    Find the best break point for chunking text at natural boundaries.

    Tries to break at paragraphs, then sentences, then words.

    Parameters
    ----------
    text : str
        Text to find break point in
    start : int
        Start position for the chunk
    end : int
        Desired end position for the chunk

    Returns
    -------
    int
        Best break point position
    """
    # Look for paragraph break (double newline)
    para_break = text.rfind("\n\n", start, end)
    if para_break > start + (end - start) // 2:
        return para_break + 2

    # Look for line break
    line_break = text.rfind("\n", start, end)
    if line_break > start + (end - start) // 2:
        return line_break + 1

    # Look for sentence end
    sent_break = text.rfind(". ", start, end)
    if sent_break > start + (end - start) // 2:
        return sent_break + 2

    # Fall back to word boundary
    word_break = text.rfind(" ", start, end)
    if word_break > start + (end - start) // 2:
        return word_break + 1

    # Last resort: break at end
    return end


def _format_modules_for_chunking(modules: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert module dictionaries to string format for chunking.

    Extracts module docstrings from ReadTheDocs module dictionaries.

    Parameters
    ----------
    modules : Dict[str, Any]
        Dictionary mapping module names to module info dictionaries

    Returns
    -------
    Dict[str, str]
        Dictionary mapping module names to docstring content
    """
    return {
        name: info.get(CODEBASE_RTD_DEFS.DOC, "")
        for name, info in modules.items()
        if isinstance(info, dict) and info.get(CODEBASE_RTD_DEFS.DOC)
    }


def _group_paragraphs_semantically(
    paragraphs: List[str], max_chunk_size: int = SEMANTIC_SEARCH_DEFS.MAX_CHUNK_SIZE
) -> List[str]:
    """
    Group related paragraphs into semantically coherent chunks.

    Uses a simple heuristic: consecutive paragraphs that together don't exceed
    max_chunk_size are kept together. More sophisticated semantic similarity
    could be added here using embeddings.

    Parameters
    ----------
    paragraphs : List[str]
        List of paragraph strings
    max_chunk_size : int, optional
        Maximum characters per chunk (default 1000)

    Returns
    -------
    List[str]
        List of grouped paragraph chunks
    """
    if not paragraphs:
        return []

    # Special case: if there's only one paragraph and it's too long, split it
    if len(paragraphs) == 1 and len(paragraphs[0].strip()) > max_chunk_size:
        logger.debug(
            f"Single long paragraph ({len(paragraphs[0].strip())} chars), splitting at sentences"
        )
        return _split_long_paragraph(paragraphs[0].strip(), max_chunk_size)

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If this single paragraph is too long, split it
        if len(para) > max_chunk_size:
            # First, finalize any current chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Split the long paragraph and add all chunks
            para_chunks = _split_long_paragraph(para, max_chunk_size)
            chunks.extend(para_chunks)
            continue

        para_length = len(para)

        # If adding this paragraph would exceed limit, finalize current chunk
        if current_length + para_length + 2 > max_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_length
        else:
            current_chunk.append(para)
            current_length += para_length + 2  # +2 for \n\n

    # Add final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _split_by_headers(text: str) -> List[str]:
    """Split text by markdown headers while preserving structure."""
    # Find all headers (##, ###, ####)
    header_pattern = r"^(#{2,4})\s+(.+)$"
    lines = text.split("\n")

    sections = []
    current_section = []

    for line in lines:
        header_match = re.match(header_pattern, line, re.MULTILINE)

        if header_match:
            # Save previous section
            if current_section:
                section_text = "\n".join(current_section)
                if section_text.strip():
                    sections.append(section_text)

            # Start new section with header
            current_section = [line]
        else:
            current_section.append(line)

    # Add final section
    if current_section:
        section_text = "\n".join(current_section)
        if section_text.strip():
            sections.append(section_text)

    # If no headers found, return original text
    return sections if sections else [text]


def _split_long_paragraph(text: str, max_chunk_size: int) -> List[str]:
    """
    Split a single long paragraph into smaller chunks at sentence boundaries.

    This handles the case where content has no paragraph breaks but is too long
    to be a single chunk.
    """
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    sentences = text.split(". ")

    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Add period back if it was split off (except for last sentence)
        if not sentence.endswith(".") and sentence != sentences[-1]:
            sentence += "."

        sentence_length = len(sentence)

        # If adding this sentence would exceed limit, finalize current chunk
        if current_length + sentence_length + 1 > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
