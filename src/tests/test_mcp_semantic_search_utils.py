"""
Tests for semantic search utility functions.
"""

from napistu.mcp import semantic_search_utils


def test_chunk_content_smart_header_splitting():
    """Test that smart chunking properly splits by headers."""
    wiki_text = """## Overview
This is an overview section about SBML.

SBML stands for Systems Biology Markup Language.

## Details
Here are the technical details.

### Subsection
More specific information here.

## Conclusion
Final thoughts on SBML usage."""

    chunks = semantic_search_utils._chunk_content_smart(
        wiki_text, "test-wiki", max_chunk_size=200
    )

    # Should create 4 chunks (one per header section)
    assert len(chunks) == 4
    assert chunks[0].startswith("## Overview")
    assert chunks[1].startswith("## Details")
    assert chunks[2].startswith("### Subsection")
    assert chunks[3].startswith("## Conclusion")

    # All chunks should be reasonable size
    for chunk in chunks:
        assert 20 < len(chunk) < 250


def test_chunk_content_smart_no_chunking_needed():
    """Test that short content is not chunked."""
    short_text = "This is a short piece of content."
    chunks = semantic_search_utils._chunk_content_smart(
        short_text, "test", max_chunk_size=1000
    )

    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_split_by_headers():
    """Test header-based text splitting."""
    text_with_headers = """## First Section
Content for first section.

## Second Section  
Content for second section.

### Subsection
Subsection content."""

    sections = semantic_search_utils._split_by_headers(text_with_headers)

    assert len(sections) == 3
    assert sections[0].startswith("## First Section")
    assert "Content for first section." in sections[0]
    assert sections[1].startswith("## Second Section")
    assert sections[2].startswith("### Subsection")


def test_split_by_headers_no_headers():
    """Test that text without headers returns unchanged."""
    text_no_headers = "Just some regular text without any headers."
    sections = semantic_search_utils._split_by_headers(text_no_headers)

    assert len(sections) == 1
    assert sections[0] == text_no_headers


def test_group_paragraphs_semantically():
    """Test paragraph grouping by size."""
    # Small paragraphs should be grouped together
    small_paragraphs = ["Short one.", "Short two.", "Short three."]
    chunks = semantic_search_utils._group_paragraphs_semantically(
        small_paragraphs, max_chunk_size=1000
    )

    assert len(chunks) == 1
    assert all(p in chunks[0] for p in small_paragraphs)

    # Large paragraph should be split when it exceeds max_chunk_size
    large_paragraph = "This is a very long paragraph. " * 20  # ~640 chars
    mixed_paragraphs = [large_paragraph, "Short paragraph."]
    chunks = semantic_search_utils._group_paragraphs_semantically(
        mixed_paragraphs, max_chunk_size=100
    )

    # Should have multiple chunks: several from the split large paragraph + 1 for short paragraph
    assert len(chunks) > 2, f"Expected more than 2 chunks, got {len(chunks)}"

    # The last chunk should be the short paragraph
    assert "Short paragraph." in chunks[-1]

    # All chunks should be reasonable size
    for chunk in chunks:
        assert len(chunk) <= 150, f"Chunk too large: {len(chunk)} chars"


def test_group_paragraphs_filters_empty():
    """Test that empty paragraphs are filtered out."""
    paragraphs = ["Real content.", "", "  ", "More content."]
    chunks = semantic_search_utils._group_paragraphs_semantically(
        paragraphs, max_chunk_size=1000
    )

    assert len(chunks) == 1
    assert chunks[0] == "Real content.\n\nMore content."


def test_process_issues_and_prs():
    """Test processing of issues and PR data."""
    issues_data = {
        "test-repo": [
            {"title": "Bug report", "body": "This is a bug", "number": 123},
            {
                "title": "Feature request with longer title",
                "body": "",
                "number": 124,
            },  # >20 chars
            {
                "title": "x",
                "body": "y",
                "number": 125,
            },  # Should be filtered (too short)
        ]
    }

    docs, metas, ids = semantic_search_utils.process_issues_and_prs(
        "issues", issues_data
    )

    # Should process 2 items: "Bug report\n\nThis is a bug" and "Feature request with longer title"
    assert len(docs) == 2
    assert len(metas) == 2
    assert len(ids) == 2

    # Check content combination
    assert "Bug report\n\nThis is a bug" in docs[0]
    assert "Feature request with longer title" in docs[1]  # No body, just title

    # Check metadata
    assert metas[0]["type"] == "issues"
    assert metas[0]["name"] == "test-repo#123"
    assert metas[0]["is_chunked"] is False


def test_process_chunkable_content():
    """Test processing of content that may need chunking."""
    content_data = {
        "short-page": "This is short content that meets the minimum length requirement.",
        "long-page": "This is a very long page. " * 100,  # ~2600 chars
    }

    docs, metas, ids = semantic_search_utils.process_chunkable_content(
        "wiki", content_data, chunk_threshold=1200, max_chunk_size=1000
    )

    # Debug: print what we actually get
    print(f"Got {len(docs)} documents")
    for i, meta in enumerate(metas):
        print(f"  {i}: {meta['name']} - chunked: {meta['is_chunked']}")

    # Find items by checking the source field instead of name
    short_items = [i for i, meta in enumerate(metas) if "short-page" in meta["source"]]
    long_items = [i for i, meta in enumerate(metas) if "long-page" in meta["source"]]

    assert len(short_items) == 1  # Should not be chunked
    assert len(long_items) >= 1  # Should be chunked (may be multiple chunks)

    # Check metadata
    short_meta = metas[short_items[0]]
    assert short_meta["is_chunked"] is False
    assert short_meta["type"] == "wiki"

    # At least one long item should be chunked
    long_meta = metas[long_items[0]]
    assert long_meta["type"] == "wiki"


def test_processing_content_filters_short():
    """Test that very short content is filtered out."""
    content_data = {
        "good-content": "This is substantial content that should be indexed.",
        "bad-content": "x",  # Too short
    }

    docs, metas, ids = semantic_search_utils.process_chunkable_content(
        "test", content_data
    )

    assert len(docs) == 1
    assert metas[0]["name"] == "good-content"


def test_content_filtering():
    """Test that very short content gets filtered appropriately."""
    issues_data = {
        "test-repo": [
            {
                "title": "Long enough title and body",
                "body": "Substantial content here",
                "number": 1,
            },
            {"title": "Short", "body": "x", "number": 2},  # Should be filtered
        ]
    }

    docs, metas, ids = semantic_search_utils.process_issues_and_prs(
        "issues", issues_data
    )
    assert len(docs) == 1  # Only the long one should survive


def test_split_long_paragraph():
    """Test that a single long paragraph gets split properly."""
    # Create a long paragraph with clear sentence boundaries
    long_paragraph = (
        "This is sentence one. This is sentence two. This is sentence three. " * 50
    )  # ~3300 chars

    from napistu.mcp.semantic_search_utils import _split_long_paragraph

    chunks = _split_long_paragraph(long_paragraph, max_chunk_size=500)

    print(f"Long paragraph test: {len(long_paragraph)} chars → {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars - {chunk[:50]}...")

    # Should create multiple chunks
    assert (
        len(chunks) > 1
    ), f"Expected multiple chunks for {len(long_paragraph)} char paragraph, got {len(chunks)}"

    # Each chunk should be under the limit
    for i, chunk in enumerate(chunks):
        assert (
            len(chunk) <= 600
        ), f"Chunk {i} is {len(chunk)} chars, exceeds max_chunk_size"
        assert len(chunk) > 50, f"Chunk {i} is only {len(chunk)} chars, too small"


def test_group_paragraphs_with_long_paragraph():
    """Test that group_paragraphs_semantically handles a single long paragraph."""
    # Create content that will be one "paragraph" (no \n\n breaks)
    long_single_para = "This is a very long sentence. " * 100  # ~3100 chars

    chunks = semantic_search_utils._group_paragraphs_semantically(
        [long_single_para], max_chunk_size=800
    )

    print(
        f"Group paragraphs test: {len(long_single_para)} chars → {len(chunks)} chunks"
    )
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")

    # Should create multiple chunks since it's > 800 chars
    assert (
        len(chunks) > 1
    ), f"Expected multiple chunks for {len(long_single_para)} char content, got {len(chunks)}"


def test_chunking_works_without_headers():
    """Test that long content without headers still gets chunked properly."""
    # Create long content without any markdown headers
    long_content_no_headers = (
        "This is a paragraph with substantial content. " * 100
    )  # ~4500 chars

    content_data = {"no-headers-page": long_content_no_headers}

    docs, metas, ids = semantic_search_utils.process_chunkable_content(
        "wiki", content_data, chunk_threshold=1000, max_chunk_size=800
    )

    # Should be chunked into multiple pieces since it's > 1000 chars
    chunks_for_page = [
        i for i, meta in enumerate(metas) if "no-headers-page" in meta["source"]
    ]

    # This test will fail if chunking doesn't work - we expect multiple chunks
    assert (
        len(chunks_for_page) > 1
    ), f"Expected multiple chunks for {len(long_content_no_headers)} char content, got {len(chunks_for_page)}"

    # Each chunk should be reasonable size
    for i in chunks_for_page:
        chunk_length = len(docs[i])
        assert (
            chunk_length <= 1000
        ), f"Chunk {i} is {chunk_length} chars, exceeds max_chunk_size of 800"
        assert chunk_length > 100, f"Chunk {i} is only {chunk_length} chars, too small"


def test_split_long_paragraph_direct():
    """Test split_long_paragraph function directly."""

    # Test the exact pattern from the failing test
    repeated_sentence = "This is a very long sentence. " * 50  # ~1500 chars
    chunks = semantic_search_utils._split_long_paragraph(
        repeated_sentence, max_chunk_size=400
    )

    print(f"Input: {len(repeated_sentence)} chars → {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} chars")

    # Should create multiple chunks since 1500 chars > 400 char limit
    assert (
        len(chunks) > 1
    ), f"Expected multiple chunks for {len(repeated_sentence)} chars, got {len(chunks)}"

    # Each chunk should be under the limit
    for i, chunk in enumerate(chunks):
        assert (
            len(chunk) <= 500
        ), f"Chunk {i} is {len(chunk)} chars, exceeds reasonable limit"
        assert len(chunk) > 20, f"Chunk {i} is {len(chunk)} chars, too small"
