import pytest
from napistu.mcp.documentation_utils import load_readme_content, fetch_wiki_page
from napistu.mcp.constants import READMES, WIKI_PAGES


@pytest.mark.asyncio
@pytest.mark.parametrize("name,url", list(READMES.items())[:2])
async def test_load_readme_content(name, url):
    content = await load_readme_content(url)
    assert isinstance(content, str)
    assert len(content) > 0
    # Optionally, check for a keyword in the content
    assert "napistu" in content.lower() or "Napistu" in content


@pytest.mark.asyncio
@pytest.mark.parametrize("page_name", list(WIKI_PAGES)[:2])
async def test_fetch_wiki_page(page_name):
    content = await fetch_wiki_page(page_name)
    assert isinstance(content, str)
    assert len(content) > 0
    assert "napistu" in content.lower() or "Napistu" in content
