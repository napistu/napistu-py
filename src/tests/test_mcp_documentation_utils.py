import pytest
from napistu.mcp.documentation_utils import load_readme_content
from napistu.mcp.constants import READMES


@pytest.mark.asyncio
@pytest.mark.parametrize("name,url", READMES.items())
async def test_load_readme_content(name, url):
    content = await load_readme_content(url)
    assert isinstance(content, str)
    assert len(content) > 0
    # Optionally, check for a keyword in the content
    assert "napistu" in content.lower() or "Napistu" in content
