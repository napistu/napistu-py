"""
Utilities for loading and processing tutorials.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import optional dependencies with error handling
try:
    import nbformat
except ImportError:
    raise ImportError(
        "Tutorial utilities require additional dependencies. Install with 'pip install napistu[mcp]'"
    )

async def load_tutorial_index(tutorials_path: str) -> List[Dict[str, Any]]:
    """
    Load the index of available tutorials.
    
    Args:
        tutorials_path: Path to the tutorials directory
    
    Returns:
        List of tutorial metadata
    """
    tutorials = []
    
    try:
        tutorials_dir = Path(tutorials_path)
        
        # Find notebook files
        notebook_files = list(tutorials_dir.glob('**/*.ipynb'))
        
        for idx, notebook_path in enumerate(notebook_files):
            # Extract tutorial ID from path
            rel_path = notebook_path.relative_to(tutorials_dir)
            tutorial_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.ipynb', '')
            
            # Load notebook metadata
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = nbformat.read(f, as_version=4)
            
            # Extract title and description
            title = notebook_path.stem.replace('_', ' ').title()
            description = ""
            
            # Try to extract title and description from first cell
            if notebook.cells and notebook.cells[0].cell_type == 'markdown':
                lines = notebook.cells[0].source.split('\n')
                
                # First line with # is title
                for line in lines:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
                
                # Description is first paragraph after title
                description_lines = []
                in_description = False
                
                for line in lines:
                    if line.startswith('# '):
                        in_description = True
                        continue
                    
                    if in_description and line.strip():
                        description_lines.append(line)
                    elif in_description and description_lines:
                        break
                
                if description_lines:
                    description = ' '.join(description_lines)
            
            tutorials.append({
                "id": tutorial_id,
                "title": title,
                "description": description,
                "path": str(rel_path),
            })
    
    except Exception as e:
        print(f"Error loading tutorial index from {tutorials_path}: {e}")
    
    return tutorials

async def get_tutorial_content(tutorials_path: str, tutorial_id: str) -> str:
    """
    Get the content of a tutorial.
    
    Args:
        tutorials_path: Path to the tutorials directory
        tutorial_id: ID of the tutorial
    
    Returns:
        Tutorial content as a string
    """
    try:
        tutorials_dir = Path(tutorials_path)
        
        # Convert tutorial ID back to path
        path_parts = tutorial_id.split('_')
        notebook_name = path_parts[-1] + '.ipynb'
        directory_parts = path_parts[:-1]
        
        notebook_path = tutorials_dir
        for part in directory_parts:
            notebook_path = notebook_path / part
        notebook_path = notebook_path / notebook_name
        
        # Fallback: search for the notebook
        if not notebook_path.exists():
            for path in tutorials_dir.glob('**/*.ipynb'):
                rel_path = path.relative_to(tutorials_dir)
                candidate_id = str(rel_path).replace('/', '_').replace('\\', '_').replace('.ipynb', '')
                if candidate_id == tutorial_id:
                    notebook_path = path
                    break
        
        # Load notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Convert to markdown
        markdown_content = notebook_to_markdown(notebook)
        
        return markdown_content
    
    except Exception as e:
        print(f"Error getting tutorial content for {tutorial_id}: {e}")
        return f"Error loading tutorial: {e}"

def notebook_to_markdown(notebook) -> str:
    """
    Convert a Jupyter notebook to Markdown.
    
    Args:
        notebook: nbformat notebook object
    
    Returns:
        Markdown representation of the notebook
    """
    markdown = []
    
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            markdown.append(cell.source)
        elif cell.cell_type == 'code':
            # Add code cell as a fenced code block
            markdown.append("```python")
            markdown.append(cell.source)
            markdown.append("```")
            
            # Add outputs if any
            if cell.outputs:
                markdown.append("\nOutput:")
                for output in cell.outputs:
                    if 'text' in output:
                        markdown.append("```")
                        markdown.append(output['text'])
                        markdown.append("```")
                    elif 'data' in output:
                        if 'text/plain' in output['data']:
                            markdown.append("```")
                            markdown.append(output['data']['text/plain'])
                            markdown.append("```")
        
        # Add a separator between cells
        markdown.append("\n---\n")
    
    return "\n".join(markdown)

def get_snippet(text: str, query: str, context: int = 100) -> str:
    """
    Get a text snippet around a search term.
    
    Args:
        text: Text to search in
        query: Search term
        context: Number of characters to include before and after the match
    
    Returns:
        Text snippet
    """
    query = query.lower()
    text_lower = text.lower()
    
    if query not in text_lower:
        return ""
    
    start_pos = text_lower.find(query)
    start = max(0, start_pos - context)
    end = min(len(text), start_pos + len(query) + context)
    
    snippet = text[start:end]
    
    # Add ellipsis if we're not at the beginning or end
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet