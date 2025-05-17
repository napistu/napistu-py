"""
Utilities for scanning and analyzing the Napistu codebase.
"""

import os
import ast
import inspect
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

async def scan_codebase(codebase_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan the Napistu codebase and extract information about modules, classes, and functions.
    
    Args:
        codebase_path: Path to the Napistu codebase root
    
    Returns:
        Dictionary with module, class, and function information
    """
    modules = {}
    classes = {}
    functions = {}
    
    codebase_path = Path(codebase_path)
    
    # Find Python files
    python_files = list(codebase_path.glob('**/*.py'))
    
    for file_path in python_files:
        # Convert file path to module path
        rel_path = file_path.relative_to(codebase_path)
        module_parts = list(rel_path.parts)
        
        # Skip test files and hidden files
        if any(part.startswith('_') and part != '__init__.py' for part in module_parts) or 'test' in module_parts:
            continue
        
        # Remove file extension
        if module_parts[-1].endswith('.py'):
            module_parts[-1] = module_parts[-1][:-3]
        
        # Skip __init__.py
        if module_parts[-1] == '__init__':
            module_parts = module_parts[:-1]
        
        # Create module name
        if not module_parts:
            continue
        
        module_name = '.'.join(module_parts)
        
        # Extract information from the file
        module_info, module_classes, module_functions = await extract_file_info(file_path, module_name)
        
        if module_info:
            modules[module_name] = module_info
        
        # Add classes
        for class_name, class_info in module_classes.items():
            full_class_name = f"{module_name}.{class_name}" if module_name else class_name
            classes[full_class_name] = class_info
        
        # Add functions
        for func_name, func_info in module_functions.items():
            full_func_name = f"{module_name}.{func_name}" if module_name else func_name
            functions[full_func_name] = func_info
    
    # Try to import modules and add runtime information
    try:
        import napistu
        for module_name in modules:
            try:
                # Attempt to import the module to get runtime info
                module = importlib.import_module(f"napistu.{module_name}")
                
                # Update module information with runtime data
                modules[module_name]["version"] = getattr(module, "__version__", "unknown")
                
                # Update function signatures with runtime data
                for func_name, func_info in functions.items():
                    if func_name.startswith(module_name):
                        short_name = func_name[len(module_name) + 1:]
                        if hasattr(module, short_name):
                            func = getattr(module, short_name)
                            func_info["signature"] = str(inspect.signature(func))
            except (ImportError, AttributeError) as e:
                print(f"Could not import module {module_name}: {e}")
    except ImportError:
        print("Napistu package not available for runtime inspection")
    
    return {
        "modules": modules,
        "classes": classes,
        "functions": functions,
    }

async def extract_file_info(file_path: Path, module_name: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Extract information from a Python file.
    
    Args:
        file_path: Path to the Python file
        module_name: Name of the module
    
    Returns:
        Tuple of (module_info, class_info, function_info)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Parse the AST
        tree = ast.parse(code)
        
        # Extract module docstring
        module_docstring = ast.get_docstring(tree) or ""
        
        # Get classes and functions
        classes = {}
        functions = {}
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # Class definition
                class_docstring = ast.get_docstring(node) or ""
                
                class_methods = []
                for method in node.body:
                    if isinstance(method, ast.FunctionDef):
                        method_docstring = ast.get_docstring(method) or ""
                        
                        # Get method parameters
                        params = []
                        for arg in method.args.args:
                            # For Python 3.8+, use arg.arg
                            param_name = arg.arg if hasattr(arg, 'arg') else arg.id
                            params.append(param_name)
                        
                        class_methods.append({
                            "name": method.name,
                            "docstring": method_docstring,
                            "parameters": params,
                        })
                
                classes[node.name] = {
                    "name": node.name,
                    "docstring": class_docstring,
                    "methods": class_methods,
                }
            
            elif isinstance(node, ast.FunctionDef):
                # Function definition
                func_docstring = ast.get_docstring(node) or ""
                
                # Get function parameters
                params = []
                for arg in node.args.args:
                    # For Python 3.8+, use arg.arg
                    param_name = arg.arg if hasattr(arg, 'arg') else arg.id
                    params.append(param_name)
                
                functions[node.name] = {
                    "name": node.name,
                    "docstring": func_docstring,
                    "parameters": params,
                    "signature": f"{node.name}({', '.join(params)})",
                }
        
        # Create module info
        module_info = {
            "name": module_name,
            "docstring": module_docstring,
            "filepath": str(file_path),
            "classes": list(classes.keys()),
            "functions": list(functions.keys()),
        }
        
        return module_info, classes, functions
    
    except Exception as e:
        print(f"Error extracting info from {file_path}: {e}")
        return {}, {}, {}