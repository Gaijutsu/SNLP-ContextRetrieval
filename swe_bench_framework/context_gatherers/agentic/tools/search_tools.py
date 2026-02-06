"""
Search tools for agentic exploration using AST parsing.

This module provides tools for searching classes and methods/functions
in Python code using abstract syntax tree (AST) parsing for accurate results.
"""

import ast
import os
from typing import Any, Dict, List, Optional, Tuple

from .base import SearchTool, validate_parameters
from ..environment import Observation


class ASTAnalyzer:
    """Helper class for AST-based code analysis."""
    
    @staticmethod
    def parse_file(file_path: str) -> Optional[ast.AST]:
        """Parse a Python file and return its AST."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return ast.parse(content)
        except SyntaxError:
            return None
        except Exception:
            return None
    
    @staticmethod
    def find_classes(tree: ast.AST) -> List[Dict[str, Any]]:
        """Find all class definitions in an AST."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'bases': [ASTAnalyzer._get_name(base) for base in node.bases],
                    'methods': ASTAnalyzer._get_methods(node)
                })
        return classes
    
    @staticmethod
    def find_functions(tree: ast.AST) -> List[Dict[str, Any]]:
        """Find all function definitions in an AST (excluding methods)."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not ASTAnalyzer._is_method(node, tree):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'end_line': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'args': ASTAnalyzer._get_args(node)
                })
        return functions
    
    @staticmethod
    def find_methods(tree: ast.AST, class_name: str) -> List[Dict[str, Any]]:
        """Find all methods in a specific class."""
        methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                methods = ASTAnalyzer._get_methods(node)
                break
        return methods
    
    @staticmethod
    def find_method_in_class(tree: ast.AST, class_name: str, method_name: str) -> Optional[Dict[str, Any]]:
        """Find a specific method in a class."""
        methods = ASTAnalyzer.find_methods(tree, class_name)
        for method in methods:
            if method['name'] == method_name:
                return method
        return None
    
    @staticmethod
    def _get_methods(class_node: ast.ClassDef) -> List[Dict[str, Any]]:
        """Extract method info from a class node."""
        methods = []
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append({
                    'name': item.name,
                    'line': item.lineno,
                    'end_line': item.end_lineno if hasattr(item, 'end_lineno') else item.lineno,
                    'args': ASTAnalyzer._get_args(item)
                })
        return methods
    
    @staticmethod
    def _get_args(func_node: ast.FunctionDef) -> List[str]:
        """Extract argument names from a function node."""
        args = []
        for arg in func_node.args.args:
            args.append(arg.arg)
        return args
    
    @staticmethod
    def _is_method(node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function node is a method (inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    @staticmethod
    def _get_name(node: ast.AST) -> str:
        """Get the name from an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return str(node)


class SearchClassTool(SearchTool):
    """
    Tool for searching class definitions in the codebase.
    
    Uses AST parsing to find class definitions accurately.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="search_class",
            description="Search for class definitions in the codebase",
            repo_path=repo_path
        )
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Search for class definitions.
        
        Parameters:
            - class_name: Name of the class to search for (required)
            - exact_match: Whether to require exact match (default: False)
            - max_results: Maximum number of results (default: 10)
            
        Returns:
            Observation with class definitions found
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['class_name'],
            optional={
                'exact_match': False,
                'max_results': 10
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        class_name = params['class_name']
        exact_match = params['exact_match']
        max_results = params['max_results']
        
        try:
            results = []
            python_files = self._get_python_files()
            
            for file_path in python_files:
                if len(results) >= max_results:
                    break
                
                tree = ASTAnalyzer.parse_file(file_path)
                if tree is None:
                    continue
                
                classes = ASTAnalyzer.find_classes(tree)
                
                for cls in classes:
                    name_match = (
                        cls['name'] == class_name if exact_match
                        else class_name.lower() in cls['name'].lower()
                    )
                    
                    if name_match:
                        rel_path = os.path.relpath(file_path, self.repo_path)
                        results.append({
                            'file': rel_path,
                            'class_name': cls['name'],
                            'line': cls['line'],
                            'end_line': cls['end_line'],
                            'bases': cls['bases'],
                            'methods': [m['name'] for m in cls['methods']]
                        })
                        
                        if len(results) >= max_results:
                            break
            
            # Format output
            if not results:
                match_type = "exact" if exact_match else "partial"
                output = f"No classes found with {match_type} match for: {class_name}"
            else:
                lines = [f"Found {len(results)} classes matching '{class_name}':"]
                for r in results:
                    base_str = f"({', '.join(r['bases'])})" if r['bases'] else ""
                    lines.append(f"  {r['class_name']}{base_str} in {r['file']}:{r['line']}")
                    if r['methods']:
                        lines.append(f"    Methods: {', '.join(r['methods'][:5])}")
                output = "\n".join(lines)
            
            return Observation(
                output=output,
                metadata={
                    'class_name': class_name,
                    'exact_match': exact_match,
                    'results_count': len(results),
                    'results': results
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error searching for class: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'class_name': {
                    'type': 'string',
                    'description': 'Name of the class to search for'
                },
                'exact_match': {
                    'type': 'boolean',
                    'description': 'Require exact name match',
                    'default': False
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results',
                    'default': 10
                }
            },
            'required': ['class_name']
        }


class SearchMethodTool(SearchTool):
    """
    Tool for searching method/function definitions in the codebase.
    
    Uses AST parsing to find function and method definitions accurately.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="search_method",
            description="Search for method/function definitions in the codebase",
            repo_path=repo_path
        )
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Search for method/function definitions.
        
        Parameters:
            - method_name: Name of the method/function to search for (required)
            - class_name: Optional class name to search within
            - file_path: Optional file path to search within
            - exact_match: Whether to require exact match (default: True)
            - max_results: Maximum number of results (default: 10)
            
        Returns:
            Observation with method definitions found
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['method_name'],
            optional={
                'class_name': None,
                'file_path': None,
                'exact_match': True,
                'max_results': 10
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        method_name = params['method_name']
        class_name = params['class_name']
        file_path = params['file_path']
        exact_match = params['exact_match']
        max_results = params['max_results']
        
        try:
            results = []
            
            # Determine files to search
            if file_path:
                resolved = os.path.join(self.repo_path, file_path)
                if os.path.exists(resolved):
                    python_files = [resolved]
                else:
                    return Observation(
                        error=f"File not found: {file_path}",
                        success=False
                    )
            else:
                python_files = self._get_python_files()
            
            for file_path_abs in python_files:
                if len(results) >= max_results:
                    break
                
                tree = ASTAnalyzer.parse_file(file_path_abs)
                if tree is None:
                    continue
                
                rel_path = os.path.relpath(file_path_abs, self.repo_path)
                
                if class_name:
                    # Search for method in specific class
                    method = ASTAnalyzer.find_method_in_class(tree, class_name, method_name)
                    if method:
                        results.append({
                            'file': rel_path,
                            'class_name': class_name,
                            'method_name': method['name'],
                            'line': method['line'],
                            'end_line': method['end_line'],
                            'args': method['args']
                        })
                else:
                    # Search for function/method anywhere
                    # First check standalone functions
                    functions = ASTAnalyzer.find_functions(tree)
                    for func in functions:
                        name_match = (
                            func['name'] == method_name if exact_match
                            else method_name.lower() in func['name'].lower()
                        )
                        if name_match:
                            results.append({
                                'file': rel_path,
                                'class_name': None,
                                'method_name': func['name'],
                                'line': func['line'],
                                'end_line': func['end_line'],
                                'args': func['args']
                            })
                            if len(results) >= max_results:
                                break
                    
                    # Then check class methods
                    classes = ASTAnalyzer.find_classes(tree)
                    for cls in classes:
                        if len(results) >= max_results:
                            break
                        for method in cls['methods']:
                            name_match = (
                                method['name'] == method_name if exact_match
                                else method_name.lower() in method['name'].lower()
                            )
                            if name_match:
                                results.append({
                                    'file': rel_path,
                                    'class_name': cls['name'],
                                    'method_name': method['name'],
                                    'line': method['line'],
                                    'end_line': method['end_line'],
                                    'args': method['args']
                                })
                                if len(results) >= max_results:
                                    break
            
            # Format output
            if not results:
                location = f" in class {class_name}" if class_name else ""
                output = f"No methods found matching: {method_name}{location}"
            else:
                lines = [f"Found {len(results)} methods/functions matching '{method_name}':"]
                for r in results:
                    class_str = f"{r['class_name']}." if r['class_name'] else ""
                    args_str = f"({', '.join(r['args'])})" if r['args'] else "()"
                    lines.append(f"  {class_str}{r['method_name']}{args_str} in {r['file']}:{r['line']}")
                output = "\n".join(lines)
            
            return Observation(
                output=output,
                metadata={
                    'method_name': method_name,
                    'class_name': class_name,
                    'exact_match': exact_match,
                    'results_count': len(results),
                    'results': results
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error searching for method: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'method_name': {
                    'type': 'string',
                    'description': 'Name of the method/function to search for'
                },
                'class_name': {
                    'type': 'string',
                    'description': 'Optional class name to search within',
                    'default': None
                },
                'file_path': {
                    'type': 'string',
                    'description': 'Optional file path to search within',
                    'default': None
                },
                'exact_match': {
                    'type': 'boolean',
                    'description': 'Require exact name match',
                    'default': True
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results',
                    'default': 10
                }
            },
            'required': ['method_name']
        }
