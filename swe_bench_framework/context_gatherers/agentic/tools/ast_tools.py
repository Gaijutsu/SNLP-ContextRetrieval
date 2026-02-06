"""
AST-based analysis tools for agentic exploration.

This module provides tools for analyzing code structure using AST parsing,
including class hierarchy analysis and call graph generation.
"""

import ast
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import SearchTool, validate_parameters
from .search_tools import ASTAnalyzer
from ..environment import Observation


class ClassHierarchyBuilder:
    """Builds and analyzes class inheritance hierarchies."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._class_index: Dict[str, Dict[str, Any]] = {}
        self._inheritance_graph: Dict[str, List[str]] = {}
        self._built = False
    
    def build_index(self) -> None:
        """Build index of all classes and their inheritance relationships."""
        if self._built:
            return
        
        python_files = []
        for root, _, files in os.walk(self.repo_path):
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['__pycache__', 'node_modules', 'venv', '.git']):
                continue
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            tree = ASTAnalyzer.parse_file(file_path)
            if tree is None:
                continue
            
            classes = ASTAnalyzer.find_classes(tree)
            rel_path = os.path.relpath(file_path, self.repo_path)
            
            for cls in classes:
                class_key = cls['name']
                self._class_index[class_key] = {
                    'name': cls['name'],
                    'file': rel_path,
                    'line': cls['line'],
                    'bases': cls['bases'],
                    'methods': cls['methods']
                }
                self._inheritance_graph[class_key] = cls['bases']
        
        self._built = True
    
    def get_ancestors(self, class_name: str) -> List[str]:
        """Get all ancestor classes (parents, grandparents, etc.)."""
        if not self._built:
            self.build_index()
        
        ancestors = []
        to_process = [class_name]
        visited = set()
        
        while to_process:
            current = to_process.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            bases = self._inheritance_graph.get(current, [])
            for base in bases:
                if base not in visited and base in self._class_index:
                    ancestors.append(base)
                    to_process.append(base)
        
        return ancestors
    
    def get_descendants(self, class_name: str) -> List[str]:
        """Get all descendant classes (children, grandchildren, etc.)."""
        if not self._built:
            self.build_index()
        
        descendants = []
        for cls_name, bases in self._inheritance_graph.items():
            if class_name in bases:
                descendants.append(cls_name)
                descendants.extend(self.get_descendants(cls_name))
        
        return descendants
    
    def get_class_info(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a class."""
        if not self._built:
            self.build_index()
        return self._class_index.get(class_name)


class CallGraphBuilder:
    """Builds and analyzes function call graphs."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self._call_graph: Dict[str, Set[str]] = {}
        self._function_index: Dict[str, Dict[str, Any]] = {}
        self._built = False
    
    def build_index(self) -> None:
        """Build index of all functions and their call relationships."""
        if self._built:
            return
        
        python_files = []
        for root, _, files in os.walk(self.repo_path):
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['__pycache__', 'node_modules', 'venv', '.git']):
                continue
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            tree = ASTAnalyzer.parse_file(file_path)
            if tree is None:
                continue
            
            rel_path = os.path.relpath(file_path, self.repo_path)
            self._analyze_file(tree, rel_path)
        
        self._built = True
    
    def _analyze_file(self, tree: ast.AST, file_path: str) -> None:
        """Analyze a single file for function definitions and calls."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = self._get_full_function_name(node, tree)
                if func_name:
                    self._function_index[func_name] = {
                        'name': node.name,
                        'file': file_path,
                        'line': node.lineno,
                        'args': ASTAnalyzer._get_args(node)
                    }
                    
                    # Find all calls within this function
                    calls = self._find_calls(node)
                    self._call_graph[func_name] = calls
    
    def _get_full_function_name(self, node: ast.FunctionDef, tree: ast.AST) -> Optional[str]:
        """Get the full qualified name of a function (including class if method)."""
        # Check if it's a method
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return f"{parent.name}.{node.name}"
        return node.name
    
    def _find_calls(self, node: ast.AST) -> Set[str]:
        """Find all function calls within a node."""
        calls = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child.func)
                if call_name:
                    calls.add(call_name)
        return calls
    
    def _get_call_name(self, node: ast.AST) -> Optional[str]:
        """Get the name of a function call."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._get_call_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        return None
    
    def get_callees(self, function_name: str) -> List[str]:
        """Get all functions called by a given function."""
        if not self._built:
            self.build_index()
        return list(self._call_graph.get(function_name, set()))
    
    def get_callers(self, function_name: str) -> List[str]:
        """Get all functions that call a given function."""
        if not self._built:
            self.build_index()
        
        callers = []
        for caller, callees in self._call_graph.items():
            if function_name in callees:
                callers.append(caller)
        return callers
    
    def get_function_info(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a function."""
        if not self._built:
            self.build_index()
        return self._function_index.get(function_name)


class GetClassHierarchyTool(SearchTool):
    """
    Tool for retrieving class inheritance hierarchy.
    
    Provides information about parent and child classes.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="get_class_hierarchy",
            description="Get the inheritance hierarchy for a class",
            repo_path=repo_path
        )
        self._hierarchy_builder: Optional[ClassHierarchyBuilder] = None
    
    def _get_builder(self) -> ClassHierarchyBuilder:
        """Get or create the hierarchy builder."""
        if self._hierarchy_builder is None:
            self._hierarchy_builder = ClassHierarchyBuilder(self.repo_path)
            self._hierarchy_builder.build_index()
        return self._hierarchy_builder
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Get class hierarchy information.
        
        Parameters:
            - class_name: Name of the class (required)
            - include_ancestors: Include parent classes (default: True)
            - include_descendants: Include child classes (default: False)
            
        Returns:
            Observation with hierarchy information
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['class_name'],
            optional={
                'include_ancestors': True,
                'include_descendants': False
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        class_name = params['class_name']
        include_ancestors = params['include_ancestors']
        include_descendants = params['include_descendants']
        
        try:
            builder = self._get_builder()
            class_info = builder.get_class_info(class_name)
            
            if class_info is None:
                return Observation(
                    output=f"Class '{class_name}' not found in the codebase.",
                    metadata={'class_name': class_name, 'found': False}
                )
            
            result = {
                'class_name': class_name,
                'file': class_info['file'],
                'line': class_info['line'],
                'direct_bases': class_info['bases'],
                'found': True
            }
            
            lines = [f"Class: {class_name}"]
            lines.append(f"  Location: {class_info['file']}:{class_info['line']}")
            lines.append(f"  Direct bases: {', '.join(class_info['bases']) if class_info['bases'] else 'None'}")
            lines.append(f"  Methods: {len(class_info['methods'])}")
            
            if include_ancestors:
                ancestors = builder.get_ancestors(class_name)
                result['ancestors'] = ancestors
                if ancestors:
                    lines.append(f"  All ancestors: {', '.join(ancestors)}")
            
            if include_descendants:
                descendants = builder.get_descendants(class_name)
                result['descendants'] = descendants
                if descendants:
                    lines.append(f"  Descendants: {', '.join(descendants)}")
            
            return Observation(
                output="\n".join(lines),
                metadata=result
            )
            
        except Exception as e:
            return Observation(
                error=f"Error getting class hierarchy: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'class_name': {
                    'type': 'string',
                    'description': 'Name of the class'
                },
                'include_ancestors': {
                    'type': 'boolean',
                    'description': 'Include parent classes',
                    'default': True
                },
                'include_descendants': {
                    'type': 'boolean',
                    'description': 'Include child classes',
                    'default': False
                }
            },
            'required': ['class_name']
        }


class GetCallGraphTool(SearchTool):
    """
    Tool for retrieving function call relationships.
    
    Provides information about which functions call or are called by
    a given function.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="get_call_graph",
            description="Get function call relationships",
            repo_path=repo_path
        )
        self._call_graph_builder: Optional[CallGraphBuilder] = None
    
    def _get_builder(self) -> CallGraphBuilder:
        """Get or create the call graph builder."""
        if self._call_graph_builder is None:
            self._call_graph_builder = CallGraphBuilder(self.repo_path)
            self._call_graph_builder.build_index()
        return self._call_graph_builder
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Get call graph information for a function.
        
        Parameters:
            - function_name: Name of the function (required)
            - include_callees: Include functions called by this one (default: True)
            - include_callers: Include functions that call this one (default: True)
            
        Returns:
            Observation with call graph information
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['function_name'],
            optional={
                'include_callees': True,
                'include_callers': True
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        function_name = params['function_name']
        include_callees = params['include_callees']
        include_callers = params['include_callers']
        
        try:
            builder = self._get_builder()
            func_info = builder.get_function_info(function_name)
            
            result = {
                'function_name': function_name,
                'found': func_info is not None
            }
            
            lines = [f"Function: {function_name}"]
            
            if func_info:
                lines.append(f"  Location: {func_info['file']}:{func_info['line']}")
                lines.append(f"  Args: {func_info['args']}")
            
            if include_callees:
                callees = builder.get_callees(function_name)
                result['callees'] = callees
                if callees:
                    lines.append(f"  Calls: {', '.join(callees[:10])}")
                    if len(callees) > 10:
                        lines.append(f"    ... and {len(callees) - 10} more")
                else:
                    lines.append("  Calls: (none found)")
            
            if include_callers:
                callers = builder.get_callers(function_name)
                result['callers'] = callers
                if callers:
                    lines.append(f"  Called by: {', '.join(callers[:10])}")
                    if len(callers) > 10:
                        lines.append(f"    ... and {len(callers) - 10} more")
                else:
                    lines.append("  Called by: (none found)")
            
            return Observation(
                output="\n".join(lines),
                metadata=result
            )
            
        except Exception as e:
            return Observation(
                error=f"Error getting call graph: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'function_name': {
                    'type': 'string',
                    'description': 'Name of the function'
                },
                'include_callees': {
                    'type': 'boolean',
                    'description': 'Include functions called by this one',
                    'default': True
                },
                'include_callers': {
                    'type': 'boolean',
                    'description': 'Include functions that call this one',
                    'default': True
                }
            },
            'required': ['function_name']
        }
