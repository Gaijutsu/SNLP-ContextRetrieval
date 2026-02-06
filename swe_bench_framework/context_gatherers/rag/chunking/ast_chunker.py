"""AST-based code chunking implementation.

This module provides chunking based on Python AST parsing, extracting semantic
units like functions, classes, and methods with their associated imports.
"""

import ast
from typing import List, Dict, Optional, Any, Set, Tuple
from pathlib import Path
import logging

from .base import CodeChunker
from ..base import CodeChunk

logger = logging.getLogger(__name__)


class ASTChunker(CodeChunker):
    """Chunk code using Python AST parsing.
    
    This chunker parses Python files into AST nodes and creates chunks for:
    - Functions (including async functions)
    - Classes (with their methods)
    - Module-level code
    - Import statements
    
    Each chunk includes relevant imports to maintain context.
    
    Example:
        ```python
        chunker = ASTChunker(config={"include_imports": True})
        chunks = chunker.chunk_file("example.py")
        
        for chunk in chunks:
            print(f"{chunk.chunk_type}: {chunk.name}")
            print(f"Lines {chunk.start_line}-{chunk.end_line}")
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AST chunker.
        
        Args:
            config: Configuration dictionary with options:
                - include_imports: Include imports with each chunk (default: True)
                - include_docstrings: Include docstrings in chunks (default: True)
                - separate_class_methods: Create separate chunks for methods (default: False)
                - include_module_level: Include module-level code chunks (default: True)
                - min_function_lines: Minimum lines for a function to be chunked (default: 2)
        """
        super().__init__(config)
        self.include_imports = config.get("include_imports", True)
        self.include_docstrings = config.get("include_docstrings", True)
        self.separate_class_methods = config.get("separate_class_methods", False)
        self.include_module_level = config.get("include_module_level", True)
        self.min_function_lines = config.get("min_function_lines", 2)
        self.file_extensions = [".py"]  # AST chunker only works with Python
    
    def chunk_file(self, file_path: str, content: Optional[str] = None) -> List[CodeChunk]:
        """Chunk a Python file using AST parsing.
        
        Args:
            file_path: Path to the Python file.
            content: Optional file content (if already loaded).
            
        Returns:
            List of CodeChunk objects.
        """
        if content is None:
            content = self._read_file(file_path)
        
        if not content.strip():
            return []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            # Fall back to treating entire file as one chunk
            lines = content.split("\n")
            return [CodeChunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                chunk_type="module",
                name=Path(file_path).stem,
                metadata={"parse_error": str(e)}
            )]
        
        # Extract all imports from the file
        imports = self._extract_imports(tree)
        
        # Create chunks for different AST node types
        chunks = []
        source_lines = content.split("\n")
        
        for node in ast.iter_child_nodes(tree):
            chunk = self._process_node(node, file_path, content, source_lines, imports)
            if chunk:
                chunks.append(chunk)
        
        # Filter small chunks
        chunks = self._filter_small_chunks(chunks)
        
        return chunks
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements from the AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_str = ast.unparse(node)
                imports.append(import_str)
            elif isinstance(node, ast.ImportFrom):
                import_str = ast.unparse(node)
                imports.append(import_str)
        
        return imports
    
    def _process_node(
        self,
        node: ast.AST,
        file_path: str,
        content: str,
        source_lines: List[str],
        imports: List[str]
    ) -> Optional[CodeChunk]:
        """Process a single AST node and create a chunk."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._process_function(node, file_path, content, source_lines, imports)
        elif isinstance(node, ast.ClassDef):
            return self._process_class(node, file_path, content, source_lines, imports)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Imports are handled separately
            return None
        elif self.include_module_level:
            return self._process_module_level(node, file_path, content, source_lines, imports)
        
        return None
    
    def _process_function(
        self,
        node: ast.FunctionDef,
        file_path: str,
        content: str,
        source_lines: List[str],
        imports: List[str]
    ) -> Optional[CodeChunk]:
        """Process a function definition node."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Skip if too small
        if end_line - start_line + 1 < self.min_function_lines:
            return None
        
        # Extract function content
        func_content = self._extract_node_content(node, content, source_lines)
        
        # Add imports if configured
        if self.include_imports and imports:
            func_content = "\n".join(imports) + "\n\n" + func_content
        
        return CodeChunk(
            content=func_content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="function",
            name=node.name,
            imports=imports if self.include_imports else [],
            metadata={
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "args": ast.unparse(node.args) if node.args else "",
            }
        )
    
    def _process_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        content: str,
        source_lines: List[str],
        imports: List[str]
    ) -> List[CodeChunk]:
        """Process a class definition node."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Extract class content
        class_content = self._extract_node_content(node, content, source_lines)
        
        # Add imports if configured
        if self.include_imports and imports:
            class_content = "\n".join(imports) + "\n\n" + class_content
        
        chunks = []
        
        # Create main class chunk
        class_chunk = CodeChunk(
            content=class_content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="class",
            name=node.name,
            imports=imports if self.include_imports else [],
            metadata={
                "bases": [ast.unparse(b) for b in node.bases],
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "method_count": len([n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]),
            }
        )
        chunks.append(class_chunk)
        
        # Optionally create separate chunks for methods
        if self.separate_class_methods:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = self._process_method(
                        child, node.name, file_path, content, source_lines, imports
                    )
                    if method_chunk:
                        chunks.append(method_chunk)
        
        return chunks
    
    def _process_method(
        self,
        node: ast.FunctionDef,
        class_name: str,
        file_path: str,
        content: str,
        source_lines: List[str],
        imports: List[str]
    ) -> Optional[CodeChunk]:
        """Process a method definition within a class."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Skip if too small
        if end_line - start_line + 1 < self.min_function_lines:
            return None
        
        method_content = self._extract_node_content(node, content, source_lines)
        
        # Add class context and imports
        if self.include_imports:
            context = f"# Class: {class_name}\n"
            if imports:
                context = "\n".join(imports) + "\n\n" + context
            method_content = context + method_content
        
        return CodeChunk(
            content=method_content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="method",
            name=f"{class_name}.{node.name}",
            imports=imports if self.include_imports else [],
            metadata={
                "class_name": class_name,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
            }
        )
    
    def _process_module_level(
        self,
        node: ast.AST,
        file_path: str,
        content: str,
        source_lines: List[str],
        imports: List[str]
    ) -> Optional[CodeChunk]:
        """Process module-level statements (assignments, expressions, etc.)."""
        # Skip certain node types
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            return None
        
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        
        node_content = self._extract_node_content(node, content, source_lines)
        
        # Skip if too small
        if len(node_content.strip()) < self.min_chunk_size:
            return None
        
        return CodeChunk(
            content=node_content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            chunk_type="module_level",
            name=None,
            imports=[],
            metadata={
                "node_type": type(node).__name__,
            }
        )
    
    def _extract_node_content(
        self,
        node: ast.AST,
        content: str,
        source_lines: List[str]
    ) -> str:
        """Extract the source content for an AST node."""
        start_line = getattr(node, 'lineno', 1) - 1  # Convert to 0-indexed
        end_line = getattr(node, 'end_lineno', start_line + 1)
        
        # Handle potential None values
        if end_line is None:
            end_line = start_line + 1
        
        # Extract lines
        node_lines = source_lines[start_line:end_line]
        return "\n".join(node_lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        stats = super().get_stats()
        stats.update({
            "include_imports": self.include_imports,
            "include_docstrings": self.include_docstrings,
            "separate_class_methods": self.separate_class_methods,
            "min_function_lines": self.min_function_lines,
        })
        return stats