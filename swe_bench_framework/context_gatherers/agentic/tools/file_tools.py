"""
File-related tools for agentic exploration.

This module provides tools for viewing files, searching file contents,
and finding files by name.
"""

import os
import re
from typing import Any, Dict, List, Optional

from .base import FileSystemTool, validate_parameters
from ..environment import Observation


class ViewFileTool(FileSystemTool):
    """
    Tool for viewing file contents with line limits.
    
    Similar to SWE-agent's file viewer, this tool displays a limited
    number of lines at a time to avoid overwhelming the agent with
    too much context.
    """
    
    def __init__(self, repo_path: str, default_limit: int = 100):
        super().__init__(
            name="view_file",
            description="View the contents of a file with optional line range",
            repo_path=repo_path
        )
        self.default_limit = default_limit
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        View a file's contents.
        
        Parameters:
            - path: Path to the file (required)
            - offset: Starting line number (0-indexed, default: 0)
            - limit: Maximum number of lines to view (default: 100)
            
        Returns:
            Observation with file content or error
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['path'],
            optional={'offset': 0, 'limit': self.default_limit}
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        file_path = params['path']
        offset = params['offset']
        limit = params['limit']
        
        try:
            resolved = self._resolve_path(file_path)
            
            if not self._is_path_safe(resolved):
                return Observation(
                    error=f"Path {file_path} is outside repository",
                    success=False
                )
            
            if not os.path.exists(resolved):
                return Observation(
                    error=f"File not found: {file_path}",
                    success=False
                )
            
            if os.path.isdir(resolved):
                return Observation(
                    error=f"{file_path} is a directory, not a file",
                    success=False
                )
            
            # Read file
            content = self._read_file(file_path, offset, limit)
            
            # Get total line count
            with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = len(f.readlines())
            
            # Format output
            end_line = min(offset + limit, total_lines)
            header = f"[File: {file_path} ({total_lines} lines total)]\n"
            header += f"[Lines {offset} to {end_line}]\n"
            
            output = header + "```\n" + content + "\n```"
            
            return Observation(
                output=output,
                metadata={
                    'file_path': file_path,
                    'offset': offset,
                    'limit': limit,
                    'total_lines': total_lines,
                    'lines_viewed': end_line - offset
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error reading file: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file to view'
                },
                'offset': {
                    'type': 'integer',
                    'description': 'Starting line number (0-indexed)',
                    'default': 0
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of lines to view',
                    'default': self.default_limit
                }
            },
            'required': ['path']
        }


class GrepTool(FileSystemTool):
    """
    Tool for searching file contents using grep-like functionality.
    
    Searches for patterns in files across the repository.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="grep",
            description="Search for a pattern in files across the repository",
            repo_path=repo_path
        )
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Search for a pattern in files.
        
        Parameters:
            - pattern: Pattern to search for (required)
            - path: Directory or file to search in (default: repo root)
            - file_pattern: File glob pattern (default: *.py)
            - case_sensitive: Whether search is case sensitive (default: False)
            - max_results: Maximum number of results (default: 50)
            
        Returns:
            Observation with search results
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['pattern'],
            optional={
                'path': '',
                'file_pattern': '*.py',
                'case_sensitive': False,
                'max_results': 50
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        pattern = params['pattern']
        search_path = params['path'] or self.repo_path
        file_pattern = params['file_pattern']
        case_sensitive = params['case_sensitive']
        max_results = params['max_results']
        
        try:
            resolved_path = self._resolve_path(search_path)
            
            if not self._is_path_safe(resolved_path):
                return Observation(
                    error=f"Path {search_path} is outside repository",
                    success=False
                )
            
            results = []
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)
            
            # Collect files to search
            files_to_search = []
            if os.path.isfile(resolved_path):
                files_to_search.append(resolved_path)
            else:
                for root, _, files in os.walk(resolved_path):
                    # Skip hidden and cache directories
                    if any(part.startswith('.') for part in root.split(os.sep)):
                        continue
                    if any(skip in root for skip in ['__pycache__', 'node_modules', 'venv']):
                        continue
                    
                    for file in files:
                        if file.endswith(file_pattern.replace('*', '')):
                            files_to_search.append(os.path.join(root, file))
            
            # Search files
            for file_path in files_to_search:
                if len(results) >= max_results:
                    break
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            if regex.search(line):
                                rel_path = os.path.relpath(file_path, self.repo_path)
                                results.append({
                                    'file': rel_path,
                                    'line': line_num,
                                    'content': line.strip()
                                })
                                if len(results) >= max_results:
                                    break
                except Exception:
                    continue
            
            # Format output
            if not results:
                output = f"No matches found for pattern: {pattern}"
            else:
                lines = [f"Found {len(results)} matches for '{pattern}':"]
                for r in results:
                    lines.append(f"  {r['file']}:{r['line']}: {r['content']}")
                output = "\n".join(lines)
            
            return Observation(
                output=output,
                metadata={
                    'pattern': pattern,
                    'results_count': len(results),
                    'results': results[:max_results]
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error searching: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'pattern': {
                    'type': 'string',
                    'description': 'Pattern to search for (regex supported)'
                },
                'path': {
                    'type': 'string',
                    'description': 'Directory or file to search in',
                    'default': ''
                },
                'file_pattern': {
                    'type': 'string',
                    'description': 'File glob pattern',
                    'default': '*.py'
                },
                'case_sensitive': {
                    'type': 'boolean',
                    'description': 'Case sensitive search',
                    'default': False
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results',
                    'default': 50
                }
            },
            'required': ['pattern']
        }


class FindFileTool(FileSystemTool):
    """
    Tool for finding files by name pattern.
    
    Similar to the Unix find command, this tool locates files
    matching a name pattern within the repository.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="find_file",
            description="Find files by name pattern",
            repo_path=repo_path
        )
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Find files matching a name pattern.
        
        Parameters:
            - pattern: File name pattern to match (required)
            - path: Directory to search in (default: repo root)
            - max_results: Maximum number of results (default: 20)
            
        Returns:
            Observation with matching file paths
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['pattern'],
            optional={
                'path': '',
                'max_results': 20
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        pattern = params['pattern']
        search_path = params['path'] or self.repo_path
        max_results = params['max_results']
        
        try:
            resolved_path = self._resolve_path(search_path)
            
            if not self._is_path_safe(resolved_path):
                return Observation(
                    error=f"Path {search_path} is outside repository",
                    success=False
                )
            
            results = []
            pattern_lower = pattern.lower()
            
            for root, dirs, files in os.walk(resolved_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                if any(skip in root for skip in ['__pycache__', 'node_modules', 'venv', '.git']):
                    continue
                
                for file in files:
                    if pattern_lower in file.lower():
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.repo_path)
                        results.append(rel_path)
                        
                        if len(results) >= max_results:
                            break
                
                if len(results) >= max_results:
                    break
            
            # Format output
            if not results:
                output = f"No files found matching: {pattern}"
            else:
                lines = [f"Found {len(results)} files matching '{pattern}':"]
                for r in results:
                    lines.append(f"  {r}")
                output = "\n".join(lines)
            
            return Observation(
                output=output,
                metadata={
                    'pattern': pattern,
                    'results_count': len(results),
                    'results': results
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error finding files: {str(e)}",
                success=False
            )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'pattern': {
                    'type': 'string',
                    'description': 'File name pattern to match'
                },
                'path': {
                    'type': 'string',
                    'description': 'Directory to search in',
                    'default': ''
                },
                'max_results': {
                    'type': 'integer',
                    'description': 'Maximum number of results',
                    'default': 20
                }
            },
            'required': ['pattern']
        }
