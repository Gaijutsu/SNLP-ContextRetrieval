"""
Base Tool class for agentic exploration.

All tools used by agents must inherit from the Tool base class defined here
or in the environment module.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
import os

from ..environment import Tool, Observation


class FileSystemTool(Tool):
    """Base class for tools that operate on the file system."""
    
    def __init__(self, name: str, description: str, repo_path: str):
        super().__init__(name, description)
        self.repo_path = repo_path
    
    def _resolve_path(self, file_path: str) -> str:
        """
        Resolve a file path relative to the repository.
        
        Args:
            file_path: The file path (can be absolute or relative)
            
        Returns:
            Absolute path
        """
        if os.path.isabs(file_path):
            # Ensure it's within the repo
            if file_path.startswith(self.repo_path):
                return file_path
            else:
                # Try to interpret as relative
                return os.path.join(self.repo_path, file_path.lstrip('/'))
        return os.path.join(self.repo_path, file_path)
    
    def _is_path_safe(self, path: str) -> bool:
        """
        Check if a path is safe (within repo boundaries).
        
        Args:
            path: The path to check
            
        Returns:
            True if path is safe
        """
        abs_path = os.path.abspath(path)
        repo_abs = os.path.abspath(self.repo_path)
        return abs_path.startswith(repo_abs)
    
    def _read_file(self, file_path: str, offset: int = 0, limit: int = 100) -> str:
        """
        Read a file with line limits.
        
        Args:
            file_path: Path to the file
            offset: Starting line (0-indexed)
            limit: Maximum number of lines to read
            
        Returns:
            File content
        """
        resolved = self._resolve_path(file_path)
        
        if not self._is_path_safe(resolved):
            raise ValueError(f"Path {file_path} is outside repository")
        
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        start = offset
        end = min(offset + limit, len(lines))
        
        return ''.join(lines[start:end])


class SearchTool(Tool):
    """Base class for search tools."""
    
    def __init__(self, name: str, description: str, repo_path: str):
        super().__init__(name, description)
        self.repo_path = repo_path
    
    def _get_python_files(self) -> List[str]:
        """Get all Python files in the repository."""
        python_files = []
        for root, _, files in os.walk(self.repo_path):
            # Skip hidden directories and common non-source directories
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(skip in root for skip in ['__pycache__', 'node_modules', 'venv', '.git']):
                continue
            
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    python_files.append(full_path)
        
        return python_files
    
    def _normalize_symbol_name(self, name: str) -> str:
        """Normalize a symbol name for comparison."""
        return name.strip().lower().replace('_', '')


class ExecutionTool(Tool):
    """Base class for execution tools."""
    
    def __init__(self, name: str, description: str, repo_path: str):
        super().__init__(name, description)
        self.repo_path = repo_path
    
    def _run_command(
        self, 
        command: List[str], 
        timeout: int = 60,
        cwd: Optional[str] = None
    ) -> tuple:
        """
        Run a shell command.
        
        Args:
            command: Command and arguments as list
            timeout: Timeout in seconds
            cwd: Working directory
            
        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        import subprocess
        
        working_dir = cwd or self.repo_path
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir
            )
            return (
                result.stdout,
                result.stderr,
                result.returncode
            )
        except subprocess.TimeoutExpired:
            return (
                "",
                f"Command timed out after {timeout} seconds",
                -1
            )
        except Exception as e:
            return (
                "",
                f"Command execution failed: {str(e)}",
                -1
            )


def validate_parameters(
    parameters: Dict[str, Any],
    required: List[str],
    optional: Dict[str, Any] = None
) -> tuple:
    """
    Validate tool parameters.
    
    Args:
        parameters: Provided parameters
        required: List of required parameter names
        optional: Dict of optional parameters with defaults
        
    Returns:
        Tuple of (is_valid, error_message, validated_params)
    """
    optional = optional or {}
    
    # Check required parameters
    for param in required:
        if param not in parameters:
            return False, f"Missing required parameter: {param}", {}
    
    # Build validated params with defaults
    validated = {}
    for param in required:
        validated[param] = parameters[param]
    
    for param, default in optional.items():
        validated[param] = parameters.get(param, default)
    
    return True, "", validated
