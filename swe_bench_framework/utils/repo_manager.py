"""
Repository manager for the SWE-bench comparison framework.

This module provides utilities for managing repository paths
for SWE-bench instances.
"""

import os
from pathlib import Path
from typing import Any, Optional


class RepoManager:
    """
    Manager for repository paths.
    
    Handles resolving repository paths for SWE-bench instances.
    """
    
    def __init__(self, repos_dir: str = './repos'):
        """
        Initialize the repo manager.
        
        Args:
            repos_dir: Base directory for repositories
        """
        self.repos_dir = Path(repos_dir)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
    
    def get_repo_path(self, instance: Any) -> str:
        """
        Get the repository path for a given instance.
        
        Args:
            instance: SWE-bench instance with 'repo' attribute
            
        Returns:
            Path to the repository
        """
        # Handle different instance types
        if hasattr(instance, 'repo'):
            repo_name = instance.repo
        elif isinstance(instance, dict):
            repo_name = instance.get('repo', 'unknown')
        else:
            repo_name = str(instance)
        
        # Clean up repo name (handle formats like "django/django")
        repo_name = repo_name.replace('/', '_')
        
        repo_path = self.repos_dir / repo_name
        
        # Return path as string
        return str(repo_path)
    
    def repo_exists(self, instance: Any) -> bool:
        """
        Check if the repository for an instance exists locally.
        
        Args:
            instance: SWE-bench instance
            
        Returns:
            True if repository exists
        """
        repo_path = Path(self.get_repo_path(instance))
        return repo_path.exists() and repo_path.is_dir()
