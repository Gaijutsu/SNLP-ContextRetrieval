"""
Utilities for the SWE-bench comparison framework.

This module provides utility functions for file operations, token counting,
git operations, and other common tasks.
"""

from .token_utils import TokenCounter, count_tokens, truncate_to_tokens
from .file_utils import (
    read_file,
    write_file,
    ensure_dir,
    get_file_extension,
    find_files,
    copy_file,
)
from .git_utils import (
    clone_repository,
    checkout_commit,
    get_repo_name_from_url,
    is_git_repository,
)

__all__ = [
    # Token utilities
    "TokenCounter",
    "count_tokens",
    "truncate_to_tokens",
    # File utilities
    "read_file",
    "write_file",
    "ensure_dir",
    "get_file_extension",
    "find_files",
    "copy_file",
    # Git utilities
    "clone_repository",
    "checkout_commit",
    "get_repo_name_from_url",
    "is_git_repository",
]
