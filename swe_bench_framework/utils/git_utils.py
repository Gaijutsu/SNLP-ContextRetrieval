"""
Git utilities for the SWE-bench comparison framework.

This module provides utility functions for git operations such as
cloning repositories, checking out commits, and inspecting repository state.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
from urllib.parse import urlparse
import logging

from ..core.exceptions import RepositoryError

logger = logging.getLogger(__name__)


def run_git_command(
    args: List[str],
    cwd: Optional[Union[str, Path]] = None,
    check: bool = True,
    capture_output: bool = True,
    timeout: Optional[float] = None
) -> Tuple[int, str, str]:
    """
    Run a git command.
    
    Args:
        args: Git command arguments
        cwd: Working directory
        check: Whether to raise an error on non-zero exit
        capture_output: Whether to capture stdout/stderr
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (returncode, stdout, stderr)
        
    Raises:
        RepositoryError: If the command fails and check=True
    """
    cmd = ['git'] + args
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=False,
            timeout=timeout,
        )
        
        if check and result.returncode != 0:
            raise RepositoryError(
                f"Git command failed: {' '.join(cmd)}",
                details={
                    'command': ' '.join(cmd),
                    'cwd': str(cwd),
                    'returncode': result.returncode,
                    'stderr': result.stderr,
                }
            )
        
        return result.returncode, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired as e:
        raise RepositoryError(
            f"Git command timed out after {timeout}s",
            details={
                'command': ' '.join(cmd),
                'timeout': timeout,
            }
        )
    except Exception as e:
        raise RepositoryError(
            f"Failed to run git command: {e}",
            details={'command': ' '.join(cmd), 'error': str(e)}
        )


def clone_repository(
    repo_url: str,
    target_path: Union[str, Path],
    branch: Optional[str] = None,
    depth: Optional[int] = None,
    timeout: float = 300.0
) -> Path:
    """
    Clone a git repository.
    
    Args:
        repo_url: URL of the repository
        target_path: Path where to clone
        branch: Branch to clone (None for default)
        depth: Clone depth (None for full clone)
        timeout: Timeout in seconds
        
    Returns:
        Path to the cloned repository
        
    Raises:
        RepositoryError: If cloning fails
    """
    target = Path(target_path)
    
    if target.exists():
        raise RepositoryError(
            f"Target path already exists: {target}",
            details={'target': str(target)}
        )
    
    args = ['clone']
    
    if branch:
        args.extend(['--branch', branch])
    
    if depth:
        args.extend(['--depth', str(depth)])
    
    args.extend([repo_url, str(target)])
    
    logger.info(f"Cloning repository: {repo_url}")
    run_git_command(args, timeout=timeout)
    
    logger.info(f"Cloned to: {target}")
    return target


def checkout_commit(
    repo_path: Union[str, Path],
    commit_hash: str,
    create_branch: bool = False,
    force: bool = False
) -> None:
    """
    Checkout a specific commit.
    
    Args:
        repo_path: Path to the repository
        commit_hash: Commit hash to checkout
        create_branch: Whether to create a new branch
        force: Whether to force checkout
        
    Raises:
        RepositoryError: If checkout fails
    """
    args = ['checkout']
    
    if force:
        args.append('--force')
    
    if create_branch:
        args.extend(['-b', f'temp-branch-{commit_hash[:8]}'])
    
    args.append(commit_hash)
    
    logger.info(f"Checking out commit: {commit_hash}")
    run_git_command(args, cwd=repo_path)


def get_current_commit(repo_path: Union[str, Path]) -> str:
    """
    Get the current commit hash.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Current commit hash
    """
    _, stdout, _ = run_git_command(
        ['rev-parse', 'HEAD'],
        cwd=repo_path
    )
    return stdout.strip()


def get_commit_info(
    repo_path: Union[str, Path],
    commit_hash: Optional[str] = None
) -> dict:
    """
    Get information about a commit.
    
    Args:
        repo_path: Path to the repository
        commit_hash: Commit hash (None for current)
        
    Returns:
        Dictionary with commit information
    """
    if commit_hash is None:
        commit_hash = 'HEAD'
    
    # Get commit message
    _, message, _ = run_git_command(
        ['log', '-1', '--format=%B', commit_hash],
        cwd=repo_path
    )
    
    # Get author
    _, author, _ = run_git_command(
        ['log', '-1', '--format=%an <%ae>', commit_hash],
        cwd=repo_path
    )
    
    # Get date
    _, date, _ = run_git_command(
        ['log', '-1', '--format=%ai', commit_hash],
        cwd=repo_path
    )
    
    return {
        'hash': commit_hash if commit_hash != 'HEAD' else get_current_commit(repo_path),
        'message': message.strip(),
        'author': author.strip(),
        'date': date.strip(),
    }


def is_git_repository(path: Union[str, Path]) -> bool:
    """
    Check if a path is a git repository.
    
    Args:
        path: Path to check
        
    Returns:
        True if the path is a git repository
    """
    git_dir = Path(path) / '.git'
    if git_dir.exists():
        return True
    
    try:
        run_git_command(['rev-parse', '--git-dir'], cwd=path, check=False)
        return True
    except RepositoryError:
        return False


def get_repo_name_from_url(repo_url: str) -> str:
    """
    Extract repository name from URL.
    
    Args:
        repo_url: Repository URL
        
    Returns:
        Repository name (e.g., "owner/repo")
    """
    # Handle SSH URLs (git@github.com:owner/repo.git)
    if repo_url.startswith('git@'):
        match = re.search(r':([^/]+/[^/]+?)(?:\.git)?$', repo_url)
        if match:
            return match.group(1)
    
    # Handle HTTPS URLs
    parsed = urlparse(repo_url)
    path = parsed.path.strip('/')
    
    # Remove .git suffix
    if path.endswith('.git'):
        path = path[:-4]
    
    return path


def get_repo_root(path: Union[str, Path]) -> Path:
    """
    Get the root directory of a git repository.
    
    Args:
        path: Path within the repository
        
    Returns:
        Path to the repository root
    """
    _, stdout, _ = run_git_command(
        ['rev-parse', '--show-toplevel'],
        cwd=path
    )
    return Path(stdout.strip())


def get_changed_files(
    repo_path: Union[str, Path],
    commit_hash: Optional[str] = None
) -> List[str]:
    """
    Get list of files changed in a commit.
    
    Args:
        repo_path: Path to the repository
        commit_hash: Commit hash (None for uncommitted changes)
        
    Returns:
        List of changed file paths
    """
    if commit_hash:
        _, stdout, _ = run_git_command(
            ['diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash],
            cwd=repo_path
        )
    else:
        _, stdout, _ = run_git_command(
            ['diff', '--name-only'],
            cwd=repo_path
        )
    
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def get_file_content_at_commit(
    repo_path: Union[str, Path],
    file_path: str,
    commit_hash: str
) -> str:
    """
    Get file content at a specific commit.
    
    Args:
        repo_path: Path to the repository
        file_path: Path to the file (relative to repo root)
        commit_hash: Commit hash
        
    Returns:
        File content
    """
    _, stdout, _ = run_git_command(
        ['show', f'{commit_hash}:{file_path}'],
        cwd=repo_path
    )
    return stdout


def apply_patch(
    repo_path: Union[str, Path],
    patch_content: str,
    check: bool = False,
    reverse: bool = False
) -> bool:
    """
    Apply a patch to the repository.
    
    Args:
        repo_path: Path to the repository
        patch_content: Patch content (unified diff)
        check: Whether to check if patch can be applied without applying
        reverse: Whether to apply patch in reverse
        
    Returns:
        True if patch was applied successfully
    """
    args = ['apply']
    
    if check:
        args.append('--check')
    
    if reverse:
        args.append('--reverse')
    
    args.append('-')  # Read from stdin
    
    try:
        result = subprocess.run(
            ['git'] + args,
            cwd=repo_path,
            input=patch_content,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to apply patch: {e}")
        return False


def reset_repository(
    repo_path: Union[str, Path],
    hard: bool = True
) -> None:
    """
    Reset repository to clean state.
    
    Args:
        repo_path: Path to the repository
        hard: Whether to do a hard reset
    """
    args = ['reset']
    
    if hard:
        args.append('--hard')
    
    args.append('HEAD')
    
    run_git_command(args, cwd=repo_path)
    
    # Clean untracked files
    if hard:
        run_git_command(['clean', '-fd'], cwd=repo_path, check=False)


def get_remote_url(repo_path: Union[str, Path]) -> Optional[str]:
    """
    Get the remote URL of a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Remote URL or None if not set
    """
    try:
        _, stdout, _ = run_git_command(
            ['remote', 'get-url', 'origin'],
            cwd=repo_path,
            check=False
        )
        return stdout.strip() or None
    except RepositoryError:
        return None


def list_branches(
    repo_path: Union[str, Path],
    remote: bool = False
) -> List[str]:
    """
    List branches in the repository.
    
    Args:
        repo_path: Path to the repository
        remote: Whether to list remote branches
        
    Returns:
        List of branch names
    """
    args = ['branch', '--format=%(refname:short)']
    
    if remote:
        args.append('-r')
    
    _, stdout, _ = run_git_command(args, cwd=repo_path)
    
    return [line.strip() for line in stdout.splitlines() if line.strip()]


def create_worktree(
    repo_path: Union[str, Path],
    worktree_path: Union[str, Path],
    commit_hash: str
) -> Path:
    """
    Create a git worktree for a specific commit.
    
    Args:
        repo_path: Path to the main repository
        worktree_path: Path for the new worktree
        commit_hash: Commit hash to checkout
        
    Returns:
        Path to the worktree
    """
    target = Path(worktree_path)
    
    run_git_command(
        ['worktree', 'add', '-d', str(target), commit_hash],
        cwd=repo_path
    )
    
    logger.info(f"Created worktree at {target} for commit {commit_hash}")
    return target


def remove_worktree(
    repo_path: Union[str, Path],
    worktree_path: Union[str, Path]
) -> None:
    """
    Remove a git worktree.
    
    Args:
        repo_path: Path to the main repository
        worktree_path: Path to the worktree
    """
    run_git_command(
        ['worktree', 'remove', str(worktree_path)],
        cwd=repo_path,
        check=False
    )
    
    logger.info(f"Removed worktree at {worktree_path}")


# Type hint for Union
from typing import Union
