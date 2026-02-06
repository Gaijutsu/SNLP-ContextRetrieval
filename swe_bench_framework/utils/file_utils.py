"""
File utilities for the SWE-bench comparison framework.

This module provides utility functions for file operations such as
reading, writing, and path manipulation.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, Iterator
import logging

logger = logging.getLogger(__name__)


def read_file(
    path: Union[str, Path],
    encoding: str = "utf-8",
    errors: str = "replace"
) -> str:
    """
    Read the contents of a file.
    
    Args:
        path: Path to the file
        encoding: File encoding
        errors: How to handle encoding errors
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
    """
    file_path = Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if not file_path.is_file():
        raise IOError(f"Path is not a file: {path}")
    
    try:
        with open(file_path, 'r', encoding=encoding, errors=errors) as f:
            return f.read()
    except Exception as e:
        raise IOError(f"Failed to read file {path}: {e}")


def read_file_lines(
    path: Union[str, Path],
    encoding: str = "utf-8",
    errors: str = "replace"
) -> List[str]:
    """
    Read a file as a list of lines.
    
    Args:
        path: Path to the file
        encoding: File encoding
        errors: How to handle encoding errors
        
    Returns:
        List of lines (without newline characters)
    """
    content = read_file(path, encoding, errors)
    return content.splitlines()


def write_file(
    path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    mkdir: bool = True
) -> None:
    """
    Write content to a file.
    
    Args:
        path: Path to the file
        content: Content to write
        encoding: File encoding
        mkdir: Whether to create parent directories
        
    Raises:
        IOError: If the file cannot be written
    """
    file_path = Path(path)
    
    if mkdir:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        logger.debug(f"Wrote file: {path}")
    except Exception as e:
        raise IOError(f"Failed to write file {path}: {e}")


def append_to_file(
    path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    mkdir: bool = True
) -> None:
    """
    Append content to a file.
    
    Args:
        path: Path to the file
        content: Content to append
        encoding: File encoding
        mkdir: Whether to create parent directories
    """
    file_path = Path(path)
    
    if mkdir:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'a', encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Failed to append to file {path}: {e}")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_extension(path: Union[str, Path]) -> str:
    """
    Get the file extension.
    
    Args:
        path: Path to the file
        
    Returns:
        File extension (including the dot), or empty string if no extension
    """
    return Path(path).suffix


def get_file_name(path: Union[str, Path], with_extension: bool = True) -> str:
    """
    Get the file name from a path.
    
    Args:
        path: Path to the file
        with_extension: Whether to include the extension
        
    Returns:
        File name
    """
    file_path = Path(path)
    if with_extension:
        return file_path.name
    else:
        return file_path.stem


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    exclude_dirs: Optional[List[str]] = None
) -> Iterator[Path]:
    """
    Find files matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        exclude_dirs: List of directory names to exclude
        
    Yields:
        Path objects for matching files
    """
    dir_path = Path(directory)
    exclude_dirs = exclude_dirs or []
    
    if recursive:
        for path in dir_path.rglob(pattern):
            if path.is_file():
                # Check if any parent is excluded
                if not any(part in exclude_dirs for part in path.parts):
                    yield path
    else:
        for path in dir_path.glob(pattern):
            if path.is_file():
                yield path


def find_files_by_extension(
    directory: Union[str, Path],
    extensions: List[str],
    recursive: bool = True,
    exclude_dirs: Optional[List[str]] = None
) -> Iterator[Path]:
    """
    Find files with specific extensions.
    
    Args:
        directory: Directory to search
        extensions: List of extensions (with or without dots)
        recursive: Whether to search recursively
        exclude_dirs: List of directory names to exclude
        
    Yields:
        Path objects for matching files
    """
    # Normalize extensions
    normalized_exts = [
        ext if ext.startswith('.') else f'.{ext}'
        for ext in extensions
    ]
    
    for file_path in find_files(directory, "*", recursive, exclude_dirs):
        if file_path.suffix in normalized_exts:
            yield file_path


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    mkdir: bool = True
) -> None:
    """
    Copy a file.
    
    Args:
        src: Source file path
        dst: Destination file path
        mkdir: Whether to create parent directories
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    
    if mkdir:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)
    logger.debug(f"Copied file: {src} -> {dst}")


def copy_directory(
    src: Union[str, Path],
    dst: Union[str, Path],
    ignore: Optional[List[str]] = None
) -> None:
    """
    Copy a directory recursively.
    
    Args:
        src: Source directory path
        dst: Destination directory path
        ignore: List of patterns to ignore
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    
    ignore_patterns = shutil.ignore_patterns(*(ignore or []))
    shutil.copytree(src_path, dst_path, ignore=ignore_patterns, dirs_exist_ok=True)
    logger.debug(f"Copied directory: {src} -> {dst}")


def delete_file(path: Union[str, Path], missing_ok: bool = True) -> None:
    """
    Delete a file.
    
    Args:
        path: Path to the file
        missing_ok: Whether to ignore if file doesn't exist
    """
    file_path = Path(path)
    
    if not file_path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"File not found: {path}")
    
    file_path.unlink()
    logger.debug(f"Deleted file: {path}")


def delete_directory(
    path: Union[str, Path],
    missing_ok: bool = True,
    recursive: bool = True
) -> None:
    """
    Delete a directory.
    
    Args:
        path: Path to the directory
        missing_ok: Whether to ignore if directory doesn't exist
        recursive: Whether to delete recursively
    """
    dir_path = Path(path)
    
    if not dir_path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(f"Directory not found: {path}")
    
    if recursive:
        shutil.rmtree(dir_path)
    else:
        dir_path.rmdir()
    
    logger.debug(f"Deleted directory: {path}")


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        path: Path to the file
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size


def get_file_line_count(path: Union[str, Path]) -> int:
    """
    Get the number of lines in a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Number of lines
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for _ in f)


def is_binary_file(path: Union[str, Path], sample_size: int = 8192) -> bool:
    """
    Check if a file is binary.
    
    Args:
        path: Path to the file
        sample_size: Number of bytes to sample
        
    Returns:
        True if the file is binary, False otherwise
    """
    try:
        with open(path, 'rb') as f:
            chunk = f.read(sample_size)
            if b'\x00' in chunk:
                return True
            # Try to decode as text
            chunk.decode('utf-8')
            return False
    except (UnicodeDecodeError, IOError):
        return True


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Get the relative path from a base directory.
    
    Args:
        path: Path to make relative
        base: Base directory
        
    Returns:
        Relative path
    """
    return Path(path).relative_to(base)


def join_paths(*paths: Union[str, Path]) -> Path:
    """
    Join multiple path components.
    
    Args:
        *paths: Path components
        
    Returns:
        Joined path
    """
    return Path(*paths)


def normalize_path(path: Union[str, Path]) -> Path:
    """
    Normalize a path (resolve symlinks, make absolute).
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path
    """
    return Path(path).resolve()
