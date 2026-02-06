#!/usr/bin/env python3
"""
Script to pre-build RAG indexes for repositories.

This script builds searchable indexes for RAG-based methods,
speeding up subsequent experiments.

Usage:
    python scripts/build_indexes.py --repo-path /path/to/repo
    python scripts/build_indexes.py --repos-dir ./repos --all
    python scripts/build_indexes.py --dataset lite --output-dir ./indexes
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_index_for_repo(
    repo_path: Path,
    output_dir: Path,
    index_type: str = 'hybrid',
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force: bool = False
) -> Optional[Path]:
    """
    Build index for a single repository.
    
    Args:
        repo_path: Path to repository
        output_dir: Directory to save index
        index_type: Type of index (bm25, dense, hybrid)
        embedding_model: Embedding model for dense index
        chunk_size: Size of code chunks
        chunk_overlap: Overlap between chunks
        force: Whether to overwrite existing index
        
    Returns:
        Path to built index or None if failed
    """
    try:
        from swe_bench_framework.context_gatherers.index_builder import (
            BM25IndexBuilder,
            DenseIndexBuilder,
            HybridIndexBuilder
        )
        
        # Check if already exists
        index_output_dir = output_dir / repo_path.name / index_type
        index_file = index_output_dir / 'index.pkl'
        
        if index_file.exists() and not force:
            logger.info(f"Index already exists for {repo_path.name}: {index_file}")
            return index_file
        
        logger.info(f"Building {index_type} index for {repo_path.name}...")
        
        # Create appropriate builder
        if index_type == 'bm25':
            builder = BM25IndexBuilder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif index_type == 'dense':
            builder = DenseIndexBuilder(
                model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:  # hybrid
            builder = HybridIndexBuilder(
                model_name=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # Build index
        index_path = builder.build(repo_path, index_output_dir)
        
        # Get stats
        stats = builder.get_stats()
        logger.info(f"Built index for {repo_path.name}: {stats}")
        
        return index_path
        
    except Exception as e:
        logger.error(f"Failed to build index for {repo_path.name}: {e}")
        return None


def build_indexes_for_dataset(
    dataset_path: Path,
    repos_dir: Path,
    output_dir: Path,
    index_type: str = 'hybrid',
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force: bool = False,
    max_workers: int = 1
) -> dict:
    """
    Build indexes for all repositories in a dataset.
    
    Args:
        dataset_path: Path to dataset file
        repos_dir: Directory containing repositories
        output_dir: Directory to save indexes
        index_type: Type of index
        embedding_model: Embedding model
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
        force: Force rebuild
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary mapping repo names to index paths
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Get unique repositories
    repos = set()
    for instance in dataset:
        repo = instance.get('repo', '')
        if repo:
            repos.add(repo)
    
    logger.info(f"Found {len(repos)} unique repositories in dataset")
    
    results = {}
    
    if max_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for repo in repos:
                repo_path = repos_dir / repo.replace('/', '_')
                if repo_path.exists():
                    future = executor.submit(
                        build_index_for_repo,
                        repo_path,
                        output_dir,
                        index_type,
                        embedding_model,
                        chunk_size,
                        chunk_overlap,
                        force
                    )
                    futures[future] = repo
                else:
                    logger.warning(f"Repository not found: {repo_path}")
                    results[repo] = None
            
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    index_path = future.result()
                    results[repo] = index_path
                except Exception as e:
                    logger.error(f"Failed to build index for {repo}: {e}")
                    results[repo] = None
    else:
        # Sequential execution
        for repo in repos:
            repo_path = repos_dir / repo.replace('/', '_')
            if repo_path.exists():
                index_path = build_index_for_repo(
                    repo_path,
                    output_dir,
                    index_type,
                    embedding_model,
                    chunk_size,
                    chunk_overlap,
                    force
                )
                results[repo] = index_path
            else:
                logger.warning(f"Repository not found: {repo_path}")
                results[repo] = None
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Build RAG indexes for repositories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for single repository
  python scripts/build_indexes.py --repo-path /path/to/django
  
  # Build hybrid index
  python scripts/build_indexes.py -r /path/to/repo --index-type hybrid
  
  # Build dense index with custom model
  python scripts/build_indexes.py -r /path/to/repo --index-type dense \\
      --embedding-model sentence-transformers/all-mpnet-base-v2
  
  # Build indexes for all repos in dataset
  python scripts/build_indexes.py --dataset ./datasets/swe-bench-lite.json \\
      --repos-dir ./repos
  
  # Build all index types
  python scripts/build_indexes.py --dataset ./datasets/swe-bench-lite.json \\
      --repos-dir ./repos --all-types
        """
    )
    
    parser.add_argument(
        '--repo-path', '-r',
        type=str,
        help='Path to single repository'
    )
    
    parser.add_argument(
        '--repos-dir',
        type=str,
        default='./repos',
        help='Directory containing repositories'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./indexes',
        help='Output directory for indexes'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Dataset file to get repository list from'
    )
    
    parser.add_argument(
        '--index-type', '-t',
        type=str,
        choices=['bm25', 'dense', 'hybrid'],
        default='hybrid',
        help='Type of index to build'
    )
    
    parser.add_argument(
        '--all-types', '-a',
        action='store_true',
        help='Build all index types (bm25, dense, hybrid)'
    )
    
    parser.add_argument(
        '--embedding-model', '-e',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model for dense index'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Size of code chunks'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=200,
        help='Overlap between chunks'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force rebuild even if index exists'
    )
    
    parser.add_argument(
        '--max-workers', '-j',
        type=int,
        default=1,
        help='Number of parallel workers'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which index types to build
    index_types = ['bm25', 'dense', 'hybrid'] if args.all_types else [args.index_type]
    
    # Build indexes
    if args.repo_path:
        # Single repository
        repo_path = Path(args.repo_path)
        if not repo_path.exists():
            logger.error(f"Repository not found: {repo_path}")
            sys.exit(1)
        
        for index_type in index_types:
            build_index_for_repo(
                repo_path,
                output_dir,
                index_type,
                args.embedding_model,
                args.chunk_size,
                args.chunk_overlap,
                args.force
            )
    
    elif args.dataset:
        # Dataset with multiple repositories
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            sys.exit(1)
        
        repos_dir = Path(args.repos_dir)
        
        for index_type in index_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Building {index_type} indexes")
            logger.info('='*60)
            
            results = build_indexes_for_dataset(
                dataset_path,
                repos_dir,
                output_dir,
                index_type,
                args.embedding_model,
                args.chunk_size,
                args.chunk_overlap,
                args.force,
                args.max_workers
            )
            
            # Print summary
            successful = sum(1 for v in results.values() if v is not None)
            total = len(results)
            
            print(f"\n{index_type.upper()} Index Summary:")
            print(f"  Successful: {successful}/{total}")
            print(f"  Failed: {total - successful}/{total}")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
