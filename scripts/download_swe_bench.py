#!/usr/bin/env python3
"""
Script to download SWE-bench datasets.

This script downloads and caches SWE-bench datasets (Lite, Verified, Full)
for use with the comparison framework.

Usage:
    python scripts/download_swe_bench.py --dataset lite
    python scripts/download_swe_bench.py --dataset verified --output-dir ./datasets
    python scripts/download_swe_bench.py --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# SWE-bench dataset URLs
SWE_BENCH_URLS = {
    'lite': 'https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/assets/swe-bench-lite.json',
    'verified': 'https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/assets/swe-bench-verified.json',
    'full': 'https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/assets/swe-bench-full.json',
    'test': 'https://raw.githubusercontent.com/princeton-nlp/SWE-bench/main/swebench/assets/swe-bench-test.json',
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        desc: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urlretrieve(url, output_path, reporthook=t.update_to)
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    force: bool = False
) -> Optional[Path]:
    """
    Download a SWE-bench dataset.
    
    Args:
        dataset_name: Name of dataset (lite, verified, full, test)
        output_dir: Directory to save dataset
        force: Whether to overwrite existing files
        
    Returns:
        Path to downloaded file or None if failed
    """
    if dataset_name not in SWE_BENCH_URLS:
        logger.error(f"Unknown dataset: {dataset_name}")
        logger.info(f"Available datasets: {', '.join(SWE_BENCH_URLS.keys())}")
        return None
    
    url = SWE_BENCH_URLS[dataset_name]
    output_path = output_dir / f'swe-bench-{dataset_name}.json'
    
    # Check if already exists
    if output_path.exists() and not force:
        logger.info(f"Dataset already exists: {output_path}")
        logger.info("Use --force to re-download")
        return output_path
    
    # Download
    logger.info(f"Downloading SWE-bench {dataset_name}...")
    logger.info(f"URL: {url}")
    logger.info(f"Output: {output_path}")
    
    if download_file(url, output_path, desc=f"swe-bench-{dataset_name}"):
        logger.info(f"Successfully downloaded to {output_path}")
        
        # Validate JSON
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Dataset contains {len(data)} instances")
            return output_path
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in downloaded file: {e}")
            output_path.unlink()  # Delete invalid file
            return None
    else:
        return None


def download_all_datasets(output_dir: Path, force: bool = False) -> dict:
    """
    Download all SWE-bench datasets.
    
    Args:
        output_dir: Directory to save datasets
        force: Whether to overwrite existing files
        
    Returns:
        Dictionary mapping dataset names to paths
    """
    results = {}
    
    for dataset_name in SWE_BENCH_URLS.keys():
        path = download_dataset(dataset_name, output_dir, force)
        results[dataset_name] = path
    
    return results


def get_dataset_info(dataset_path: Path) -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_path: Path to dataset file
        
    Returns:
        Dictionary with dataset information
    """
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Extract statistics
        repos = set()
        for instance in data:
            repo = instance.get('repo', 'unknown')
            repos.add(repo)
        
        return {
            'total_instances': len(data),
            'unique_repos': len(repos),
            'repos': sorted(list(repos)),
        }
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        return {}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download SWE-bench datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download SWE-bench Lite
  python scripts/download_swe_bench.py --dataset lite
  
  # Download to specific directory
  python scripts/download_swe_bench.py --dataset verified --output-dir ./datasets
  
  # Download all datasets
  python scripts/download_swe_bench.py --all
  
  # Force re-download
  python scripts/download_swe_bench.py --dataset lite --force
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=list(SWE_BENCH_URLS.keys()),
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./datasets',
        help='Output directory for datasets'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Download all datasets'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if file exists'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show dataset information after download'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    if args.all:
        logger.info("Downloading all SWE-bench datasets...")
        results = download_all_datasets(output_dir, args.force)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Download Summary")
        print("=" * 60)
        for name, path in results.items():
            status = "✓" if path else "✗"
            print(f"{status} {name}: {path if path else 'FAILED'}")
    
    elif args.dataset:
        path = download_dataset(args.dataset, output_dir, args.force)
        
        if path and args.info:
            info = get_dataset_info(path)
            print("\n" + "=" * 60)
            print("Dataset Information")
            print("=" * 60)
            print(f"Name: {args.dataset}")
            print(f"Path: {path}")
            print(f"Total instances: {info.get('total_instances', 'N/A')}")
            print(f"Unique repositories: {info.get('unique_repos', 'N/A')}")
            
            repos = info.get('repos', [])
            if repos:
                print(f"\nRepositories ({len(repos)}):")
                for repo in repos[:10]:  # Show first 10
                    print(f"  - {repo}")
                if len(repos) > 10:
                    print(f"  ... and {len(repos) - 10} more")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
