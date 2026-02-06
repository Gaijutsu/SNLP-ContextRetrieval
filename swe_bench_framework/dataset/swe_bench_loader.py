"""
SWE-bench specific loader for the comparison framework.

This module provides loaders for the official SWE-bench datasets
from HuggingFace, including Lite, Verified, and Full variants.
"""

from typing import Any, Dict, Iterator, List, Optional
from pathlib import Path

from .loader import DatasetLoader, SWEInstance


class SWEBenchLoader(DatasetLoader):
    """
    Loader for official SWE-bench datasets from HuggingFace.
    
    Supports:
    - SWE-bench Lite (300 instances)
    - SWE-bench Verified (500 instances)
    - SWE-bench Full (2,294 instances)
    - SWE-bench Multimodal
    - SWE-bench Multilingual
    
    Example:
        loader = SWEBenchLoader({
            'dataset_name': 'princeton-nlp/SWE-bench_Lite',
            'split': 'test',
            'cache_dir': './cache'
        })
        loader.initialize()
        
        # Load all instances
        instances = loader.load_all()
        print(f"Loaded {len(instances)} instances")
        
        # Or iterate
        for instance in loader.load():
            print(instance.instance_id)
    """
    
    # Dataset name mappings
    DATASET_NAMES = {
        'lite': 'princeton-nlp/SWE-bench_Lite',
        'verified': 'princeton-nlp/SWE-bench_Verified',
        'full': 'princeton-nlp/SWE-bench',
        'multimodal': 'princeton-nlp/SWE-bench_Multimodal',
        'multilingual': 'princeton-nlp/SWE-bench_Multilingual'
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-bench loader.
        
        Args:
            config: Configuration dictionary
                - dataset_name: Dataset name or alias ('lite', 'verified', 'full', etc.)
                - split: Dataset split (default: 'test')
                - cache_dir: Cache directory for datasets
                - filter: Filter criteria (repos, max_instances, etc.)
        """
        super().__init__(config)
        
        # Resolve dataset name
        dataset_name = config.get('dataset_name', 'lite')
        if dataset_name.lower() in self.DATASET_NAMES:
            self.dataset_name = self.DATASET_NAMES[dataset_name.lower()]
        else:
            self.dataset_name = dataset_name
        
        self._dataset = None
    
    def initialize(self) -> None:
        """Initialize by loading the dataset from HuggingFace."""
        import logging
        logger = logging.getLogger(__name__)
        
        super().initialize()
        
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading dataset: {self.dataset_name} (split: {self.split})")
            if self.cache_dir:
                logger.info(f"Using cache directory: {self.cache_dir}")
            
            # Always use streaming mode for memory efficiency
            # Even with instance_ids, we'll filter during iteration
            logger.info("Loading dataset in streaming mode (memory efficient)")
            self._dataset = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=True
            )
            logger.info("Dataset loaded in streaming mode")
            
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    
    def load(self) -> Iterator[SWEInstance]:
        """
        Load and yield instances from the dataset.
        
        Yields:
            SWEInstance objects
        """
        if not self._initialized:
            raise RuntimeError("Loader not initialized. Call initialize() first.")
        
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded")
        
        for item in self._dataset:
            instance = self._convert_to_instance(item)
            
            # Apply filters
            if self._passes_filter(instance):
                yield instance
    
    def load_all(self) -> List[SWEInstance]:
        """
        Load all instances from the dataset.
        
        Returns:
            List of SWEInstance objects
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check filters
        instance_ids = self.filter.get('instance_ids')
        max_instances = self.filter.get('max_instances')
        
        if instance_ids:
            # Only load matching instances
            logger.info(f"Loading only {len(instance_ids)} specified instances from streaming dataset...")
            instances = []
            target_ids = set(instance_ids)
            count = 0
            
            # Use iter() for streaming datasets
            for item in iter(self._dataset):
                count += 1
                instance_id = item.get('instance_id', '')
                if instance_id in target_ids:
                    instance = self._convert_to_instance(item)
                    instances.append(instance)
                    logger.info(f"Found instance: {instance_id}")
                    
                    # Stop early if we found all
                    if len(instances) == len(target_ids):
                        logger.info(f"Found all {len(target_ids)} instances after checking {count} records")
                        break
                
                # Safety limit - don't scan forever
                if count > 10000:
                    logger.warning(f"Scanned {count} records but only found {len(instances)} matches. Stopping.")
                    break
            
            logger.info(f"Loaded {len(instances)} matching instances (scanned {count} total)")
            return instances
        
        elif max_instances:
            # Load only first N instances
            logger.info(f"Loading first {max_instances} instances from streaming dataset...")
            instances = []
            count = 0
            
            for item in iter(self._dataset):
                count += 1
                instance = self._convert_to_instance(item)
                
                # Apply other filters
                if self._passes_filter(instance):
                    instances.append(instance)
                    logger.debug(f"Loaded instance: {instance.instance_id}")
                    
                    # Stop when we have enough
                    if len(instances) >= max_instances:
                        logger.info(f"Loaded {len(instances)} instances (scanned {count} total)")
                        break
                
                # Safety limit
                if count > 10000:
                    logger.warning(f"Scanned {count} records but only found {len(instances)} matches. Stopping.")
                    break
            
            logger.info(f"Loaded {len(instances)} instances (scanned {count} total)")
            return instances
        
        else:
            # Load all (may be memory intensive)
            return list(self.load())
    
    def _convert_to_instance(self, item: Dict[str, Any]) -> SWEInstance:
        """
        Convert a dataset item to SWEInstance.
        
        Args:
            item: Dataset item from HuggingFace
            
        Returns:
            SWEInstance
        """
        # Extract test lists
        failed_tests = self._parse_tests(item.get('FAIL_TO_PASS', ''))
        passed_tests = self._parse_tests(item.get('PASS_TO_PASS', ''))
        
        # Extract modified files and methods
        modified_files = self._extract_modified_files(item.get('patch', ''))
        modified_methods = self._extract_modified_methods(item.get('patch', ''))
        
        return SWEInstance(
            instance_id=item.get('instance_id', ''),
            repo=item.get('repo', ''),
            base_commit=item.get('base_commit', ''),
            problem_statement=item.get('problem_statement', ''),
            hints_text=item.get('hints_text'),
            test_patch=item.get('test_patch', ''),
            patch=item.get('patch', ''),
            failed_tests=failed_tests,
            passed_tests=passed_tests,
            modified_files=modified_files,
            modified_methods=modified_methods,
            metadata={
                'version': item.get('version'),
                'issue_url': item.get('issue_url'),
                'pr_url': item.get('pr_url'),
                'created_at': item.get('created_at')
            }
        )
    
    def _parse_tests(self, test_string: str) -> List[str]:
        """
        Parse test string into list of test names.
        
        Args:
            test_string: String containing test names (may be newline-separated)
            
        Returns:
            List of test names
        """
        if not test_string:
            return []
        
        # Handle both list and string formats
        if isinstance(test_string, list):
            return test_string
        
        # Split by newline and filter empty lines
        tests = [t.strip() for t in test_string.split('\n') if t.strip()]
        return tests
    
    def _extract_modified_files(self, patch: str) -> List[str]:
        """
        Extract list of modified files from patch.
        
        Args:
            patch: Unified diff patch
            
        Returns:
            List of file paths
        """
        files = []
        
        for line in patch.split('\n'):
            if line.startswith('diff --git'):
                # Extract file path from diff header
                parts = line.split()
                if len(parts) >= 4:
                    # Format: diff --git a/path b/path
                    file_path = parts[2][2:]  # Remove 'a/' prefix
                    if file_path not in files:
                        files.append(file_path)
            elif line.startswith('--- a/'):
                # Alternative: extract from --- line
                file_path = line[6:].split('\t')[0]
                if file_path and file_path not in files:
                    files.append(file_path)
        
        return files
    
    def _extract_modified_methods(self, patch: str) -> List[str]:
        """
        Extract list of modified methods/functions from patch.
        
        Args:
            patch: Unified diff patch
            
        Returns:
            List of method/function names
        """
        methods = []
        
        # Simple heuristic: look for function/class definitions in context
        import re
        
        pattern = r'^[\+\-]\s*(def|class)\s+(\w+)'
        
        for line in patch.split('\n'):
            match = re.match(pattern, line)
            if match:
                method_name = match.group(2)
                if method_name not in methods:
                    methods.append(method_name)
        
        return methods
    
    def _passes_filter(self, instance: SWEInstance) -> bool:
        """
        Check if instance passes filter criteria.
        
        Args:
            instance: Instance to check
            
        Returns:
            True if instance passes filter
        """
        # Repository filter
        if 'repos' in self.filter:
            repos = self.filter['repos']
            if isinstance(repos, str):
                repos = [repos]
            
            repo_match = (
                instance.repo in repos or
                instance.repo_name in repos or
                instance.repo_owner in repos
            )
            if not repo_match:
                return False
        
        # Instance ID filter
        if 'instance_ids' in self.filter:
            ids = self.filter['instance_ids']
            if isinstance(ids, str):
                ids = [ids]
            if instance.instance_id not in ids:
                return False
        
        # Exclude filter
        if 'exclude_ids' in self.filter:
            exclude_ids = self.filter['exclude_ids']
            if isinstance(exclude_ids, str):
                exclude_ids = [exclude_ids]
            if instance.instance_id in exclude_ids:
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = super().get_stats()
        
        if self._dataset is not None:
            stats['num_raw_instances'] = len(self._dataset)
            
            # Count by repository
            repo_counts = {}
            for item in self._dataset:
                repo = item.get('repo', 'unknown')
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
            
            stats['repo_distribution'] = repo_counts
        
        return stats


class SWEBenchLiteLoader(SWEBenchLoader):
    """Convenience loader for SWE-bench Lite."""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['dataset_name'] = 'lite'
        super().__init__(config)


class SWEBenchVerifiedLoader(SWEBenchLoader):
    """Convenience loader for SWE-bench Verified."""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['dataset_name'] = 'verified'
        super().__init__(config)


class SWEBenchFullLoader(SWEBenchLoader):
    """Convenience loader for SWE-bench Full."""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        config['dataset_name'] = 'full'
        super().__init__(config)


class SWEBenchPredictionsLoader:
    """
    Loader for SWE-bench prediction files.
    
    Loads prediction files in the format required by SWE-bench evaluation.
    """
    
    def __init__(self, predictions_path: str):
        """
        Initialize the predictions loader.
        
        Args:
            predictions_path: Path to predictions file (JSON or JSONL)
        """
        self.predictions_path = Path(predictions_path)
    
    def load(self) -> Dict[str, Dict[str, Any]]:
        """
        Load predictions as a dictionary.
        
        Returns:
            Dictionary mapping instance_id to prediction data
        """
        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_path}")
        
        predictions = {}
        
        if self.predictions_path.suffix == '.jsonl':
            import json
            with open(self.predictions_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    instance_id = data.get('instance_id')
                    if instance_id:
                        predictions[instance_id] = data
        
        elif self.predictions_path.suffix == '.json':
            import json
            with open(self.predictions_path, 'r') as f:
                data = json.load(f)
            
            # Handle both dict and list formats
            if isinstance(data, dict):
                predictions = data
            elif isinstance(data, list):
                for item in data:
                    instance_id = item.get('instance_id')
                    if instance_id:
                        predictions[instance_id] = item
        
        else:
            raise ValueError(f"Unsupported file format: {self.predictions_path.suffix}")
        
        return predictions
    
    def get_prediction(self, instance_id: str) -> Optional[str]:
        """
        Get prediction for a specific instance.
        
        Args:
            instance_id: Instance ID
            
        Returns:
            Model patch or None
        """
        predictions = self.load()
        prediction = predictions.get(instance_id, {})
        return prediction.get('model_patch')


class SWEBenchPredictionsWriter:
    """
    Writer for SWE-bench prediction files.
    
    Writes predictions in the format required by SWE-bench evaluation.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize the predictions writer.
        
        Args:
            output_path: Path to write predictions
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._predictions: List[Dict[str, Any]] = []
    
    def add_prediction(
        self,
        instance_id: str,
        model_patch: str,
        model_name_or_path: str = "unknown"
    ) -> None:
        """
        Add a prediction.
        
        Args:
            instance_id: Instance ID
            model_patch: Generated patch
            model_name_or_path: Model identifier
        """
        self._predictions.append({
            'instance_id': instance_id,
            'model_patch': model_patch,
            'model_name_or_path': model_name_or_path
        })
    
    def write(self, format: str = 'jsonl') -> None:
        """
        Write predictions to file.
        
        Args:
            format: Output format ('jsonl' or 'json')
        """
        import json
        
        if format == 'jsonl':
            with open(self.output_path, 'w') as f:
                for pred in self._predictions:
                    f.write(json.dumps(pred) + '\n')
        
        elif format == 'json':
            with open(self.output_path, 'w') as f:
                json.dump(self._predictions, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear(self) -> None:
        """Clear all predictions."""
        self._predictions = []
