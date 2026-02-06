"""
Dataset loader for the SWE-bench comparison framework.

This module provides base classes and utilities for loading SWE-bench
datasets in various formats (Lite, Verified, Full).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set
from pathlib import Path


@dataclass
class SWEInstance:
    """
    Represents a SWE-bench instance.
    
    Attributes:
        instance_id: Unique identifier (format: "owner__repo-pr_number")
        repo: Repository name (format: "owner/repo")
        base_commit: Git commit hash to checkout
        problem_statement: Issue description
        hints_text: Optional hints for solving the issue
        test_patch: Patch containing test cases
        patch: Gold/reference patch (the solution)
        failed_tests: List of FAIL_TO_PASS tests (fail before, pass after fix)
        passed_tests: List of PASS_TO_PASS tests (must still pass after fix)
        modified_files: List of files modified by the gold patch
        modified_methods: List of methods/functions modified
        metadata: Additional metadata
    """
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str] = None
    test_patch: str = ""
    patch: str = ""
    failed_tests: List[str] = field(default_factory=list)
    passed_tests: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    modified_methods: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate instance after initialization."""
        # Ensure lists are not None
        if self.failed_tests is None:
            self.failed_tests = []
        if self.passed_tests is None:
            self.passed_tests = []
        if self.modified_files is None:
            self.modified_files = []
        if self.modified_methods is None:
            self.modified_methods = []
    
    @property
    def repo_owner(self) -> str:
        """Get repository owner."""
        return self.repo.split('/')[0] if '/' in self.repo else ''
    
    @property
    def repo_name(self) -> str:
        """Get repository name."""
        return self.repo.split('/')[1] if '/' in self.repo else self.repo
    
    def get_gold_context(self) -> Dict[str, List[str]]:
        """Get ground truth context for evaluation."""
        return {
            'files': self.modified_files,
            'methods': self.modified_methods
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'instance_id': self.instance_id,
            'repo': self.repo,
            'base_commit': self.base_commit,
            'problem_statement': self.problem_statement,
            'hints_text': self.hints_text,
            'test_patch': self.test_patch,
            'patch': self.patch,
            'failed_tests': self.failed_tests,
            'passed_tests': self.passed_tests,
            'modified_files': self.modified_files,
            'modified_methods': self.modified_methods,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SWEInstance':
        """Create SWEInstance from dictionary."""
        # Extract fields
        instance_data = {
            'instance_id': data.get('instance_id', ''),
            'repo': data.get('repo', ''),
            'base_commit': data.get('base_commit', ''),
            'problem_statement': data.get('problem_statement', ''),
            'hints_text': data.get('hints_text'),
            'test_patch': data.get('test_patch', ''),
            'patch': data.get('patch', ''),
            'failed_tests': data.get('failed_tests', []),
            'passed_tests': data.get('passed_tests', []),
            'modified_files': data.get('modified_files', []),
            'modified_methods': data.get('modified_methods', []),
            'metadata': {k: v for k, v in data.items() 
                        if k not in ['instance_id', 'repo', 'base_commit',
                                   'problem_statement', 'hints_text', 'test_patch',
                                   'patch', 'failed_tests', 'passed_tests',
                                   'modified_files', 'modified_methods']}
        }
        
        return cls(**instance_data)


class DatasetLoader(ABC):
    """
    Abstract base class for dataset loaders.
    
    This class defines the interface for loading SWE-bench datasets
    from various sources (HuggingFace, local files, etc.).
    
    Example:
        loader = SWEBenchLoader(config)
        loader.initialize()
        
        for instance in loader.load():
            print(f"Processing {instance.instance_id}")
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dataset loader.
        
        Args:
            config: Configuration dictionary
                - dataset_name: Name of the dataset
                - split: Dataset split (train/test/val)
                - cache_dir: Directory for caching
                - filter: Filter criteria
        """
        self.config = config
        self.dataset_name = config.get('dataset_name', 'swe-bench-lite')
        self.split = config.get('split', 'test')
        self.cache_dir = config.get('cache_dir', None)
        self.filter = config.get('filter', {})
        
        self._initialized = False
        self._instances: List[SWEInstance] = []
    
    @abstractmethod
    def load(self) -> Iterator[SWEInstance]:
        """
        Load and yield instances from the dataset.
        
        Yields:
            SWEInstance objects
        """
        pass
    
    @abstractmethod
    def load_all(self) -> List[SWEInstance]:
        """
        Load all instances from the dataset.
        
        Returns:
            List of SWEInstance objects
        """
        pass
    
    def initialize(self) -> None:
        """Initialize the loader."""
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self._initialized = False
        self._instances = []
    
    def filter_instances(
        self,
        instances: List[SWEInstance]
    ) -> List[SWEInstance]:
        """
        Filter instances based on configuration.
        
        Args:
            instances: List of instances to filter
            
        Returns:
            Filtered list of instances
        """
        filtered = instances
        
        # Filter by repository
        if 'repos' in self.filter:
            repos = set(self.filter['repos'])
            filtered = [i for i in filtered if i.repo in repos or 
                       i.repo_name in repos or i.repo_owner in repos]
        
        # Filter by maximum instances
        if 'max_instances' in self.filter:
            max_instances = self.filter['max_instances']
            filtered = filtered[:max_instances]
        
        # Filter by instance IDs
        if 'instance_ids' in self.filter:
            ids = set(self.filter['instance_ids'])
            filtered = [i for i in filtered if i.instance_id in ids]
        
        # Filter by exclude IDs
        if 'exclude_ids' in self.filter:
            exclude_ids = set(self.filter['exclude_ids'])
            filtered = [i for i in filtered if i.instance_id not in exclude_ids]
        
        return filtered
    
    def with_filter(
        self,
        instance_ids: Optional[List[str]] = None,
        repos: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None,
        max_instances: Optional[int] = None
    ) -> 'DatasetLoader':
        """
        Create a filtered copy of this loader with additional filter criteria.
        
        Args:
            instance_ids: Only include these instance IDs
            repos: Only include instances from these repositories
            exclude_ids: Exclude these instance IDs
            max_instances: Maximum number of instances to include
            
        Returns:
            A new DatasetLoader with the filter applied
        """
        # Create new config with updated filter
        new_config = dict(self.config)
        new_filter = dict(self.filter)
        
        if instance_ids is not None:
            new_filter['instance_ids'] = instance_ids
        if repos is not None:
            new_filter['repos'] = repos
        if exclude_ids is not None:
            new_filter['exclude_ids'] = exclude_ids
        if max_instances is not None:
            new_filter['max_instances'] = max_instances
        
        new_config['filter'] = new_filter
        
        # Create new instance of the same class
        new_loader = self.__class__(new_config)
        return new_loader
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'dataset_name': self.dataset_name,
            'split': self.split,
            'initialized': self._initialized,
            'num_instances': len(self._instances),
            'filter': self.filter
        }


class LocalDatasetLoader(DatasetLoader):
    """
    Loader for local dataset files (JSON, JSONL).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local dataset loader.
        
        Args:
            config: Configuration with:
                - file_path: Path to the dataset file
                - format: File format ('json', 'jsonl')
        """
        super().__init__(config)
        self.file_path = config.get('file_path')
        self.format = config.get('format', 'jsonl')
    
    def load(self) -> Iterator[SWEInstance]:
        """Load instances from local file."""
        if not self.file_path:
            raise ValueError("file_path not specified")
        
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        if self.format == 'jsonl':
            yield from self._load_jsonl(path)
        elif self.format == 'json':
            yield from self._load_json(path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def load_all(self) -> List[SWEInstance]:
        """Load all instances from local file."""
        return list(self.load())
    
    def _load_jsonl(self, path: Path) -> Iterator[SWEInstance]:
        """Load from JSONL file."""
        import json
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                instance = self._create_instance(data)
                if instance:
                    yield instance
    
    def _load_json(self, path: Path) -> Iterator[SWEInstance]:
        """Load from JSON file."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, list):
            for item in data:
                instance = self._create_instance(item)
                if instance:
                    yield instance
        elif isinstance(data, dict):
            for item_id, item_data in data.items():
                if 'instance_id' not in item_data:
                    item_data['instance_id'] = item_id
                instance = self._create_instance(item_data)
                if instance:
                    yield instance
    
    def _create_instance(self, data: Dict[str, Any]) -> Optional[SWEInstance]:
        """Create SWEInstance from data dictionary."""
        try:
            return SWEInstance.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to create instance: {e}")
            return None


class DatasetSplitter:
    """
    Utility for splitting datasets into train/validation/test sets.
    
    Note: SWE-bench is primarily for evaluation, but splits can be
    created for experimentation purposes.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the splitter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
    
    def split(
        self,
        instances: List[SWEInstance],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by_repo: bool = True
    ) -> Dict[str, List[SWEInstance]]:
        """
        Split instances into train/validation/test sets.
        
        Args:
            instances: List of instances to split
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            stratify_by_repo: Whether to stratify by repository
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        import random
        random.seed(self.seed)
        
        if stratify_by_repo:
            # Group by repository
            by_repo: Dict[str, List[SWEInstance]] = {}
            for instance in instances:
                repo = instance.repo
                if repo not in by_repo:
                    by_repo[repo] = []
                by_repo[repo].append(instance)
            
            # Split each repository
            train, val, test = [], [], []
            
            for repo_instances in by_repo.values():
                shuffled = repo_instances.copy()
                random.shuffle(shuffled)
                
                n = len(shuffled)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                
                train.extend(shuffled[:n_train])
                val.extend(shuffled[n_train:n_train + n_val])
                test.extend(shuffled[n_train + n_val:])
            
            return {'train': train, 'val': val, 'test': test}
        
        else:
            # Simple random split
            shuffled = instances.copy()
            random.shuffle(shuffled)
            
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            return {
                'train': shuffled[:n_train],
                'val': shuffled[n_train:n_train + n_val],
                'test': shuffled[n_train + n_val:]
            }


class DatasetValidator:
    """
    Validator for dataset instances.
    
    Checks that instances have required fields and valid values.
    """
    
    REQUIRED_FIELDS = ['instance_id', 'repo', 'base_commit', 'problem_statement']
    
    def validate(self, instance: SWEInstance) -> List[str]:
        """
        Validate an instance.
        
        Args:
            instance: Instance to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            value = getattr(instance, field, None)
            if not value:
                errors.append(f"Missing required field: {field}")
        
        # Validate instance_id format
        if instance.instance_id:
            if '__' not in instance.instance_id:
                errors.append("Invalid instance_id format (expected: owner__repo-pr_number)")
        
        # Validate repo format
        if instance.repo:
            if '/' not in instance.repo:
                errors.append("Invalid repo format (expected: owner/repo)")
        
        return errors
    
    def validate_batch(
        self,
        instances: List[SWEInstance]
    ) -> Dict[str, List[str]]:
        """
        Validate multiple instances.
        
        Args:
            instances: List of instances to validate
            
        Returns:
            Dictionary mapping instance_id to list of errors
        """
        results = {}
        
        for instance in instances:
            errors = self.validate(instance)
            if errors:
                results[instance.instance_id] = errors
        
        return results
