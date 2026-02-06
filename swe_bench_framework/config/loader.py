"""
Configuration loader for the SWE-bench comparison framework.

This module provides functions to load configuration from YAML files,
with support for environment variable substitution and validation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .schema import (
    ExperimentConfig,
    DatasetConfig,
    LLMConfig,
    MethodConfig,
    PatchGenerationConfig,
    EvaluationConfig,
    LoggingConfig,
    RateLimitConfig,
    SandboxConfig,
    ExperimentTrackingConfig,
)
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def _resolve_env_vars(obj: Any) -> Any:
    """
    Recursively replace ${VAR} with environment variable values.
    
    Args:
        obj: Object to process (dict, list, str, or other)
        
    Returns:
        Object with environment variables resolved
    """
    pattern = r'\$\{([^}]+)\}'
    
    def replace_vars(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: replace_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [replace_vars(item) for item in value]
        elif isinstance(value, str):
            def replacer(match: re.Match) -> str:
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))
            return re.sub(pattern, replacer, value)
        return value
    
    return replace_vars(obj)


def _load_yaml_file(path: str) -> Dict[str, Any]:
    """
    Load a YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary with the loaded configuration
        
    Raises:
        ConfigurationError: If the file cannot be loaded
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError(
            "PyYAML is required to load YAML configuration files. "
            "Install it with: pip install pyyaml"
        )
    
    file_path = Path(path)
    
    if not file_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}",
            details={'path': str(file_path.absolute())}
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML file: {e}",
            details={'path': str(file_path.absolute()), 'error': str(e)}
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to read configuration file: {e}",
            details={'path': str(file_path.absolute()), 'error': str(e)}
        )
    
    if content is None:
        content = {}
    
    if not isinstance(content, dict):
        raise ConfigurationError(
            "Configuration file must contain a dictionary",
            details={'path': str(file_path.absolute()), 'type': type(content).__name__}
        )
    
    return content


def _build_llm_config(config: Dict[str, Any]) -> LLMConfig:
    """Build LLMConfig from dictionary."""
    rate_limit_config = config.get('rate_limit', {})
    if isinstance(rate_limit_config, dict):
        rate_limit = RateLimitConfig(**rate_limit_config)
    else:
        rate_limit = RateLimitConfig()
    
    return LLMConfig(
        provider=config.get('provider', 'openai'),
        model=config.get('model', 'gpt-5-mini'),
        temperature=config.get('temperature', 0.0),
        max_tokens=config.get('max_tokens', 4096),
        top_p=config.get('top_p', 1.0),
        api_key=config.get('api_key'),
        api_base=config.get('api_base'),
        rate_limit=rate_limit,
        additional_params=config.get('additional_params', {}),
    )


def _build_dataset_config(config: Dict[str, Any]) -> DatasetConfig:
    """Build DatasetConfig from dictionary."""
    return DatasetConfig(
        name=config.get('name', 'swe-bench-lite'),
        split=config.get('split', 'test'),
        filter_repos=config.get('filter_repos'),
        max_instances=config.get('max_instances'),
        instance_ids=config.get('instance_ids'),
        cache_dir=config.get('cache_dir'),
    )


def _build_method_config(config: Dict[str, Any]) -> MethodConfig:
    """Build MethodConfig from dictionary."""
    # Build config dict from all fields not explicitly handled
    method_config = dict(config)
    method_config.pop('name', None)
    method_config.pop('type', None)
    method_config.pop('enabled', None)
    
    return MethodConfig(
        name=config['name'],
        type=config['type'],
        enabled=config.get('enabled', True),
        config=method_config,
    )


def _build_patch_generation_config(config: Dict[str, Any]) -> PatchGenerationConfig:
    """Build PatchGenerationConfig from dictionary."""
    return PatchGenerationConfig(
        strategy=config.get('strategy', 'direct'),
        max_attempts=config.get('max_attempts', 3),
        syntax_check=config.get('syntax_check', True),
        test_before_submit=config.get('test_before_submit', False),
        prompt_template=config.get('prompt_template'),
        max_context_tokens=config.get('max_context_tokens', 8000),
    )


def _build_sandbox_config(config: Dict[str, Any]) -> SandboxConfig:
    """Build SandboxConfig from dictionary."""
    return SandboxConfig(
        type=config.get('type', 'docker'),
        image=config.get('image', 'swe-bench/sandbox:latest'),
        timeout=config.get('timeout', 300.0),
        memory_limit=config.get('memory_limit', '4g'),
        cpu_limit=config.get('cpu_limit'),
        network_disabled=config.get('network_disabled', True),
    )


def _build_evaluation_config(config: Dict[str, Any]) -> EvaluationConfig:
    """Build EvaluationConfig from dictionary."""
    sandbox_config = config.get('sandbox', {})
    if isinstance(sandbox_config, dict):
        sandbox = _build_sandbox_config(sandbox_config)
    else:
        sandbox = SandboxConfig()
    
    return EvaluationConfig(
        sandbox=sandbox,
        metrics=config.get('metrics', [
            'resolution_rate',
            'localization_accuracy',
            'codebleu',
            'token_usage',
        ]),
        localization_k_values=config.get('localization_k_values', [1, 3, 5, 10]),
        compute_codebleu=config.get('compute_codebleu', True),
    )


def _build_experiment_tracking_config(config: Dict[str, Any]) -> ExperimentTrackingConfig:
    """Build ExperimentTrackingConfig from dictionary."""
    return ExperimentTrackingConfig(
        enabled=config.get('enabled', True),
        backend=config.get('backend', 'mlflow'),
        uri=config.get('uri'),
        experiment_name=config.get('experiment_name'),
    )


def _build_logging_config(config: Dict[str, Any]) -> LoggingConfig:
    """Build LoggingConfig from dictionary."""
    tracking_config = config.get('experiment_tracking', {})
    if isinstance(tracking_config, dict):
        tracking = _build_experiment_tracking_config(tracking_config)
    else:
        tracking = ExperimentTrackingConfig()
    
    return LoggingConfig(
        level=config.get('level', 'INFO'),
        format=config.get('format', 'structured'),
        file_path=config.get('file_path'),
        stdout=config.get('stdout', True),
        experiment_tracking=tracking,
    )


def load_config_from_dict(config: Dict[str, Any]) -> ExperimentConfig:
    """
    Load experiment configuration from a dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ExperimentConfig object
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    # Resolve environment variables
    config = _resolve_env_vars(config)
    
    # Get experiment section
    experiment = config.get('experiment', {})
    if not experiment:
        # Try using top-level keys - map top-level names to experiment names
        # Build dataset config from top-level keys
        dataset_value = config.get('dataset', {})
        if isinstance(dataset_value, str):
            dataset = {'name': dataset_value}
        elif isinstance(dataset_value, dict):
            dataset = dataset_value
        else:
            dataset = {}
        # Include instance_ids from top level if present
        if 'instance_ids' in config:
            dataset['instance_ids'] = config['instance_ids']
        
        experiment = {
            'name': config.get('experiment_name', 'unnamed_experiment'),
            'description': config.get('description', ''),
            'output_dir': config.get('output_dir', './results'),
            'random_seed': config.get('random_seed', 42),
            'dataset': dataset,
            'llm': config.get('llm', {}),
            'methods': config.get('methods', []),
            'patch_generation': config.get('patch_generation', {}),
            'evaluation': config.get('evaluation', {}),
            'logging': config.get('logging', {}),
            'tracking': config.get('tracking', {}),
        }
    
    # Build sub-configs
    dataset_config = experiment.get('dataset', {})
    if isinstance(dataset_config, str):
        # Handle shorthand: dataset: "lite" or dataset: "verified"
        dataset = DatasetConfig(name=dataset_config)
    elif isinstance(dataset_config, dict):
        dataset = _build_dataset_config(dataset_config)
    else:
        dataset = DatasetConfig()
    
    llm_config = experiment.get('llm', {})
    if isinstance(llm_config, dict):
        llm = _build_llm_config(llm_config)
    else:
        llm = LLMConfig()
    
    methods_config = experiment.get('methods', [])
    if isinstance(methods_config, dict):
        # Handle dict format: method_name: {config...}
        methods = []
        for name, cfg in methods_config.items():
            method_cfg = dict(cfg) if isinstance(cfg, dict) else {}
            method_cfg['name'] = name
            methods.append(_build_method_config(method_cfg))
    elif isinstance(methods_config, list):
        methods = [_build_method_config(m) for m in methods_config]
    else:
        methods = []
    
    patch_gen_config = experiment.get('patch_generation', {})
    if isinstance(patch_gen_config, dict):
        patch_generation = _build_patch_generation_config(patch_gen_config)
    else:
        patch_generation = PatchGenerationConfig()
    
    eval_config = experiment.get('evaluation', {})
    if isinstance(eval_config, dict):
        evaluation = _build_evaluation_config(eval_config)
    else:
        evaluation = EvaluationConfig()
    
    logging_config = experiment.get('logging', {})
    if isinstance(logging_config, dict):
        logging_cfg = _build_logging_config(logging_config)
    else:
        logging_cfg = LoggingConfig()
    
    # Get tracking config (top-level or from experiment dict)
    tracking_config = experiment.get('tracking', {})
    
    # Build main config
    return ExperimentConfig(
        name=experiment.get('name', 'unnamed_experiment'),
        description=experiment.get('description', ''),
        output_dir=experiment.get('output_dir', './results'),
        random_seed=experiment.get('random_seed', 42),
        dataset=dataset,
        llm=llm,
        methods=methods,
        patch_generation=patch_generation,
        evaluation=evaluation,
        logging=logging_cfg,
        tracking=tracking_config,
    )


def load_config(path: str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.
    
    This function loads a YAML configuration file, resolves environment
    variables (syntax: ${VAR_NAME}), and validates the configuration.
    
    Args:
        path: Path to the YAML configuration file
        
    Returns:
        ExperimentConfig object
        
    Raises:
        ConfigurationError: If the file cannot be loaded or is invalid
        
    Example:
        >>> config = load_config('experiment_config.yaml')
        >>> print(config.name)
        'my_experiment'
        >>> print(config.llm.model)
        'gpt-5-mini'
    """
    # Load YAML file
    raw_config = _load_yaml_file(path)
    
    # Build config from dictionary
    return load_config_from_dict(raw_config)


def load_config_from_env(
    env_var: str = "SWE_BENCH_CONFIG",
    default_path: Optional[str] = None
) -> ExperimentConfig:
    """
    Load configuration from environment variable or default path.
    
    Args:
        env_var: Environment variable name containing config path
        default_path: Default configuration path if env var not set
        
    Returns:
        ExperimentConfig object
        
    Raises:
        ConfigurationError: If no configuration can be loaded
    """
    config_path = os.environ.get(env_var, default_path)
    
    if config_path is None:
        raise ConfigurationError(
            f"No configuration found. Set {env_var} environment variable "
            "or provide a default path."
        )
    
    return load_config(config_path)


def validate_config(config: ExperimentConfig) -> None:
    """
    Validate an experiment configuration.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ConfigurationError: If the configuration is invalid
    """
    errors = []
    
    # Check required fields
    if not config.name:
        errors.append("Experiment name is required")
    
    # Check methods
    if not config.methods:
        errors.append("At least one method must be configured")
    
    enabled_methods = config.get_enabled_methods()
    if not enabled_methods:
        errors.append("At least one method must be enabled")
    
    # Check for duplicate method names
    method_names = [m.name for m in config.methods]
    if len(method_names) != len(set(method_names)):
        errors.append("Duplicate method names found")
    
    # Check LLM configuration
    if config.llm.provider not in ('openai', 'anthropic', 'local', 'azure'):
        errors.append(
            f"Invalid LLM provider: {config.llm.provider}. "
            "Must be one of: openai, anthropic, local, azure"
        )
    
    # Check output directory
    if config.output_dir:
        output_path = Path(config.output_dir)
        if output_path.exists() and not output_path.is_dir():
            errors.append(f"Output path exists but is not a directory: {config.output_dir}")
    
    if errors:
        raise ConfigurationError(
            "Configuration validation failed",
            details={'errors': errors}
        )
    
    logger.info("Configuration validation passed")


class ConfigLoader:
    """
    Configuration loader class for loading and creating orchestrators from config files.
    
    This class wraps the config loading functions to provide a convenient interface
    used by the CLI.
    """
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        config = load_config(path)
        
        # Normalize dataset name to format expected by CLI
        dataset_name = config.dataset.name
        if 'lite' in dataset_name.lower():
            dataset = 'lite'
        elif 'verified' in dataset_name.lower():
            dataset = 'verified'
        elif 'full' in dataset_name.lower() or 'swe-bench' in dataset_name.lower():
            dataset = 'full'
        else:
            dataset = dataset_name
        
        # Extract methods from config
        methods = {}
        for method in config.methods:
            methods[method.name] = {
                'type': method.type,
                'enabled': method.enabled,
                'config': method.config,
            }
        
        # Convert back to dict format expected by CLI
        result = {
            'experiment_name': config.name,
            'output_dir': config.output_dir,
            'dataset': dataset,
            'sandbox': config.evaluation.sandbox.type if hasattr(config.evaluation, 'sandbox') else 'local',
            'methods': methods,
            'max_workers': 1,
        }
        
        # Include instance_ids if specified in dataset config
        if config.dataset.instance_ids:
            result['instance_ids'] = config.dataset.instance_ids
        
        # Include max_instances if specified
        if config.dataset.max_instances:
            result['max_instances'] = config.dataset.max_instances
        
        # Include tracking config if specified
        # Use top-level tracking config from the ExperimentConfig
        if config.tracking:
            result['tracking'] = config.tracking
        elif config.logging and config.logging.experiment_tracking:
            # Fallback to logging.experiment_tracking for backward compatibility
            tracking = config.logging.experiment_tracking
            result['tracking'] = {
                'enabled': tracking.enabled,
                'backend': tracking.backend,
                'uri': tracking.uri,
                'experiment_name': tracking.experiment_name,
            }
        
        return result
    
    def create_orchestrator(self, experiment_config: Dict[str, Any]) -> Any:
        """
        Create an orchestrator from experiment configuration.
        
        Args:
            experiment_config: Experiment configuration dictionary
            
        Returns:
            ExperimentOrchestrator instance
        """
        import logging
        logger = logging.getLogger(__name__)
        
        from ..experiment.orchestrator import ExperimentOrchestrator, OrchestratorConfig
        
        # Get tracking config from logging section or top-level tracking section
        tracking_config = experiment_config.get('tracking', {})
        logger.info(f"[CONFIG] Top-level tracking config: {tracking_config}")
        
        if not tracking_config:
            # Try to get from logging.experiment_tracking
            logging_config = experiment_config.get('logging', {})
            if isinstance(logging_config, dict):
                tracking_config = logging_config.get('experiment_tracking', {})
                logger.info(f"[CONFIG] Tracking config from logging: {tracking_config}")
        
        config = OrchestratorConfig(
            experiment_name=experiment_config.get('experiment_name', 'unnamed'),
            output_dir=experiment_config.get('output_dir', './results'),
            max_workers_per_method=experiment_config.get('max_workers', 1),
            tracking=tracking_config,
        )
        
        logger.info(f"[CONFIG] Orchestrator tracking config: {config.tracking}")
        
        return ExperimentOrchestrator(config)


def save_config(config: ExperimentConfig, path: str) -> None:
    """
    Save experiment configuration to a YAML file.
    
    Args:
        config: Configuration to save
        path: Path to save the configuration
        
    Raises:
        ConfigurationError: If the configuration cannot be saved
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError(
            "PyYAML is required to save YAML configuration files. "
            "Install it with: pip install pyyaml"
        )
    
    try:
        config_dict = config.to_dict()
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {path}")
    except Exception as e:
        raise ConfigurationError(
            f"Failed to save configuration: {e}",
            details={'path': path, 'error': str(e)}
        )
