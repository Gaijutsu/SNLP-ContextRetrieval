"""
Component registry for the SWE-bench comparison framework.

This module implements the factory pattern for creating components,
allowing easy registration and lookup of gatherers, generators, evaluators,
and other components.
"""

from typing import Any, Callable, Dict, Type, TypeVar, Optional
import logging

from .interfaces import (
    ContextGatherer,
    PatchGenerator,
    Evaluator,
    RepositoryIndexer,
)
from .exceptions import RegistryError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ComponentRegistry:
    """
    Registry for framework components.
    
    This class implements the factory pattern, allowing components to be
    registered by name and created on demand. It supports registration
    of context gatherers, patch generators, evaluators, and indexers.
    
    Example:
        >>> registry = ComponentRegistry()
        >>> registry.register_gatherer('my_gatherer', MyGatherer)
        >>> gatherer = registry.create_gatherer('my_gatherer', config)
    
    The registry is typically used as a singleton through the module-level
    convenience functions.
    """
    
    def __init__(self):
        """Initialize the component registry."""
        self._gatherers: Dict[str, Type[ContextGatherer]] = {}
        self._generators: Dict[str, Type[PatchGenerator]] = {}
        self._evaluators: Dict[str, Type[Evaluator]] = {}
        self._indexers: Dict[str, Type[RepositoryIndexer]] = {}
    
    # Context Gatherer Registration
    
    def register_gatherer(
        self,
        name: str,
        gatherer_class: Type[ContextGatherer]
    ) -> None:
        """
        Register a context gatherer class.
        
        Args:
            name: Unique name for the gatherer
            gatherer_class: The gatherer class to register
            
        Raises:
            RegistryError: If the name is already registered
        """
        if name in self._gatherers:
            raise RegistryError(
                f"Gatherer '{name}' is already registered",
                details={'existing': self._gatherers[name].__name__}
            )
        
        self._gatherers[name] = gatherer_class
        logger.debug(f"Registered gatherer: {name}")
    
    def create_gatherer(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> ContextGatherer:
        """
        Create a context gatherer instance.
        
        Args:
            name: Name of the registered gatherer
            config: Configuration dictionary for the gatherer
            
        Returns:
            Instance of the requested gatherer
            
        Raises:
            RegistryError: If the gatherer is not registered
        """
        if name not in self._gatherers:
            available = list(self._gatherers.keys())
            raise RegistryError(
                f"Gatherer '{name}' not found in registry",
                details={
                    'available': available,
                    'requested': name
                }
            )
        
        gatherer_class = self._gatherers[name]
        return gatherer_class(config)
    
    def list_gatherers(self) -> Dict[str, str]:
        """
        List all registered gatherers.
        
        Returns:
            Dictionary mapping names to class names
        """
        return {name: cls.__name__ for name, cls in self._gatherers.items()}
    
    def is_gatherer_registered(self, name: str) -> bool:
        """Check if a gatherer is registered."""
        return name in self._gatherers
    
    # Patch Generator Registration
    
    def register_generator(
        self,
        name: str,
        generator_class: Type[PatchGenerator]
    ) -> None:
        """
        Register a patch generator class.
        
        Args:
            name: Unique name for the generator
            generator_class: The generator class to register
            
        Raises:
            RegistryError: If the name is already registered
        """
        if name in self._generators:
            raise RegistryError(
                f"Generator '{name}' is already registered",
                details={'existing': self._generators[name].__name__}
            )
        
        self._generators[name] = generator_class
        logger.debug(f"Registered generator: {name}")
    
    def create_generator(
        self,
        name: str,
        llm_config: Dict[str, Any],
        config: Dict[str, Any]
    ) -> PatchGenerator:
        """
        Create a patch generator instance.
        
        Args:
            name: Name of the registered generator
            llm_config: LLM configuration dictionary
            config: Generator configuration dictionary
            
        Returns:
            Instance of the requested generator
            
        Raises:
            RegistryError: If the generator is not registered
        """
        if name not in self._generators:
            available = list(self._generators.keys())
            raise RegistryError(
                f"Generator '{name}' not found in registry",
                details={
                    'available': available,
                    'requested': name
                }
            )
        
        generator_class = self._generators[name]
        return generator_class(llm_config, config)
    
    def list_generators(self) -> Dict[str, str]:
        """
        List all registered generators.
        
        Returns:
            Dictionary mapping names to class names
        """
        return {name: cls.__name__ for name, cls in self._generators.items()}
    
    def is_generator_registered(self, name: str) -> bool:
        """Check if a generator is registered."""
        return name in self._generators
    
    # Evaluator Registration
    
    def register_evaluator(
        self,
        name: str,
        evaluator_class: Type[Evaluator]
    ) -> None:
        """
        Register an evaluator class.
        
        Args:
            name: Unique name for the evaluator
            evaluator_class: The evaluator class to register
            
        Raises:
            RegistryError: If the name is already registered
        """
        if name in self._evaluators:
            raise RegistryError(
                f"Evaluator '{name}' is already registered",
                details={'existing': self._evaluators[name].__name__}
            )
        
        self._evaluators[name] = evaluator_class
        logger.debug(f"Registered evaluator: {name}")
    
    def create_evaluator(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> Evaluator:
        """
        Create an evaluator instance.
        
        Args:
            name: Name of the registered evaluator
            config: Configuration dictionary for the evaluator
            
        Returns:
            Instance of the requested evaluator
            
        Raises:
            RegistryError: If the evaluator is not registered
        """
        if name not in self._evaluators:
            available = list(self._evaluators.keys())
            raise RegistryError(
                f"Evaluator '{name}' not found in registry",
                details={
                    'available': available,
                    'requested': name
                }
            )
        
        evaluator_class = self._evaluators[name]
        return evaluator_class(config)
    
    def list_evaluators(self) -> Dict[str, str]:
        """
        List all registered evaluators.
        
        Returns:
            Dictionary mapping names to class names
        """
        return {name: cls.__name__ for name, cls in self._evaluators.items()}
    
    def is_evaluator_registered(self, name: str) -> bool:
        """Check if an evaluator is registered."""
        return name in self._evaluators
    
    # Repository Indexer Registration
    
    def register_indexer(
        self,
        name: str,
        indexer_class: Type[RepositoryIndexer]
    ) -> None:
        """
        Register a repository indexer class.
        
        Args:
            name: Unique name for the indexer
            indexer_class: The indexer class to register
            
        Raises:
            RegistryError: If the name is already registered
        """
        if name in self._indexers:
            raise RegistryError(
                f"Indexer '{name}' is already registered",
                details={'existing': self._indexers[name].__name__}
            )
        
        self._indexers[name] = indexer_class
        logger.debug(f"Registered indexer: {name}")
    
    def create_indexer(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> RepositoryIndexer:
        """
        Create a repository indexer instance.
        
        Args:
            name: Name of the registered indexer
            config: Configuration dictionary for the indexer
            
        Returns:
            Instance of the requested indexer
            
        Raises:
            RegistryError: If the indexer is not registered
        """
        if name not in self._indexers:
            available = list(self._indexers.keys())
            raise RegistryError(
                f"Indexer '{name}' not found in registry",
                details={
                    'available': available,
                    'requested': name
                }
            )
        
        indexer_class = self._indexers[name]
        return indexer_class(config)
    
    def list_indexers(self) -> Dict[str, str]:
        """
        List all registered indexers.
        
        Returns:
            Dictionary mapping names to class names
        """
        return {name: cls.__name__ for name, cls in self._indexers.items()}
    
    def is_indexer_registered(self, name: str) -> bool:
        """Check if an indexer is registered."""
        return name in self._indexers
    
    # Utility Methods
    
    def clear(self) -> None:
        """Clear all registered components."""
        self._gatherers.clear()
        self._generators.clear()
        self._evaluators.clear()
        self._indexers.clear()
        logger.debug("Cleared component registry")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with counts of registered components
        """
        return {
            'gatherers': len(self._gatherers),
            'generators': len(self._generators),
            'evaluators': len(self._evaluators),
            'indexers': len(self._indexers),
        }


# Global registry instance
_global_registry = ComponentRegistry()


# Convenience functions that use the global registry

def register_gatherer(name: str, gatherer_class: Type[ContextGatherer]) -> None:
    """
    Register a context gatherer in the global registry.
    
    Args:
        name: Unique name for the gatherer
        gatherer_class: The gatherer class to register
    """
    _global_registry.register_gatherer(name, gatherer_class)


def register_generator(name: str, generator_class: Type[PatchGenerator]) -> None:
    """
    Register a patch generator in the global registry.
    
    Args:
        name: Unique name for the generator
        generator_class: The generator class to register
    """
    _global_registry.register_generator(name, generator_class)


def register_evaluator(name: str, evaluator_class: Type[Evaluator]) -> None:
    """
    Register an evaluator in the global registry.
    
    Args:
        name: Unique name for the evaluator
        evaluator_class: The evaluator class to register
    """
    _global_registry.register_evaluator(name, evaluator_class)


def register_indexer(name: str, indexer_class: Type[RepositoryIndexer]) -> None:
    """
    Register a repository indexer in the global registry.
    
    Args:
        name: Unique name for the indexer
        indexer_class: The indexer class to register
    """
    _global_registry.register_indexer(name, indexer_class)


def create_gatherer(name: str, config: Dict[str, Any]) -> ContextGatherer:
    """
    Create a context gatherer from the global registry.
    
    Args:
        name: Name of the registered gatherer
        config: Configuration dictionary for the gatherer
        
    Returns:
        Instance of the requested gatherer
    """
    return _global_registry.create_gatherer(name, config)


def create_generator(
    name: str,
    llm_config: Dict[str, Any],
    config: Dict[str, Any]
) -> PatchGenerator:
    """
    Create a patch generator from the global registry.
    
    Args:
        name: Name of the registered generator
        llm_config: LLM configuration dictionary
        config: Generator configuration dictionary
        
    Returns:
        Instance of the requested generator
    """
    return _global_registry.create_generator(name, llm_config, config)


def create_evaluator(name: str, config: Dict[str, Any]) -> Evaluator:
    """
    Create an evaluator from the global registry.
    
    Args:
        name: Name of the registered evaluator
        config: Configuration dictionary for the evaluator
        
    Returns:
        Instance of the requested evaluator
    """
    return _global_registry.create_evaluator(name, config)


def create_indexer(name: str, config: Dict[str, Any]) -> RepositoryIndexer:
    """
    Create a repository indexer from the global registry.
    
    Args:
        name: Name of the registered indexer
        config: Configuration dictionary for the indexer
        
    Returns:
        Instance of the requested indexer
    """
    return _global_registry.create_indexer(name, config)


def list_registered_components() -> Dict[str, Dict[str, str]]:
    """
    List all registered components.
    
    Returns:
        Dictionary with component types as keys and
        dictionaries of registered components as values
    """
    return {
        'gatherers': _global_registry.list_gatherers(),
        'generators': _global_registry.list_generators(),
        'evaluators': _global_registry.list_evaluators(),
        'indexers': _global_registry.list_indexers(),
    }


def get_registry_stats() -> Dict[str, int]:
    """
    Get global registry statistics.
    
    Returns:
        Dictionary with counts of registered components
    """
    return _global_registry.get_stats()


def clear_registry() -> None:
    """Clear the global registry."""
    _global_registry.clear()


# Decorator-based registration

def gatherer(name: str) -> Callable[[Type[ContextGatherer]], Type[ContextGatherer]]:
    """
    Decorator to register a context gatherer.
    
    Example:
        >>> @gatherer('my_gatherer')
        ... class MyGatherer(ContextGatherer):
        ...     pass
    """
    def decorator(cls: Type[ContextGatherer]) -> Type[ContextGatherer]:
        register_gatherer(name, cls)
        return cls
    return decorator


def generator(name: str) -> Callable[[Type[PatchGenerator]], Type[PatchGenerator]]:
    """
    Decorator to register a patch generator.
    
    Example:
        >>> @generator('my_generator')
        ... class MyGenerator(PatchGenerator):
        ...     pass
    """
    def decorator(cls: Type[PatchGenerator]) -> Type[PatchGenerator]:
        register_generator(name, cls)
        return cls
    return decorator


def evaluator(name: str) -> Callable[[Type[Evaluator]], Type[Evaluator]]:
    """
    Decorator to register an evaluator.
    
    Example:
        >>> @evaluator('my_evaluator')
        ... class MyEvaluator(Evaluator):
        ...     pass
    """
    def decorator(cls: Type[Evaluator]) -> Type[Evaluator]:
        register_evaluator(name, cls)
        return cls
    return decorator


def indexer(name: str) -> Callable[[Type[RepositoryIndexer]], Type[RepositoryIndexer]]:
    """
    Decorator to register a repository indexer.
    
    Example:
        >>> @indexer('my_indexer')
        ... class MyIndexer(RepositoryIndexer):
        ...     pass
    """
    def decorator(cls: Type[RepositoryIndexer]) -> Type[RepositoryIndexer]:
        register_indexer(name, cls)
        return cls
    return decorator
