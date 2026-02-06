"""
Agentic exploration methods for context gathering.

This module provides agentic (exploration-based) context gathering strategies
for the SWE-bench comparison framework. These methods use iterative exploration
with tools to navigate and understand codebases.

Available gatherers:
- AutoCodeRoverGatherer: Stratified retrieval with AST-based search
- SWEAgentGatherer: ReAct pattern with ACI-style tools
- AgentlessGatherer: Hierarchical localization with fixed workflow
"""

from .base import BaseAgenticGatherer
from .environment import (
    AgentEnvironment,
    AgentAction,
    Observation,
    AgentState,
    Tool
)

from .autocoderover import AutoCodeRoverGatherer
from .swe_agent import SWEAgentGatherer
from .agentless import AgentlessGatherer

__all__ = [
    # Base classes
    'BaseAgenticGatherer',
    'AgentEnvironment',
    'AgentAction',
    'Observation',
    'AgentState',
    'Tool',
    
    # Agent implementations
    'AutoCodeRoverGatherer',
    'SWEAgentGatherer',
    'AgentlessGatherer',
]

# Version of the agentic module
__version__ = '0.1.0'


# Factory function for creating agentic gatherers
def create_agentic_gatherer(
    gatherer_type: str,
    config: dict
) -> BaseAgenticGatherer:
    """
    Factory function to create agentic gatherer instances.
    
    Args:
        gatherer_type: Type of gatherer to create
            - 'autocoderover': AutoCodeRover-style stratified retrieval
            - 'swe_agent': SWE-agent style ReAct exploration
            - 'agentless': Agentless-style hierarchical localization
        config: Configuration dictionary for the gatherer
        
    Returns:
        Configured gatherer instance
        
    Raises:
        ValueError: If gatherer_type is not recognized
        
    Example:
        >>> gatherer = create_agentic_gatherer('autocoderover', {
        ...     'max_iterations': 50,
        ...     'max_files_per_layer': 5
        ... })
    """
    gatherer_map = {
        'autocoderover': AutoCodeRoverGatherer,
        'swe_agent': SWEAgentGatherer,
        'agentless': AgentlessGatherer,
    }
    
    if gatherer_type not in gatherer_map:
        raise ValueError(
            f"Unknown gatherer type: {gatherer_type}. "
            f"Available types: {list(gatherer_map.keys())}"
        )
    
    return gatherer_map[gatherer_type](config)


# Registry for custom gatherers
_gatherer_registry: dict = {}


def register_gatherer(name: str, gatherer_class: type) -> None:
    """
    Register a custom agentic gatherer.
    
    Args:
        name: Name to register the gatherer under
        gatherer_class: Class inheriting from BaseAgenticGatherer
        
    Example:
        >>> class MyCustomGatherer(BaseAgenticGatherer):
        ...     def explore(self, instance):
        ...         pass
        >>> register_gatherer('my_custom', MyCustomGatherer)
    """
    if not issubclass(gatherer_class, BaseAgenticGatherer):
        raise ValueError("Gatherer must inherit from BaseAgenticGatherer")
    
    _gatherer_registry[name] = gatherer_class


def get_registered_gatherers() -> dict:
    """Get all registered custom gatherers."""
    return _gatherer_registry.copy()
