"""
Agent Environment for agentic exploration methods.

This module provides the environment infrastructure for agentic context gathering,
including state management, tool registry, and action execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable
import time

from ..base import ContextChunk, ContextType


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    tool_name: str
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    iteration: int = 0
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.parameters.items())
        return f"AgentAction({self.tool_name}: {params_str})"


@dataclass
class Observation:
    """Result of executing an action."""
    output: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    
    def __post_init__(self):
        if self.error and self.success:
            self.success = False
    
    def __repr__(self) -> str:
        if self.error:
            return f"Observation(error={self.error!r})"
        return f"Observation(output={self.output[:100]!r}...)" if len(self.output) > 100 else f"Observation(output={self.output!r})"


@dataclass
class AgentState:
    """
    Tracks the state of an agent during exploration.
    
    This includes viewed files, search history, and other stateful
    information needed for reproducibility and context assembly.
    """
    # Files that have been viewed
    viewed_files: Dict[str, List[tuple]] = field(default_factory=dict)
    # Search history
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    # Action history
    action_history: List[AgentAction] = field(default_factory=list)
    # Current iteration
    current_iteration: int = 0
    # Maximum iterations allowed
    max_iterations: int = 50
    # Files that have been modified (for tracking)
    touched_files: Set[str] = field(default_factory=set)
    # Classes/methods found
    found_symbols: Dict[str, List[str]] = field(default_factory=lambda: {
        'classes': [],
        'methods': [],
        'functions': []
    })
    # Context gathered so far (for incremental gathering)
    context_chunks: List[ContextChunk] = field(default_factory=list)
    # Whether gathering is complete
    is_complete: bool = False
    # Reason for completion
    completion_reason: Optional[str] = None
    # Start time
    start_time: float = field(default_factory=time.time)
    # Token usage tracking
    token_usage: Dict[str, int] = field(default_factory=lambda: {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    })
    
    def record_file_view(self, file_path: str, start_line: int, end_line: int) -> None:
        """Record that a file was viewed."""
        if file_path not in self.viewed_files:
            self.viewed_files[file_path] = []
        self.viewed_files[file_path].append((start_line, end_line))
        self.touched_files.add(file_path)
    
    def record_search(self, search_type: str, query: str, results: List[str]) -> None:
        """Record a search operation."""
        self.search_history.append({
            'type': search_type,
            'query': query,
            'results': results,
            'timestamp': time.time()
        })
    
    def record_action(self, action: AgentAction) -> None:
        """Record an action in the history."""
        action.iteration = self.current_iteration
        self.action_history.append(action)
    
    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.current_iteration += 1
    
    def is_at_iteration_limit(self) -> bool:
        """Check if the agent has reached its iteration limit."""
        return self.current_iteration >= self.max_iterations
    
    def add_context_chunk(self, chunk: ContextChunk) -> None:
        """Add a context chunk to the gathered context."""
        self.context_chunks.append(chunk)
    
    def mark_complete(self, reason: str) -> None:
        """Mark the exploration as complete."""
        self.is_complete = True
        self.completion_reason = reason
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time
    
    def get_unique_files_viewed(self) -> Set[str]:
        """Get set of unique files that have been viewed."""
        return set(self.viewed_files.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'viewed_files': self.viewed_files,
            'search_history': self.search_history,
            'current_iteration': self.current_iteration,
            'max_iterations': self.max_iterations,
            'touched_files': list(self.touched_files),
            'found_symbols': self.found_symbols,
            'is_complete': self.is_complete,
            'completion_reason': self.completion_reason,
            'elapsed_time': self.get_elapsed_time(),
            'token_usage': self.token_usage
        }


class Tool:
    """Base class for agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Execute the tool with given parameters.
        
        Args:
            parameters: Tool-specific parameters
            
        Returns:
            Observation containing the result
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {}
        }


class AgentEnvironment:
    """
    Environment for agentic exploration.
    
    Provides tools and state management for agents. This is the core
    infrastructure that enables agents to explore codebases systematically.
    """
    
    def __init__(self, repo_path: str, config: Dict[str, Any]):
        """
        Initialize the agent environment.
        
        Args:
            repo_path: Path to the repository being explored
            config: Configuration dictionary
        """
        self.repo_path = repo_path
        self.config = config
        self.state = AgentState(
            max_iterations=config.get('max_iterations', 50)
        )
        self.tools: Dict[str, Tool] = {}
        self._observers: List[Callable[[AgentAction, Observation], None]] = []
        
    def register_tool(self, name: str, tool: Tool) -> None:
        """
        Register a tool for agent use.
        
        Args:
            name: Tool name
            tool: Tool instance
        """
        self.tools[name] = tool
        
    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())
    
    def execute_action(self, action: AgentAction) -> Observation:
        """
        Execute an agent action and return observation.
        
        Args:
            action: The action to execute
            
        Returns:
            Observation containing the result
        """
        tool = self.tools.get(action.tool_name)
        
        if not tool:
            return Observation(
                error=f"Unknown tool: {action.tool_name}",
                success=False
            )
        
        # Record the action
        self.state.record_action(action)
        
        try:
            # Execute the tool
            observation = tool.execute(action.parameters)
        except Exception as e:
            observation = Observation(
                error=f"Tool execution failed: {str(e)}",
                success=False
            )
        
        # Notify observers
        for observer in self._observers:
            try:
                observer(action, observation)
            except Exception:
                pass  # Don't let observers break execution
        
        return observation
    
    def add_observer(self, callback: Callable[[AgentAction, Observation], None]) -> None:
        """
        Add an observer to be notified of all actions and observations.
        
        Args:
            callback: Function to call with (action, observation)
        """
        self._observers.append(callback)
    
    def get_context(self) -> List[ContextChunk]:
        """
        Convert agent state to context chunks.
        
        Returns:
            List of context chunks from the exploration
        """
        return self.state.context_chunks
    
    def get_tool_descriptions(self) -> str:
        """
        Get descriptions of all available tools.
        
        Returns:
            Formatted string with tool descriptions
        """
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def reset(self) -> None:
        """Reset the environment state."""
        self.state = AgentState(
            max_iterations=self.config.get('max_iterations', 50)
        )
    
    def is_exploration_complete(self) -> bool:
        """Check if exploration should stop."""
        return (
            self.state.is_complete or
            self.state.is_at_iteration_limit()
        )
