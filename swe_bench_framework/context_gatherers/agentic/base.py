"""
Base AgenticGatherer class for agentic exploration methods.

This module provides the abstract base class that all agentic context gathering
methods must implement, including common agent infrastructure and state management.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
import time

from ..base import ContextGatherer, ContextBundle, ContextChunk, ContextType, SWEInstance
from .environment import AgentEnvironment, AgentAction, Observation, AgentState


class BaseAgenticGatherer(ContextGatherer):
    """
    Abstract base class for agentic context gathering methods.
    
    This class provides common infrastructure for agentic methods including:
    - Agent environment management
    - State tracking
    - Common tool registration
    - Context bundle assembly from agent state
    
    Subclasses must implement the `explore` method to define their
    specific exploration strategy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agentic gatherer.
        
        Args:
            config: Configuration dictionary containing:
                - max_iterations: Maximum exploration iterations (default: 50)
                - timeout: Maximum time in seconds (default: 300)
                - tools: List of tool names to enable
                - llm_config: Configuration for LLM-based decisions
        """
        super().__init__(config)
        self.max_iterations = config.get('max_iterations', 50)
        self.timeout = config.get('timeout', 300)
        self.llm_config = config.get('llm_config', {})
        
        # These will be initialized per-instance
        self.environment: Optional[AgentEnvironment] = None
        self.repo_path: Optional[str] = None
        self._stats: Dict[str, Any] = {
            'total_iterations': 0,
            'total_time': 0.0,
            'tools_used': {},
            'files_viewed': 0
        }
    
    def initialize(self, repo_path: str) -> None:
        """
        Initialize the agent environment for a repository.
        
        Args:
            repo_path: Path to the repository checkout
        """
        import logging
        logger = logging.getLogger(__name__)
        
        self.repo_path = repo_path
        logger.info(f"Initializing agent environment for: {repo_path}")
        
        self.environment = AgentEnvironment(repo_path, self.config)
        self._register_default_tools()
        self._stats = {
            'total_iterations': 0,
            'total_time': 0.0,
            'tools_used': {},
            'files_viewed': 0
        }
        logger.info(f"Agent environment initialized (max_iterations: {self.max_iterations})")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.environment = None
        self.repo_path = None
    
    def gather_context(
        self,
        instance: SWEInstance,
        repo_path: str
    ) -> ContextBundle:
        """
        Gather context using agentic exploration.
        
        This method orchestrates the exploration process:
        1. Initialize environment
        2. Run exploration loop
        3. Convert state to ContextBundle
        
        Args:
            instance: The SWE-bench instance
            repo_path: Path to the repository
            
        Returns:
            ContextBundle containing gathered context
        """
        start_time = time.time()
        
        # Initialize if not already done
        if self.environment is None or self.repo_path != repo_path:
            self.initialize(repo_path)
        
        # Run exploration
        self.explore(instance)
        
        # Convert state to context bundle
        context_bundle = self._state_to_context_bundle(instance)
        
        # Update stats
        elapsed = time.time() - start_time
        self._stats['total_time'] = elapsed
        if self.environment:
            self._stats['total_iterations'] = self.environment.state.current_iteration
            self._stats['files_viewed'] = len(self.environment.state.viewed_files)
        
        return context_bundle
    
    @abstractmethod
    def explore(self, instance: SWEInstance) -> None:
        """
        Run the agentic exploration for an instance.
        
        This is the main method that subclasses must implement.
        It should use self.environment to execute actions and gather context.
        
        Args:
            instance: The SWE-bench instance to explore
        """
        pass
    
    def _register_default_tools(self) -> None:
        """
        Register default tools available to agents.
        
        Subclasses can override this to customize available tools.
        """
        # Import tools here to avoid circular dependencies
        from .tools.file_tools import ViewFileTool, GrepTool, FindFileTool
        from .tools.search_tools import SearchClassTool, SearchMethodTool
        from .tools.execution_tools import RunTestTool, LinterTool
        
        if self.environment is None:
            return
        
        # File tools
        self.environment.register_tool('view_file', ViewFileTool(self.repo_path))
        self.environment.register_tool('grep', GrepTool(self.repo_path))
        self.environment.register_tool('find_file', FindFileTool(self.repo_path))
        
        # Search tools
        self.environment.register_tool('search_class', SearchClassTool(self.repo_path))
        self.environment.register_tool('search_method', SearchMethodTool(self.repo_path))
        
        # Execution tools
        self.environment.register_tool('run_test', RunTestTool(self.repo_path))
        self.environment.register_tool('linter', LinterTool(self.repo_path))
    
    def _state_to_context_bundle(self, instance: SWEInstance) -> ContextBundle:
        """
        Convert agent state to ContextBundle.
        
        Args:
            instance: The SWE-bench instance
            
        Returns:
            ContextBundle with gathered context
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized")
        
        state = self.environment.state
        
        # Get context chunks from state
        chunks = state.context_chunks.copy()
        
        # Add viewed files as context if not already included
        for file_path, ranges in state.viewed_files.items():
            # Check if this file is already in chunks
            already_included = any(
                c.source_file == file_path for c in chunks
            )
            if not already_included:
                # Add as a file content chunk
                chunks.append(ContextChunk(
                    content=f"File viewed: {file_path}",
                    source_file=file_path,
                    context_type=ContextType.FILE_CONTENT,
                    start_line=0,
                    end_line=0,
                    relevance_score=0.5,  # Medium relevance
                    metadata={'viewed_ranges': ranges}
                ))
        
        # Add search history as metadata
        search_context = self._format_search_history(state.search_history)
        if search_context:
            chunks.append(ContextChunk(
                content=search_context,
                source_file="search_history",
                context_type=ContextType.REPO_STRUCTURE,
                start_line=0,
                end_line=0,
                relevance_score=0.3,
                metadata={'search_count': len(state.search_history)}
            ))
        
        # Estimate token count
        token_count = sum(len(c.content) // 4 for c in chunks)
        
        # Build repo structure info
        repo_structure = {
            'files_viewed': list(state.viewed_files.keys()),
            'symbols_found': state.found_symbols,
            'search_count': len(state.search_history),
            'iteration_count': state.current_iteration
        }
        
        return ContextBundle(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            chunks=chunks,
            repo_structure=repo_structure,
            gatherer_type=self.name,
            token_count=token_count,
            metadata={
                'iterations': state.current_iteration,
                'completion_reason': state.completion_reason,
                'elapsed_time': state.get_elapsed_time()
            }
        )
    
    def _format_search_history(self, search_history: List[Dict[str, Any]]) -> str:
        """Format search history as a string."""
        if not search_history:
            return ""
        
        lines = ["## Search History"]
        for i, search in enumerate(search_history[-10:], 1):  # Last 10 searches
            lines.append(f"{i}. {search['type']}: {search['query']}")
            if search.get('results'):
                lines.append(f"   Results: {', '.join(search['results'][:3])}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about context gathering."""
        return self._stats.copy()
    
    def execute_tool(self, tool_name: str, **parameters) -> Observation:
        """
        Convenience method to execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **parameters: Tool parameters
            
        Returns:
            Observation from tool execution
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized")
        
        action = AgentAction(tool_name=tool_name, parameters=parameters)
        observation = self.environment.execute_action(action)
        
        # Update tool usage stats
        if tool_name not in self._stats['tools_used']:
            self._stats['tools_used'][tool_name] = 0
        self._stats['tools_used'][tool_name] += 1
        
        return observation
