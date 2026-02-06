"""
SWE-agent style agentic context gatherer.

This module implements the SWE-agent approach to context gathering,
featuring:
- Agent-Computer Interface (ACI) with limited output
- ReAct pattern (Thought → Action → Observation)
- Iterative exploration loop
- Linter integration

Reference: https://arxiv.org/abs/2405.15793
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgenticGatherer
from .environment import AgentAction, Observation, AgentState
from ..base import ContextChunk, ContextType, SWEInstance


class ReActStep:
    """Represents a single step in the ReAct loop."""
    
    def __init__(
        self,
        iteration: int,
        thought: str,
        action: AgentAction,
        observation: Observation
    ):
        self.iteration = iteration
        self.thought = thought
        self.action = action
        self.observation = observation
    
    def to_string(self) -> str:
        """Convert step to string representation."""
        lines = [
            f"Step {self.iteration}:",
            f"  Thought: {self.thought}",
            f"  Action: {self.action}",
            f"  Observation: {self.observation.output[:200]}..." if len(self.observation.output) > 200 else f"  Observation: {self.observation.output}"
        ]
        return "\n".join(lines)


class SWEAgentGatherer(BaseAgenticGatherer):
    """
    SWE-agent style context gatherer using ReAct pattern.
    
    Implements the Agent-Computer Interface (ACI) approach where:
    1. The agent thinks about what to do next
    2. Takes an action using available tools
    3. Observes the result
    4. Repeats until sufficient context is gathered
    
    Key features:
    - Limited output from tools (ACI principle)
    - Iterative decision making
    - State tracking across iterations
    - Completion detection
    
    Attributes:
        config: Configuration dictionary with keys:
            - max_iterations: Maximum exploration iterations
            - max_thought_length: Maximum length of thoughts
            - enable_completion_detection: Auto-detect when done
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SWE-agent gatherer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_thought_length = config.get('max_thought_length', 500)
        self.enable_completion_detection = config.get('enable_completion_detection', True)
        self.min_iterations = config.get('min_iterations', 3)
        
        # ReAct history
        self._react_history: List[ReActStep] = []
        
        # Current context being built
        self._current_files: List[str] = []
        self._current_classes: List[str] = []
        self._current_methods: List[str] = []
    
    def explore(self, instance: SWEInstance) -> None:
        """
        Run SWE-agent style ReAct exploration.
        
        Implements the ReAct loop:
        1. Think about what information is needed
        2. Choose an action
        3. Execute and observe
        4. Repeat
        
        Args:
            instance: The SWE-bench instance to explore
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized")
        
        state = self.environment.state
        
        # Initial thought
        thought = self._generate_initial_thought(instance)
        
        # ReAct loop
        while not self._should_stop(state):
            # Decide on action
            action = self._decide_action(thought, state, instance)
            
            # Execute action
            observation = self.environment.execute_action(action)
            state.increment_iteration()
            
            # Record step
            step = ReActStep(
                iteration=state.current_iteration,
                thought=thought,
                action=action,
                observation=observation
            )
            self._react_history.append(step)
            
            # Process observation and update context
            self._process_observation(observation, action, state)
            
            # Generate next thought
            thought = self._generate_thought(observation, state, instance)
        
        # Mark complete
        reason = "Max iterations reached" if state.is_at_iteration_limit() else "Context gathering complete"
        state.mark_complete(reason)
    
    def _generate_initial_thought(self, instance: SWEInstance) -> str:
        """
        Generate the initial thought for exploration.
        
        Args:
            instance: The SWE-bench instance
            
        Returns:
            Initial thought string
        """
        problem = instance.problem_statement[:200]
        
        thought = (
            f"I need to understand the issue: {problem}...\n"
            f"Let me start by exploring the repository structure and searching for "
            f"relevant files, classes, and methods related to this problem."
        )
        
        return thought[:self.max_thought_length]
    
    def _generate_thought(
        self,
        observation: Observation,
        state: AgentState,
        instance: SWEInstance
    ) -> str:
        """
        Generate a thought based on the observation.
        
        In a real implementation, this would use an LLM to generate thoughts.
        Here we use a rule-based approach for demonstration.
        
        Args:
            observation: The last observation
            state: Current agent state
            instance: The SWE-bench instance
            
        Returns:
            Thought string
        """
        # Simple rule-based thought generation
        if not observation.success:
            return f"The last action failed with error: {observation.error}. Let me try a different approach."
        
        # Check what we found so far
        files_viewed = len(state.viewed_files)
        searches_done = len(state.search_history)
        
        if files_viewed == 0 and searches_done < 3:
            return "I should search for relevant files using keywords from the problem statement."
        
        if files_viewed < 3:
            return "I've found some relevant files. Let me view their contents to understand the code better."
        
        if len(state.found_symbols['classes']) < 2:
            return "I should search for relevant classes that might be related to this issue."
        
        if len(state.found_symbols['methods']) < 3:
            return "Let me search for specific methods that might need to be modified."
        
        return "I have gathered sufficient context about the issue. I can now proceed with understanding the problem."
    
    def _decide_action(
        self,
        thought: str,
        state: AgentState,
        instance: SWEInstance
    ) -> AgentAction:
        """
        Decide on the next action based on thought and state.
        
        In a real implementation, this would use an LLM to decide.
        Here we use a rule-based approach for demonstration.
        
        Args:
            thought: Current thought
            state: Current agent state
            instance: The SWE-bench instance
            
        Returns:
            AgentAction to execute
        """
        iteration = state.current_iteration
        
        # First few iterations: search for keywords
        if iteration < 3:
            keywords = self._extract_keywords(instance.problem_statement)
            if keywords:
                keyword = keywords[iteration % len(keywords)]
                return AgentAction(
                    tool_name='grep',
                    parameters={'pattern': keyword, 'max_results': 10}
                )
        
        # Next: view files that were found
        if iteration < 6:
            # Get files from search history
            files_to_view = self._get_files_from_search_history(state)
            if files_to_view:
                file_path = files_to_view[iteration % len(files_to_view)]
                return AgentAction(
                    tool_name='view_file',
                    parameters={'path': file_path, 'limit': 100}
                )
        
        # Search for classes
        if iteration < 9:
            keywords = self._extract_keywords(instance.problem_statement)
            class_keywords = [k for k in keywords if k and k[0].isupper()]
            if class_keywords:
                class_name = class_keywords[iteration % len(class_keywords)]
                return AgentAction(
                    tool_name='search_class',
                    parameters={'class_name': class_name, 'max_results': 5}
                )
        
        # Search for methods
        if iteration < 12:
            keywords = self._extract_keywords(instance.problem_statement)
            method_keywords = [k for k in keywords if '_' in k or (k and k[0].islower())]
            if method_keywords:
                method_name = method_keywords[iteration % len(method_keywords)]
                return AgentAction(
                    tool_name='search_method',
                    parameters={'method_name': method_name, 'max_results': 5}
                )
        
        # Get class hierarchy for found classes
        if iteration < 15 and state.found_symbols['classes']:
            class_name = state.found_symbols['classes'][iteration % len(state.found_symbols['classes'])]
            return AgentAction(
                tool_name='get_class_hierarchy',
                parameters={'class_name': class_name, 'include_ancestors': True}
            )
        
        # Default: find more files
        return AgentAction(
            tool_name='find_file',
            parameters={'pattern': '.py', 'max_results': 10}
        )
    
    def _process_observation(
        self,
        observation: Observation,
        action: AgentAction,
        state: AgentState
    ) -> None:
        """
        Process an observation and update context.
        
        Args:
            observation: The observation to process
            action: The action that produced the observation
            state: Current agent state
        """
        if not observation.success:
            return
        
        tool_name = action.tool_name
        metadata = observation.metadata or {}
        
        # Process based on tool type
        if tool_name == 'view_file':
            file_path = metadata.get('file_path', '')
            if file_path:
                state.record_file_view(
                    file_path,
                    metadata.get('offset', 0),
                    metadata.get('offset', 0) + metadata.get('limit', 100)
                )
                
                # Add as context chunk
                chunk = ContextChunk(
                    content=observation.output,
                    source_file=file_path,
                    context_type=ContextType.FILE_CONTENT,
                    start_line=metadata.get('offset', 0),
                    end_line=metadata.get('offset', 0) + metadata.get('limit', 100),
                    relevance_score=0.7,
                    metadata={'gather_method': 'swe_agent_view'}
                )
                state.add_context_chunk(chunk)
        
        elif tool_name == 'grep':
            results = metadata.get('results', [])
            files = list(set(r.get('file', '') for r in results if r.get('file')))
            state.record_search('grep', metadata.get('pattern', ''), files)
        
        elif tool_name == 'search_class':
            results = metadata.get('results', [])
            for result in results:
                class_name = result.get('class_name', '')
                if class_name and class_name not in state.found_symbols['classes']:
                    state.found_symbols['classes'].append(class_name)
                
                # Add class info as context
                file_path = result.get('file', '')
                if file_path:
                    chunk = ContextChunk(
                        content=f"Class {class_name} found in {file_path}:{result.get('line', 0)}",
                        source_file=file_path,
                        context_type=ContextType.CLASS_DEFINITION,
                        start_line=result.get('line', 0),
                        end_line=result.get('end_line', 0),
                        relevance_score=0.8,
                        metadata={
                            'class_name': class_name,
                            'bases': result.get('bases', []),
                            'gather_method': 'swe_agent_search'
                        }
                    )
                    state.add_context_chunk(chunk)
        
        elif tool_name == 'search_method':
            results = metadata.get('results', [])
            for result in results:
                method_name = result.get('method_name', '')
                if method_name and method_name not in state.found_symbols['methods']:
                    state.found_symbols['methods'].append(method_name)
                
                # Add method info as context
                file_path = result.get('file', '')
                if file_path:
                    chunk = ContextChunk(
                        content=f"Method {method_name} found in {file_path}:{result.get('line', 0)}",
                        source_file=file_path,
                        context_type=ContextType.FUNCTION_DEFINITION,
                        start_line=result.get('line', 0),
                        end_line=result.get('end_line', 0),
                        relevance_score=0.85,
                        metadata={
                            'method_name': method_name,
                            'class_name': result.get('class_name'),
                            'args': result.get('args', []),
                            'gather_method': 'swe_agent_search'
                        }
                    )
                    state.add_context_chunk(chunk)
        
        elif tool_name == 'get_class_hierarchy':
            class_name = metadata.get('class_name', '')
            if class_name:
                chunk = ContextChunk(
                    content=observation.output,
                    source_file='hierarchy',
                    context_type=ContextType.REPO_STRUCTURE,
                    start_line=0,
                    end_line=0,
                    relevance_score=0.6,
                    metadata={
                        'class_name': class_name,
                        'ancestors': metadata.get('ancestors', []),
                        'gather_method': 'swe_agent_hierarchy'
                    }
                )
                state.add_context_chunk(chunk)
    
    def _should_stop(self, state: AgentState) -> bool:
        """
        Determine if exploration should stop.
        
        Args:
            state: Current agent state
            
        Returns:
            True if exploration should stop
        """
        # Always respect iteration limit
        if state.is_at_iteration_limit():
            return True
        
        # Minimum iterations
        if state.current_iteration < self.min_iterations:
            return False
        
        # Check if we have enough context
        if self.enable_completion_detection:
            # Have we viewed enough files?
            if len(state.viewed_files) >= 3:
                # Have we found relevant symbols?
                if len(state.found_symbols['classes']) >= 1:
                    if len(state.found_symbols['methods']) >= 2:
                        return True
        
        return False
    
    def _extract_keywords(self, problem_statement: str) -> List[str]:
        """
        Extract keywords from problem statement.
        
        Args:
            problem_statement: The problem description
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Find CamelCase words (likely class names)
        camel_case = re.findall(r'\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b', problem_statement)
        keywords.extend(camel_case)
        
        # Find snake_case words
        snake_case = re.findall(r'\b[a-z_][a-z0-9_]*\b', problem_statement)
        keywords.extend([w for w in snake_case if '_' in w or len(w) > 4])
        
        # Remove duplicates
        seen = set()
        unique = []
        for kw in keywords:
            if kw.lower() not in seen and len(kw) > 2:
                seen.add(kw.lower())
                unique.append(kw)
        
        return unique[:10]
    
    def _get_files_from_search_history(self, state: AgentState) -> List[str]:
        """Extract unique files from search history."""
        files = []
        for search in state.search_history:
            for result in search.get('results', []):
                if isinstance(result, str) and result not in files:
                    files.append(result)
        return files
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about context gathering."""
        stats = super().get_stats()
        stats.update({
            'react_steps': len(self._react_history),
            'files_viewed': len(self.environment.state.viewed_files) if self.environment else 0,
            'classes_found': len(self.environment.state.found_symbols['classes']) if self.environment else 0,
            'methods_found': len(self.environment.state.found_symbols['methods']) if self.environment else 0,
            'pattern': 'ReAct'
        })
        return stats
