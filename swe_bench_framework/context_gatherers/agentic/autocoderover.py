"""
AutoCodeRover-style agentic context gatherer.

This module implements the AutoCodeRover approach to context gathering,
featuring:
- AST-based code search
- Stratified retrieval (file → class → function)
- Two-phase exploration: localization then context refinement
- Spectrum-based fault localization (SBFL) when tests available

Reference: https://arxiv.org/abs/2404.05427
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import BaseAgenticGatherer
from .environment import AgentAction, Observation, AgentState
from .tools.search_tools import ASTAnalyzer
from ..base import ContextChunk, ContextType, SWEInstance


class AutoCodeRoverGatherer(BaseAgenticGatherer):
    """
    AutoCodeRover-style context gatherer.
    
    Implements stratified retrieval where the agent:
    1. First searches for relevant files using keywords from the problem
    2. Then searches for classes within those files
    3. Finally searches for specific methods/functions
    
    This hierarchical approach helps focus on the most relevant code.
    
    Attributes:
        config: Configuration dictionary with keys:
            - max_iterations: Maximum exploration iterations
            - max_files_per_layer: Max files to consider at file layer
            - max_classes_per_layer: Max classes to consider at class layer
            - max_functions_per_layer: Max functions at function layer
            - use_sbfl: Whether to use spectrum-based fault localization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoCodeRover gatherer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_files_per_layer = config.get('max_files_per_layer', 5)
        self.max_classes_per_layer = config.get('max_classes_per_layer', 10)
        self.max_functions_per_layer = config.get('max_functions_per_layer', 20)
        self.use_sbfl = config.get('use_sbfl', True)
        
        # Tracking for stratified retrieval
        self._suspicious_files: List[str] = []
        self._suspicious_classes: List[str] = []
        self._suspicious_functions: List[str] = []
        self._keywords: List[str] = []
    
    def explore(self, instance: SWEInstance) -> None:
        """
        Run AutoCodeRover-style exploration.
        
        Implements the two-phase stratified retrieval:
        1. Localization phase: Identify suspicious files, classes, functions
        2. Context refinement: Gather detailed context for identified items
        
        Args:
            instance: The SWE-bench instance to explore
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized")
        
        # Phase 1: Extract keywords from problem statement
        self._keywords = self._extract_keywords(instance.problem_statement)
        
        # Phase 2: File-level localization
        self._localize_files(instance)
        
        # Phase 3: Class-level localization
        self._localize_classes()
        
        # Phase 4: Function-level localization
        self._localize_functions()
        
        # Phase 5: Gather detailed context
        self._gather_detailed_context(instance)
        
        # Mark exploration complete
        self.environment.state.mark_complete("Stratified retrieval complete")
    
    def _extract_keywords(self, problem_statement: str) -> List[str]:
        """
        Extract search keywords from problem statement.
        
        Uses simple heuristics to identify:
        - CamelCase class names
        - snake_case function names
        - Quoted strings
        - Error messages
        
        Args:
            problem_statement: The problem description
            
        Returns:
            List of extracted keywords
        """
        keywords = []
        
        # Find CamelCase words (likely class names)
        camel_case = re.findall(r'\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b', problem_statement)
        keywords.extend(camel_case)
        
        # Find snake_case words that look like functions
        snake_case = re.findall(r'\b[a-z_][a-z0-9_]*\b', problem_statement)
        # Filter for likely function names (contain underscores or are verbs)
        likely_functions = [w for w in snake_case if '_' in w or len(w) > 5]
        keywords.extend(likely_functions)
        
        # Find quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', problem_statement)
        keywords.extend(quoted)
        
        # Find error messages (lines with "Error" or "Exception")
        error_lines = re.findall(r'(?:Error|Exception|Traceback)[^\n]*', problem_statement)
        for line in error_lines:
            # Extract function/file references from tracebacks
            file_refs = re.findall(r'File "([^"]+)"[^\n]*', line)
            for ref in file_refs:
                # Extract just the filename
                keywords.append(ref.split('/')[-1].replace('.py', ''))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and len(kw) > 2:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords[:15]  # Limit to top 15 keywords
    
    def _localize_files(self, instance: SWEInstance) -> None:
        """
        Phase 1: Identify suspicious files.
        
        Uses keyword search and test information to find relevant files.
        
        Args:
            instance: The SWE-bench instance
        """
        state = self.environment.state
        file_scores: Dict[str, float] = {}
        
        # Search for each keyword
        for keyword in self._keywords:
            # Grep for keyword
            obs = self.execute_tool('grep', pattern=keyword, max_results=20)
            state.increment_iteration()
            
            if obs.success and obs.metadata:
                results = obs.metadata.get('results', [])
                for result in results:
                    file_path = result.get('file', '')
                    if file_path:
                        # Score based on keyword match
                        score = 1.0
                        # Boost score for exact matches
                        if keyword.lower() in file_path.lower():
                            score += 0.5
                        file_scores[file_path] = file_scores.get(file_path, 0) + score
        
        # If we have test information, use it
        if instance.failed_tests and self.use_sbfl:
            self._apply_test_based_localization(instance, file_scores)
        
        # Sort by score and take top files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        self._suspicious_files = [f for f, _ in sorted_files[:self.max_files_per_layer]]
        
        # Record in state
        for file_path in self._suspicious_files:
            state.record_search('file_localization', 'keyword_search', [file_path])
    
    def _apply_test_based_localization(
        self, 
        instance: SWEInstance,
        file_scores: Dict[str, float]
    ) -> None:
        """
        Apply spectrum-based fault localization using test info.
        
        Boosts scores for files mentioned in failing tests.
        
        Args:
            instance: The SWE-bench instance
            file_scores: Dictionary of file scores to modify
        """
        # Look for file references in test names and error traces
        for test in instance.failed_tests:
            # Extract potential file names from test names
            parts = test.split('.')
            for part in parts:
                if '_' in part or part[0].isupper():
                    # Search for files matching this part
                    obs = self.execute_tool('find_file', pattern=part, max_results=5)
                    if obs.success and obs.metadata:
                        for file_path in obs.metadata.get('results', []):
                            # Boost score significantly for test-related files
                            file_scores[file_path] = file_scores.get(file_path, 0) + 2.0
    
    def _localize_classes(self) -> None:
        """
        Phase 2: Identify suspicious classes within files.
        
        Searches for classes mentioned in keywords or in suspicious files.
        """
        state = self.environment.state
        class_scores: Dict[str, Dict[str, Any]] = {}
        
        # Search for classes matching keywords
        for keyword in self._keywords:
            # Check if keyword looks like a class name
            if keyword[0].isupper():
                obs = self.execute_tool('search_class', class_name=keyword, max_results=5)
                state.increment_iteration()
                
                if obs.success and obs.metadata:
                    results = obs.metadata.get('results', [])
                    for result in results:
                        class_name = result.get('class_name', '')
                        file_path = result.get('file', '')
                        key = f"{file_path}::{class_name}"
                        
                        # Score based on relevance
                        score = 2.0 if keyword.lower() == class_name.lower() else 1.0
                        
                        # Boost if in suspicious file
                        if file_path in self._suspicious_files:
                            score += 1.0
                        
                        class_scores[key] = {
                            'class_name': class_name,
                            'file': file_path,
                            'line': result.get('line', 0),
                            'score': class_scores.get(key, {}).get('score', 0) + score
                        }
        
        # Also search classes in suspicious files
        for file_path in self._suspicious_files[:3]:  # Limit to top 3 files
            resolved = f"{self.repo_path}/{file_path}"
            try:
                import ast
                with open(resolved, 'r', encoding='utf-8', errors='ignore') as f:
                    tree = ast.parse(f.read())
                
                classes = ASTAnalyzer.find_classes(tree)
                for cls in classes:
                    class_name = cls['name']
                    key = f"{file_path}::{class_name}"
                    
                    # Check if class name matches any keyword
                    score = 0.5
                    for kw in self._keywords:
                        if kw.lower() in class_name.lower():
                            score += 1.0
                    
                    class_scores[key] = {
                        'class_name': class_name,
                        'file': file_path,
                        'line': cls['line'],
                        'score': class_scores.get(key, {}).get('score', 0) + score
                    }
            except Exception:
                pass
        
        # Sort and select top classes
        sorted_classes = sorted(
            class_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        self._suspicious_classes = [
            f"{info['file']}::{info['class_name']}" 
            for _, info in sorted_classes[:self.max_classes_per_layer]
        ]
    
    def _localize_functions(self) -> None:
        """
        Phase 3: Identify suspicious functions/methods.
        
        Searches for functions in suspicious classes and files.
        """
        state = self.environment.state
        function_scores: Dict[str, Dict[str, Any]] = {}
        
        # Search for functions matching keywords
        for keyword in self._keywords:
            # Check if keyword looks like a function name
            if keyword[0].islower() or '_' in keyword:
                obs = self.execute_tool('search_method', method_name=keyword, max_results=5)
                state.increment_iteration()
                
                if obs.success and obs.metadata:
                    results = obs.metadata.get('results', [])
                    for result in results:
                        func_name = result.get('method_name', '')
                        file_path = result.get('file', '')
                        class_name = result.get('class_name', '')
                        
                        key = f"{file_path}::{class_name or ''}::{func_name}"
                        
                        # Score based on relevance
                        score = 2.0 if keyword.lower() == func_name.lower() else 1.0
                        
                        # Boost if in suspicious file or class
                        if file_path in self._suspicious_files:
                            score += 1.0
                        if class_name and any(cls.endswith(f"::{class_name}") for cls in self._suspicious_classes):
                            score += 1.5
                        
                        function_scores[key] = {
                            'function_name': func_name,
                            'class_name': class_name,
                            'file': file_path,
                            'line': result.get('line', 0),
                            'score': function_scores.get(key, {}).get('score', 0) + score
                        }
        
        # Sort and select top functions
        sorted_functions = sorted(
            function_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )
        self._suspicious_functions = [
            f"{info['file']}::{info.get('class_name', '')}::{info['function_name']}"
            for _, info in sorted_functions[:self.max_functions_per_layer]
        ]
    
    def _gather_detailed_context(self, instance: SWEInstance) -> None:
        """
        Phase 4: Gather detailed context for identified items.
        
        Views the content of suspicious files, classes, and functions.
        
        Args:
            instance: The SWE-bench instance
        """
        state = self.environment.state
        
        # Gather context for suspicious functions (most specific)
        for func_key in self._suspicious_functions[:10]:
            parts = func_key.split('::')
            file_path = parts[0]
            class_name = parts[1] if len(parts) > 2 and parts[1] else None
            func_name = parts[-1]
            
            # View the file around the function
            obs = self.execute_tool('search_method', 
                method_name=func_name,
                class_name=class_name,
                file_path=file_path
            )
            state.increment_iteration()
            
            if obs.success and obs.metadata:
                results = obs.metadata.get('results', [])
                for result in results:
                    line = result.get('line', 0)
                    end_line = result.get('end_line', line + 20)
                    
                    # View the function content
                    view_obs = self.execute_tool('view_file',
                        path=file_path,
                        offset=max(0, line - 5),
                        limit=end_line - line + 10
                    )
                    
                    if view_obs.success:
                        chunk = ContextChunk(
                            content=view_obs.output,
                            source_file=file_path,
                            context_type=ContextType.FUNCTION_DEFINITION,
                            start_line=line,
                            end_line=end_line,
                            relevance_score=0.9,
                            metadata={
                                'function_name': func_name,
                                'class_name': class_name,
                                'gather_method': 'stratified_function'
                            }
                        )
                        state.add_context_chunk(chunk)
        
        # Gather context for suspicious classes
        for class_key in self._suspicious_classes[:5]:
            parts = class_key.split('::')
            file_path = parts[0]
            class_name = parts[1]
            
            # Get class hierarchy
            obs = self.execute_tool('get_class_hierarchy', 
                class_name=class_name,
                include_ancestors=True
            )
            state.increment_iteration()
            
            if obs.success:
                chunk = ContextChunk(
                    content=obs.output,
                    source_file=file_path,
                    context_type=ContextType.CLASS_DEFINITION,
                    start_line=0,
                    end_line=0,
                    relevance_score=0.8,
                    metadata={
                        'class_name': class_name,
                        'gather_method': 'stratified_class'
                    }
                )
                state.add_context_chunk(chunk)
        
        # Gather context for suspicious files
        for file_path in self._suspicious_files[:3]:
            # View the file (first 100 lines)
            obs = self.execute_tool('view_file', path=file_path, offset=0, limit=100)
            state.increment_iteration()
            
            if obs.success:
                chunk = ContextChunk(
                    content=obs.output,
                    source_file=file_path,
                    context_type=ContextType.FILE_CONTENT,
                    start_line=0,
                    end_line=100,
                    relevance_score=0.7,
                    metadata={'gather_method': 'stratified_file'}
                )
                state.add_context_chunk(chunk)
        
        # Record viewed files
        for file_path in self._suspicious_files[:3]:
            state.record_file_view(file_path, 0, 100)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about context gathering."""
        stats = super().get_stats()
        stats.update({
            'keywords_extracted': len(self._keywords),
            'suspicious_files': len(self._suspicious_files),
            'suspicious_classes': len(self._suspicious_classes),
            'suspicious_functions': len(self._suspicious_functions),
            'stratified_layers': ['file', 'class', 'function']
        })
        return stats
