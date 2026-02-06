"""
Agentless-style hierarchical context gatherer.

This module implements the Agentless approach to context gathering,
featuring:
- Simple 3-phase hierarchical localization (file → class/method → edit location)
- No complex agent scaffolding
- Fixed workflow without autonomous decision-making
- Focus on simplicity and reliability

Reference: https://arxiv.org/abs/2407.01489
"""

import ast
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAgenticGatherer
from .environment import AgentAction, Observation, AgentState
from .tools.search_tools import ASTAnalyzer
from ..base import ContextChunk, ContextType, SWEInstance


class AgentlessGatherer(BaseAgenticGatherer):
    """
    Agentless-style hierarchical context gatherer.
    
    Implements a simple three-phase approach:
    1. File selection: Identify suspicious files
    2. Class/method localization: Find relevant classes and methods
    3. Edit location pinpointing: Identify exact edit locations
    
    Unlike autonomous agents, this uses a fixed workflow without
    complex decision-making, which can be more reliable.
    
    Attributes:
        config: Configuration dictionary with keys:
            - max_files: Maximum files to select
            - max_classes_per_file: Maximum classes to extract per file
            - max_methods_per_class: Maximum methods to extract per class
            - context_window_lines: Lines of context around locations
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Agentless gatherer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.max_files = config.get('max_files', 5)
        self.max_classes_per_file = config.get('max_classes_per_file', 3)
        self.max_methods_per_class = config.get('max_methods_per_class', 5)
        self.context_window_lines = config.get('context_window_lines', 10)
        
        # Results from each phase
        self._selected_files: List[Dict[str, Any]] = []
        self._located_symbols: List[Dict[str, Any]] = []
        self._edit_locations: List[Dict[str, Any]] = []
    
    def explore(self, instance: SWEInstance) -> None:
        """
        Run Agentless-style hierarchical exploration.
        
        Executes the three-phase workflow:
        1. Phase 1: File selection based on keywords
        2. Phase 2: Class/method localization within selected files
        3. Phase 3: Pinpoint exact edit locations
        
        Args:
            instance: The SWE-bench instance to explore
        """
        if self.environment is None:
            raise RuntimeError("Environment not initialized")
        
        # Phase 1: File selection
        self._phase1_file_selection(instance)
        
        # Phase 2: Class/method localization
        self._phase2_symbol_localization()
        
        # Phase 3: Edit location pinpointing
        self._phase3_edit_location_pinpointing(instance)
        
        # Mark complete
        self.environment.state.mark_complete("Hierarchical localization complete")
    
    def _phase1_file_selection(self, instance: SWEInstance) -> None:
        """
        Phase 1: Select suspicious files.
        
        Uses keyword matching from the problem statement to identify
        files that are likely relevant to the issue.
        
        Args:
            instance: The SWE-bench instance
        """
        state = self.environment.state
        
        # Extract keywords from problem statement
        keywords = self._extract_keywords(instance.problem_statement)
        
        # Score files based on keyword matches
        file_scores: Dict[str, float] = {}
        
        for keyword in keywords:
            # Search for keyword in files
            obs = self.execute_tool('grep', pattern=keyword, max_results=30)
            state.increment_iteration()
            
            if obs.success and obs.metadata:
                results = obs.metadata.get('results', [])
                for result in results:
                    file_path = result.get('file', '')
                    if not file_path:
                        continue
                    
                    # Calculate score
                    score = 1.0
                    
                    # Boost if keyword appears in filename
                    filename = os.path.basename(file_path).replace('.py', '')
                    if keyword.lower() in filename.lower():
                        score += 2.0
                    
                    # Boost for exact keyword match in content
                    content = result.get('content', '').lower()
                    if keyword.lower() in content:
                        score += 0.5
                    
                    file_scores[file_path] = file_scores.get(file_path, 0) + score
        
        # Also check test files for hints
        if instance.failed_tests:
            for test in instance.failed_tests[:5]:
                # Extract potential module names from test names
                parts = test.split('.')
                for part in parts:
                    if len(part) > 3:
                        obs = self.execute_tool('find_file', pattern=part, max_results=5)
                        if obs.success and obs.metadata:
                            for file_path in obs.metadata.get('results', []):
                                file_scores[file_path] = file_scores.get(file_path, 0) + 1.5
        
        # Sort and select top files
        sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        
        self._selected_files = [
            {
                'path': file_path,
                'score': score,
                'keywords_matched': self._get_matching_keywords(file_path, keywords)
            }
            for file_path, score in sorted_files[:self.max_files]
        ]
        
        # Record in state
        for file_info in self._selected_files:
            state.record_search('file_selection', 'keyword_match', [file_info['path']])
    
    def _phase2_symbol_localization(self) -> None:
        """
        Phase 2: Localize classes and methods within selected files.
        
        Parses selected files and extracts class and method definitions,
        scoring them based on relevance to the problem.
        """
        state = self.environment.state
        
        for file_info in self._selected_files:
            file_path = file_info['path']
            resolved_path = os.path.join(self.repo_path, file_path)
            
            if not os.path.exists(resolved_path):
                continue
            
            try:
                # Parse the file
                with open(resolved_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                tree = ast.parse(content)
                
                # Find classes
                classes = ASTAnalyzer.find_classes(tree)
                
                for cls in classes[:self.max_classes_per_file]:
                    class_name = cls['name']
                    
                    # Score class based on keyword matches
                    class_score = self._score_symbol(class_name, cls['methods'])
                    
                    symbol_info = {
                        'type': 'class',
                        'name': class_name,
                        'file': file_path,
                        'line': cls['line'],
                        'end_line': cls['end_line'],
                        'score': class_score + file_info['score'],
                        'methods': []
                    }
                    
                    # Extract methods for this class
                    for method in cls['methods'][:self.max_methods_per_class]:
                        method_score = self._score_symbol(method['name'], [])
                        symbol_info['methods'].append({
                            'name': method['name'],
                            'line': method['line'],
                            'end_line': method['end_line'],
                            'score': method_score
                        })
                    
                    self._located_symbols.append(symbol_info)
                    state.found_symbols['classes'].append(class_name)
                
                # Find standalone functions
                functions = ASTAnalyzer.find_functions(tree)
                for func in functions[:self.max_methods_per_class]:
                    func_score = self._score_symbol(func['name'], [])
                    
                    self._located_symbols.append({
                        'type': 'function',
                        'name': func['name'],
                        'file': file_path,
                        'line': func['line'],
                        'end_line': func['end_line'],
                        'score': func_score + file_info['score'],
                        'methods': []
                    })
                    state.found_symbols['functions'].append(func['name'])
                
                state.increment_iteration()
                
            except SyntaxError:
                # Skip files with syntax errors
                continue
            except Exception:
                continue
        
        # Sort symbols by score
        self._located_symbols.sort(key=lambda x: x['score'], reverse=True)
    
    def _phase3_edit_location_pinpointing(self, instance: SWEInstance) -> None:
        """
        Phase 3: Pinpoint exact edit locations.
        
        Analyzes the located symbols to identify the most likely
        edit locations based on the problem statement.
        
        Args:
            instance: The SWE-bench instance
        """
        state = self.environment.state
        
        # Get top symbols
        top_symbols = self._located_symbols[:15]
        
        for symbol in top_symbols:
            file_path = symbol['file']
            
            # View the symbol's code
            start_line = max(0, symbol['line'] - self.context_window_lines)
            end_line = symbol['end_line'] + self.context_window_lines
            limit = end_line - start_line
            
            obs = self.execute_tool('view_file',
                path=file_path,
                offset=start_line,
                limit=limit
            )
            state.increment_iteration()
            
            if obs.success:
                # Create context chunk
                context_type = (
                    ContextType.CLASS_DEFINITION 
                    if symbol['type'] == 'class' 
                    else ContextType.FUNCTION_DEFINITION
                )
                
                chunk = ContextChunk(
                    content=obs.output,
                    source_file=file_path,
                    context_type=context_type,
                    start_line=symbol['line'],
                    end_line=symbol['end_line'],
                    relevance_score=min(1.0, symbol['score'] / 5.0),
                    metadata={
                        'symbol_name': symbol['name'],
                        'symbol_type': symbol['type'],
                        'score': symbol['score'],
                        'gather_method': 'agentless_hierarchical'
                    }
                )
                state.add_context_chunk(chunk)
                state.record_file_view(file_path, start_line, end_line)
                
                # Store edit location
                self._edit_locations.append({
                    'file': file_path,
                    'symbol_name': symbol['name'],
                    'symbol_type': symbol['type'],
                    'start_line': symbol['line'],
                    'end_line': symbol['end_line'],
                    'score': symbol['score']
                })
        
        # Add file-level context for top files
        for file_info in self._selected_files[:3]:
            file_path = file_info['path']
            
            # View first part of file (imports and docstring)
            obs = self.execute_tool('view_file', path=file_path, offset=0, limit=50)
            
            if obs.success:
                chunk = ContextChunk(
                    content=obs.output,
                    source_file=file_path,
                    context_type=ContextType.FILE_CONTENT,
                    start_line=0,
                    end_line=50,
                    relevance_score=0.5,
                    metadata={'gather_method': 'agentless_file_header'}
                )
                state.add_context_chunk(chunk)
    
    def _extract_keywords(self, problem_statement: str) -> List[str]:
        """
        Extract keywords from problem statement.
        
        Args:
            problem_statement: The problem description
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Extract CamelCase (class names)
        camel_case = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', problem_statement)
        keywords.extend(camel_case)
        
        # Extract snake_case (function/variable names)
        snake_case = re.findall(r'\b[a-z_][a-z0-9_]*\b', problem_statement)
        keywords.extend([w for w in snake_case if len(w) > 3])
        
        # Extract quoted strings
        quoted = re.findall(r'["\']([^"\']+)["\']', problem_statement)
        keywords.extend(quoted)
        
        # Extract error-related words
        error_words = re.findall(r'\b(?:error|exception|fail|bug|issue|problem)[s]?\b', 
                                 problem_statement, re.IGNORECASE)
        keywords.extend(error_words)
        
        # Remove duplicates and short words
        seen = set()
        unique = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen and len(kw) > 2:
                seen.add(kw_lower)
                unique.append(kw)
        
        return unique[:20]
    
    def _get_matching_keywords(self, file_path: str, keywords: List[str]) -> List[str]:
        """Get keywords that match the file path."""
        filename = os.path.basename(file_path).lower()
        return [kw for kw in keywords if kw.lower() in filename]
    
    def _score_symbol(self, name: str, methods: List[Dict]) -> float:
        """
        Score a symbol based on various heuristics.
        
        Args:
            name: Symbol name
            methods: List of methods (for classes)
            
        Returns:
            Relevance score
        """
        score = 1.0
        
        # Boost for descriptive names
        if '_' in name:
            score += 0.5
        
        # Boost for longer names (more descriptive)
        if len(name) > 10:
            score += 0.3
        
        # Boost for classes with methods
        score += len(methods) * 0.1
        
        # Boost for common method patterns
        method_names = [m['name'] if isinstance(m, dict) else m for m in methods]
        important_patterns = ['handle', 'process', 'validate', 'parse', 'render', 
                             'compute', 'calculate', 'get_', 'set_', 'update_']
        for pattern in important_patterns:
            if any(pattern in mn for mn in method_names):
                score += 0.2
        
        return score
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about context gathering."""
        stats = super().get_stats()
        stats.update({
            'selected_files': len(self._selected_files),
            'located_symbols': len(self._located_symbols),
            'edit_locations': len(self._edit_locations),
            'phases': ['file_selection', 'symbol_localization', 'edit_pinpointing'],
            'pattern': 'hierarchical_fixed_workflow'
        })
        return stats
