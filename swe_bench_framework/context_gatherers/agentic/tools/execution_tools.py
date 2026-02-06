"""
Execution tools for agentic exploration.

This module provides tools for running tests and linting code,
enabling agents to validate their understanding of the codebase.
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional

from .base import ExecutionTool, validate_parameters
from ..environment import Observation


class RunTestTool(ExecutionTool):
    """
    Tool for running tests in the repository.
    
    Supports pytest, unittest, and other common Python test frameworks.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="run_test",
            description="Run tests in the repository",
            repo_path=repo_path
        )
        self._test_framework: Optional[str] = None
    
    def _detect_test_framework(self) -> str:
        """Detect the test framework used in the repository."""
        if self._test_framework:
            return self._test_framework
        
        # Check for pytest
        if os.path.exists(os.path.join(self.repo_path, 'pytest.ini')):
            self._test_framework = 'pytest'
            return 'pytest'
        
        if os.path.exists(os.path.join(self.repo_path, 'setup.cfg')):
            with open(os.path.join(self.repo_path, 'setup.cfg'), 'r') as f:
                if '[tool:pytest]' in f.read():
                    self._test_framework = 'pytest'
                    return 'pytest'
        
        # Check for pyproject.toml with pytest
        if os.path.exists(os.path.join(self.repo_path, 'pyproject.toml')):
            with open(os.path.join(self.repo_path, 'pyproject.toml'), 'r') as f:
                content = f.read()
                if '[tool.pytest]' in content or 'pytest' in content:
                    self._test_framework = 'pytest'
                    return 'pytest'
        
        # Check for tox
        if os.path.exists(os.path.join(self.repo_path, 'tox.ini')):
            self._test_framework = 'tox'
            return 'tox'
        
        # Default to pytest
        self._test_framework = 'pytest'
        return 'pytest'
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Run tests.
        
        Parameters:
            - test_path: Path to test file or directory (default: auto-detect)
            - test_name: Specific test name to run (optional)
            - verbose: Show verbose output (default: True)
            - timeout: Timeout in seconds (default: 120)
            
        Returns:
            Observation with test results
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=[],
            optional={
                'test_path': None,
                'test_name': None,
                'verbose': True,
                'timeout': 120
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        test_path = params['test_path']
        test_name = params['test_name']
        verbose = params['verbose']
        timeout = params['timeout']
        
        try:
            framework = self._detect_test_framework()
            
            # Build command
            if framework == 'pytest':
                command = [sys.executable, '-m', 'pytest']
                if verbose:
                    command.append('-v')
                if test_path:
                    command.append(test_path)
                if test_name:
                    command.extend(['-k', test_name])
            elif framework == 'tox':
                command = ['tox']
                if test_name:
                    command.extend(['-e', test_name])
            else:
                command = [sys.executable, '-m', 'unittest']
                if verbose:
                    command.append('-v')
                if test_path:
                    command.append(test_path)
            
            stdout, stderr, return_code = self._run_command(
                command, 
                timeout=timeout,
                cwd=self.repo_path
            )
            
            # Parse results
            passed, failed, errors = self._parse_test_results(stdout + stderr)
            
            success = return_code == 0
            output = stdout if stdout else stderr
            
            # Truncate if too long
            if len(output) > 5000:
                output = output[:2500] + "\n... [output truncated] ...\n" + output[-2500:]
            
            return Observation(
                output=output or "Tests completed (no output)",
                error=stderr if not success else None,
                success=success,
                metadata={
                    'framework': framework,
                    'return_code': return_code,
                    'tests_passed': passed,
                    'tests_failed': failed,
                    'test_errors': errors
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error running tests: {str(e)}",
                success=False
            )
    
    def _parse_test_results(self, output: str) -> tuple:
        """Parse test output to count passed/failed tests."""
        passed = 0
        failed = 0
        errors = 0
        
        # Look for pytest summary
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        error_match = re.search(r'(\d+) error', output)
        
        if passed_match:
            passed = int(passed_match.group(1))
        if failed_match:
            failed = int(failed_match.group(1))
        if error_match:
            errors = int(error_match.group(1))
        
        return passed, failed, errors
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'test_path': {
                    'type': 'string',
                    'description': 'Path to test file or directory',
                    'default': None
                },
                'test_name': {
                    'type': 'string',
                    'description': 'Specific test name to run',
                    'default': None
                },
                'verbose': {
                    'type': 'boolean',
                    'description': 'Show verbose output',
                    'default': True
                },
                'timeout': {
                    'type': 'integer',
                    'description': 'Timeout in seconds',
                    'default': 120
                }
            },
            'required': []
        }


class LinterTool(ExecutionTool):
    """
    Tool for running Python linters on code.
    
    Supports flake8, pylint, and pyright. Falls back to py_compile
    for basic syntax checking.
    """
    
    def __init__(self, repo_path: str):
        super().__init__(
            name="linter",
            description="Run Python linter to check code syntax and style",
            repo_path=repo_path
        )
    
    def _detect_linter(self) -> str:
        """Detect which linter is available."""
        # Check for project-specific config
        if os.path.exists(os.path.join(self.repo_path, '.flake8')):
            return 'flake8'
        if os.path.exists(os.path.join(self.repo_path, 'pylintrc')):
            return 'pylint'
        if os.path.exists(os.path.join(self.repo_path, 'pyproject.toml')):
            with open(os.path.join(self.repo_path, 'pyproject.toml'), 'r') as f:
                content = f.read()
                if 'flake8' in content:
                    return 'flake8'
                if 'pylint' in content:
                    return 'pylint'
        
        # Default to py_compile (always available)
        return 'py_compile'
    
    def execute(self, parameters: Dict[str, Any]) -> Observation:
        """
        Run linter on a file.
        
        Parameters:
            - file_path: Path to file to lint (required)
            - linter: Linter to use (default: auto-detect)
            - max_issues: Maximum issues to report (default: 20)
            
        Returns:
            Observation with linting results
        """
        is_valid, error, params = validate_parameters(
            parameters,
            required=['file_path'],
            optional={
                'linter': None,
                'max_issues': 20
            }
        )
        
        if not is_valid:
            return Observation(error=error, success=False)
        
        file_path = params['file_path']
        linter = params['linter'] or self._detect_linter()
        max_issues = params['max_issues']
        
        try:
            resolved = os.path.join(self.repo_path, file_path)
            
            if not os.path.exists(resolved):
                return Observation(
                    error=f"File not found: {file_path}",
                    success=False
                )
            
            if os.path.isdir(resolved):
                return Observation(
                    error=f"{file_path} is a directory",
                    success=False
                )
            
            # Run linter
            if linter == 'flake8':
                command = [sys.executable, '-m', 'flake8', '--max-line-length=100', resolved]
            elif linter == 'pylint':
                command = [sys.executable, '-m', 'pylint', '--disable=R,C', resolved]
            else:
                # Use py_compile for syntax checking
                return self._py_compile_check(resolved)
            
            stdout, stderr, return_code = self._run_command(
                command,
                timeout=30,
                cwd=self.repo_path
            )
            
            # Parse issues
            issues = self._parse_linter_output(stdout + stderr, linter)
            
            success = len(issues) == 0
            
            # Format output
            if not issues:
                output = f"No issues found in {file_path}"
            else:
                lines = [f"Found {len(issues)} issues in {file_path}:"]
                for issue in issues[:max_issues]:
                    lines.append(f"  Line {issue.get('line', '?')}: {issue.get('message', '')}")
                if len(issues) > max_issues:
                    lines.append(f"  ... and {len(issues) - max_issues} more issues")
                output = "\n".join(lines)
            
            return Observation(
                output=output,
                success=success,
                metadata={
                    'file_path': file_path,
                    'linter': linter,
                    'issues_count': len(issues),
                    'issues': issues[:max_issues]
                }
            )
            
        except Exception as e:
            return Observation(
                error=f"Error running linter: {str(e)}",
                success=False
            )
    
    def _py_compile_check(self, file_path: str) -> Observation:
        """Check Python syntax using py_compile."""
        import py_compile
        
        try:
            py_compile.compile(file_path, doraise=True)
            return Observation(
                output=f"Syntax check passed for {os.path.basename(file_path)}",
                success=True,
                metadata={'file_path': file_path, 'issues_count': 0}
            )
        except py_compile.PyCompileError as e:
            return Observation(
                output=f"Syntax error in {os.path.basename(file_path)}: {e}",
                error=str(e),
                success=False,
                metadata={
                    'file_path': file_path,
                    'issues_count': 1,
                    'issues': [{'line': getattr(e, 'lineno', 0), 'message': str(e)}]
                }
            )
    
    def _parse_linter_output(self, output: str, linter: str) -> List[Dict[str, Any]]:
        """Parse linter output to extract issues."""
        issues = []
        
        if linter == 'flake8':
            # Parse flake8 output: file:line:col: code message
            pattern = r'^(.+?):(\d+):(\d+):\s*(\w+)\s*(.+)$'
            for line in output.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    issues.append({
                        'file': match.group(1),
                        'line': int(match.group(2)),
                        'column': int(match.group(3)),
                        'code': match.group(4),
                        'message': match.group(5)
                    })
        
        elif linter == 'pylint':
            # Parse pylint output: file:line:col: code: message
            pattern = r'^(.+?):(\d+):(\d+):\s*(\w+):\s*(.+)$'
            for line in output.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    issues.append({
                        'file': match.group(1),
                        'line': int(match.group(2)),
                        'column': int(match.group(3)),
                        'code': match.group(4),
                        'message': match.group(5)
                    })
        
        return issues
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': {
                'file_path': {
                    'type': 'string',
                    'description': 'Path to file to lint'
                },
                'linter': {
                    'type': 'string',
                    'description': 'Linter to use (flake8, pylint, or auto)',
                    'default': None
                },
                'max_issues': {
                    'type': 'integer',
                    'description': 'Maximum issues to report',
                    'default': 20
                }
            },
            'required': ['file_path']
        }
