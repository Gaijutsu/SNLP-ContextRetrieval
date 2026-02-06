"""
Docker-based sandbox for test execution in the SWE-bench comparison framework.

This module provides a Docker-based environment for applying patches and running
tests in an isolated, reproducible manner.
"""

import os
import re
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from .base import TestResults

if TYPE_CHECKING:
    from ..dataset.loader import SWEInstance


@dataclass
class SandboxConfig:
    """Configuration for Docker sandbox."""
    image: str = "python:3.10-slim"  # Default to standard Python image
    timeout: int = 300  # seconds
    memory_limit: str = "4g"
    cpu_limit: float = 2.0
    network_disabled: bool = True
    cleanup_on_exit: bool = True
    cache_level: str = "env"  # none, base, env, instance


class DockerSandbox:
    """
    Docker-based sandbox for test execution.
    
    This class provides methods for:
    - Creating and managing Docker containers
    - Applying patches in isolated environments
    - Running test suites
    - Capturing and parsing test results
    
    Example:
        sandbox = DockerSandbox(config)
        sandbox.initialize()
        
        # Apply patch
        applied = sandbox.apply_patch(patch_content, repo_path)
        
        # Run tests
        results = sandbox.run_tests(instance, repo_path)
        
        sandbox.cleanup()
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Docker sandbox.
        
        Args:
            config: Configuration dictionary
                - image: Docker image name (default: 'swe-bench')
                - timeout: Test timeout in seconds (default: 300)
                - memory_limit: Memory limit (default: '4g')
                - cpu_limit: CPU limit (default: 2.0)
                - network_disabled: Disable network (default: True)
                - cleanup_on_exit: Cleanup containers (default: True)
                - cache_level: Cache level (default: 'env')
        """
        self.config = SandboxConfig(**config)
        self._container_id: Optional[str] = None
        self._initialized = False
        
        # Don't check Docker here - do it lazily in initialize()
        self._docker_available: Optional[bool] = None
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def initialize(self) -> None:
        """Initialize the sandbox."""
        # Lazy check Docker availability
        if self._docker_available is None:
            self._docker_available = self._check_docker()
        
        if not self._docker_available:
            raise RuntimeError("Docker is not available")
        self._initialized = True
    
    def _get_or_create_container(
        self,
        repo_path: str,
        instance: Optional['SWEInstance'] = None
    ) -> str:
        """Get existing container or create a new one."""
        if self._container_id:
            return self._container_id
        
        if instance is None:
            raise ValueError("Instance required to create container")
        
        # Create a new container
        return self.create_container(repo_path, instance)
    
    def cleanup(self) -> None:
        """Cleanup sandbox resources."""
        if self._container_id and self.config.cleanup_on_exit:
            self._remove_container(self._container_id)
        self._initialized = False
    
    def create_container(
        self,
        repo_path: str,
        instance: 'SWEInstance',
        env_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a Docker container for evaluation.
        
        Args:
            repo_path: Path to the repository
            instance: SWE-bench instance
            env_vars: Additional environment variables
            
        Returns:
            Container ID
        """
        if not self._initialized:
            raise RuntimeError("Sandbox not initialized")
        
        # Convert Windows path to Docker-compatible format
        # Docker requires forward slashes and absolute paths
        abs_path = Path(repo_path).resolve()
        docker_path = str(abs_path).replace('\\', '/')
        
        # Build Docker run command
        cmd = [
            'docker', 'run',
            '-d',  # Detached mode
            '--rm',  # Remove on exit
            '--network', 'none' if self.config.network_disabled else 'bridge',
            '--memory', self.config.memory_limit,
            '--cpus', str(self.config.cpu_limit),
            '-v', f'{docker_path}:/repo:ro',  # Mount repo read-only
            '-w', '/repo',
        ]
        
        # Add environment variables
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(['-e', f'{key}={value}'])
        
        # Add image and command
        cmd.extend([self.config.image, 'sleep', '3600'])  # Keep container alive
        
        # Create container
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")
        
        self._container_id = result.stdout.strip()
        return self._container_id
    
    def apply_patch(
        self,
        patch_content: Optional[str],
        repo_path: str,
        container_id: Optional[str] = None,
        instance: Optional['SWEInstance'] = None
    ) -> bool:
        """
        Apply a patch in the sandbox.
        
        Args:
            patch_content: The patch content to apply
            repo_path: Path to the repository
            container_id: Optional container ID (uses managed container if None)
            instance: Optional instance for container creation
            
        Returns:
            True if patch was applied successfully
        """
        if not patch_content:
            return False
        
        if not self._initialized:
            raise RuntimeError("Sandbox not initialized")
        
        # Get or create container for Docker mode
        use_container = container_id or self._container_id
        if not use_container:
            use_container = self._get_or_create_container(repo_path, instance)
        
        # Create temporary patch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_content)
            patch_file = f.name
        
        try:
            if use_container:
                # Apply in container
                return self._apply_patch_in_container(patch_file, use_container)
            else:
                # No container available, apply locally (for non-Docker mode)
                return self._apply_patch_local(patch_file, repo_path)
            
        finally:
            # Clean up temporary file
            os.unlink(patch_file)
    
    def _apply_patch_local(self, patch_file: str, repo_path: str) -> bool:
        """Apply patch locally using git."""
        try:
            result = subprocess.run(
                ['git', 'apply', '--check', patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False
            
            result = subprocess.run(
                ['git', 'apply', patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error applying patch: {e}")
            return False
    
    def _apply_patch_in_container(
        self,
        patch_file: str,
        container_id: str
    ) -> bool:
        """Apply patch inside Docker container."""
        try:
            # Copy patch to container
            container_patch_path = f"/tmp/patch_{int(time.time())}.patch"
            result = subprocess.run(
                ['docker', 'cp', patch_file, f'{container_id}:{container_patch_path}'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False
            
            # Apply patch in container
            result = subprocess.run(
                ['docker', 'exec', container_id, 'git', 'apply', container_patch_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error applying patch in container: {e}")
            return False
    
    def run_tests(
        self,
        instance: 'SWEInstance',
        repo_path: str,
        container_id: Optional[str] = None
    ) -> TestResults:
        """
        Run tests in the sandbox.
        
        Args:
            instance: SWE-bench instance with test information
            repo_path: Path to the repository
            container_id: Optional container ID
            
        Returns:
            TestResults with pass/fail information
        """
        if not self._initialized:
            raise RuntimeError("Sandbox not initialized")
        
        # Get or create container
        use_container = container_id or self._container_id
        if not use_container:
            use_container = self._get_or_create_container(repo_path, instance)
        
        start_time = time.time()
        
        # Get test commands from instance
        fail_to_pass = getattr(instance, 'failed_tests', [])
        pass_to_pass = getattr(instance, 'passed_tests', [])
        
        all_tests = fail_to_pass + pass_to_pass
        
        if not all_tests:
            # No tests specified, try to detect
            return self._run_auto_detected_tests(repo_path, use_container)
        
        # Run specific tests
        results = TestResults()
        
        for test in all_tests:
            passed, output = self._run_single_test(
                test, repo_path, use_container
            )
            if passed:
                results.passed.append(test)
            else:
                results.failed.append(test)
            results.raw_output += f"\n{'='*50}\n{test}\n{'='*50}\n{output}"
        
        results.execution_time = time.time() - start_time
        return results
    
    def _run_single_test(
        self,
        test_name: str,
        repo_path: str,
        container_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Run a single test.
        
        Args:
            test_name: Name of the test to run
            repo_path: Path to the repository
            container_id: Optional container ID
            
        Returns:
            Tuple of (passed, output)
        """
        # Determine test framework and build command
        test_cmd = self._build_test_command(test_name, repo_path)
        
        try:
            if container_id:
                result = subprocess.run(
                    ['docker', 'exec', container_id] + test_cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
            else:
                result = subprocess.run(
                    test_cmd,
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
            
            output = result.stdout + result.stderr
            passed = result.returncode == 0
            
            return passed, output
            
        except subprocess.TimeoutExpired:
            return False, f"Test timed out after {self.config.timeout}s"
        except Exception as e:
            return False, f"Error running test: {str(e)}"
    
    def _build_test_command(self, test_name: str, repo_path: str) -> List[str]:
        """
        Build the test command based on test name format.
        
        Args:
            test_name: Name of the test
            repo_path: Path to the repository
            
        Returns:
            Command as list of strings
        """
        # Detect test framework
        if '::' in test_name or test_name.startswith('test_'):
            # pytest format
            return ['python', '-m', 'pytest', test_name, '-v']
        elif test_name.endswith('.py'):
            # Direct Python file
            return ['python', test_name]
        elif '.' in test_name and not test_name.endswith('.py'):
            # Python module path (e.g., module.submodule.test_function)
            return ['python', '-m', 'pytest', test_name, '-v']
        else:
            # Default to pytest
            return ['python', '-m', 'pytest', test_name, '-v']
    
    def _run_auto_detected_tests(
        self,
        repo_path: str,
        container_id: Optional[str] = None
    ) -> TestResults:
        """
        Auto-detect and run tests.
        
        Args:
            repo_path: Path to the repository
            container_id: Optional container ID
            
        Returns:
            TestResults
        """
        results = TestResults()
        
        # Look for common test directories
        test_dirs = ['tests', 'test', 'Tests', 'Test']
        
        for test_dir in test_dirs:
            test_path = Path(repo_path) / test_dir
            if test_path.exists():
                # Run pytest on the test directory
                cmd = ['python', '-m', 'pytest', str(test_path), '-v']
                
                try:
                    if container_id:
                        result = subprocess.run(
                            ['docker', 'exec', container_id] + cmd,
                            capture_output=True,
                            text=True,
                            timeout=self.config.timeout
                        )
                    else:
                        result = subprocess.run(
                            cmd,
                            cwd=repo_path,
                            capture_output=True,
                            text=True,
                            timeout=self.config.timeout
                        )
                    
                    output = result.stdout + result.stderr
                    results.raw_output = output
                    
                    # Parse results
                    parsed = self._parse_pytest_output(output)
                    results.passed = parsed['passed']
                    results.failed = parsed['failed']
                    results.errors = parsed['errors']
                    results.skipped = parsed['skipped']
                    
                    break
                    
                except Exception as e:
                    results.errors.append(f"Error running tests: {str(e)}")
        
        return results
    
    def _parse_pytest_output(self, output: str) -> Dict[str, List[str]]:
        """
        Parse pytest output to extract test results.
        
        Args:
            output: Raw pytest output
            
        Returns:
            Dictionary with passed, failed, errors, skipped lists
        """
        results = {
            'passed': [],
            'failed': [],
            'errors': [],
            'skipped': []
        }
        
        # Parse individual test results
        # Pattern: test_file.py::test_name PASSED/FAILED/ERROR/SKIPPED
        pattern = r'(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)'
        
        for match in re.finditer(pattern, output):
            test_name = match.group(1)
            status = match.group(2)
            
            if status == 'PASSED':
                results['passed'].append(test_name)
            elif status == 'FAILED':
                results['failed'].append(test_name)
            elif status == 'ERROR':
                results['errors'].append(test_name)
            elif status == 'SKIPPED':
                results['skipped'].append(test_name)
        
        return results
    
    def _remove_container(self, container_id: str) -> None:
        """Remove a Docker container."""
        try:
            subprocess.run(
                ['docker', 'rm', '-f', container_id],
                capture_output=True,
                timeout=30
            )
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        return {
            'docker_available': self._docker_available,
            'initialized': self._initialized,
            'container_id': self._container_id,
            'config': {
                'image': self.config.image,
                'timeout': self.config.timeout,
                'memory_limit': self.config.memory_limit,
                'cpu_limit': self.config.cpu_limit,
                'network_disabled': self.config.network_disabled,
                'cleanup_on_exit': self.config.cleanup_on_exit,
                'cache_level': self.config.cache_level
            }
        }


class LocalSandbox(DockerSandbox):
    """
    Local sandbox that runs tests without Docker.
    
    This is useful for testing and development when Docker is not available.
    Note: This does not provide the same isolation as Docker.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the local sandbox.
        
        Args:
            config: Configuration (timeout, etc.)
        """
        # Override Docker check
        super().__init__(config)
        self._docker_available = True  # Pretend Docker is available
    
    def create_container(self, *args, **kwargs) -> str:
        """Create container returns a dummy ID for local execution."""
        self._container_id = "local"
        return self._container_id
    
    def apply_patch(
        self,
        patch_content: Optional[str],
        repo_path: str,
        container_id: Optional[str] = None
    ) -> bool:
        """Apply patch locally."""
        return self._apply_patch_local_via_git(patch_content, repo_path)
    
    def _apply_patch_local_via_git(
        self,
        patch_content: Optional[str],
        repo_path: str
    ) -> bool:
        """Apply patch locally using git."""
        if not patch_content:
            return False
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            f.write(patch_content)
            patch_file = f.name
        
        try:
            # Check if patch can be applied
            result = subprocess.run(
                ['git', 'apply', '--check', patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return False
            
            # Apply the patch
            result = subprocess.run(
                ['git', 'apply', patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode == 0
            
        finally:
            os.unlink(patch_file)
    
    def _remove_container(self, container_id: str) -> None:
        """No-op for local sandbox."""
        pass


# Forward reference imports
from ..dataset.loader import SWEInstance  # noqa: E402
