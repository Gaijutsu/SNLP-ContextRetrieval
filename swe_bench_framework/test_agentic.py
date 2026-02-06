"""
Test script for agentic exploration module.

This script tests the basic functionality of the agentic context gathering
implementations to ensure they work correctly.
"""

import os
import sys
import tempfile
import shutil

# Add the framework to path
sys.path.insert(0, '/mnt/okcomputer/output/swe_bench_framework')

from context_gatherers import (
    create_gatherer,
    list_available_gatherers,
    get_gatherer_info,
    SWEInstance,
    ContextType
)


def create_test_repo() -> str:
    """Create a temporary test repository with sample Python files."""
    repo_path = tempfile.mkdtemp(prefix="test_repo_")
    
    # Create a sample Python file
    sample_code = '''
"""Sample module for testing."""

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result
    
    def subtract(self, a: int, b: int) -> int:
        """Subtract two numbers."""
        result = a - b
        self.history.append(f"subtract({a}, {b}) = {result}")
        return result
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"multiply({a}, {b}) = {result}")
        return result
    
    def divide(self, a: int, b: int) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"divide({a}, {b}) = {result}")
        return result


class AdvancedCalculator(Calculator):
    """An advanced calculator with more operations."""
    
    def power(self, base: int, exponent: int) -> int:
        """Calculate power."""
        result = base ** exponent
        self.history.append(f"power({base}, {exponent}) = {result}")
        return result
    
    def factorial(self, n: int) -> int:
        """Calculate factorial."""
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        result = 1
        for i in range(2, n + 1):
            result *= i
        self.history.append(f"factorial({n}) = {result}")
        return result


def helper_function():
    """A helper function."""
    return "Helper called"
'''
    
    with open(os.path.join(repo_path, 'calculator.py'), 'w') as f:
        f.write(sample_code)
    
    # Create another file
    utils_code = '''
"""Utility functions."""

import os


def read_file(path: str) -> str:
    """Read a file."""
    with open(path, 'r') as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write to a file."""
    with open(path, 'w') as f:
        f.write(content)


class FileManager:
    """Manage file operations."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def get_full_path(self, filename: str) -> str:
        """Get full path for a file."""
        return os.path.join(self.base_path, filename)
    
    def exists(self, filename: str) -> bool:
        """Check if file exists."""
        return os.path.exists(self.get_full_path(filename))
'''
    
    with open(os.path.join(repo_path, 'utils.py'), 'w') as f:
        f.write(utils_code)
    
    return repo_path


def test_list_gatherers():
    """Test listing available gatherers."""
    print("\n=== Testing list_available_gatherers ===")
    gatherers = list_available_gatherers()
    print(f"Available gatherers: {gatherers}")
    assert 'autocoderover' in gatherers
    assert 'swe_agent' in gatherers
    assert 'agentless' in gatherers
    print("✓ Passed")


def test_get_gatherer_info():
    """Test getting gatherer info."""
    print("\n=== Testing get_gatherer_info ===")
    for gatherer_type in ['autocoderover', 'swe_agent', 'agentless']:
        info = get_gatherer_info(gatherer_type)
        print(f"\n{gatherer_type}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Pattern: {info['pattern']}")
        assert 'name' in info
        assert 'description' in info
    print("✓ Passed")


def test_create_gatherer():
    """Test creating gatherer instances."""
    print("\n=== Testing create_gatherer ===")
    
    config = {'max_iterations': 10}
    
    for gatherer_type in ['autocoderover', 'swe_agent', 'agentless']:
        gatherer = create_gatherer(gatherer_type, config)
        print(f"Created {gatherer_type}: {gatherer.name}")
        assert gatherer is not None
        assert gatherer.config == config
    
    print("✓ Passed")


def test_agentless_exploration():
    """Test Agentless gatherer with a real repository."""
    print("\n=== Testing Agentless Exploration ===")
    
    # Create test repository
    repo_path = create_test_repo()
    
    try:
        # Create instance
        instance = SWEInstance(
            instance_id='test-001',
            repo='test-repo',
            base_commit='abc123',
            problem_statement='The Calculator.divide method has a bug when dividing by zero. It should return infinity instead of raising an error.',
            hints_text='Look at the Calculator class in calculator.py',
            test_patch='',
            patch='',
            failed_tests=[],
            passed_tests=[],
            modified_files=['calculator.py'],
            modified_methods=['Calculator.divide']
        )
        
        # Create gatherer
        config = {
            'max_iterations': 20,
            'max_files': 5,
            'max_classes_per_file': 3,
            'context_window_lines': 5
        }
        gatherer = create_gatherer('agentless', config)
        
        # Run exploration
        print(f"Exploring repository: {repo_path}")
        bundle = gatherer.gather_context(instance, repo_path)
        
        # Check results
        print(f"\nResults:")
        print(f"  Instance ID: {bundle.instance_id}")
        print(f"  Gatherer type: {bundle.gatherer_type}")
        print(f"  Number of chunks: {len(bundle.chunks)}")
        print(f"  Token count: {bundle.token_count}")
        print(f"  Repo structure keys: {list(bundle.repo_structure.keys())}")
        
        # Check for expected content
        chunk_types = [c.context_type.value for c in bundle.chunks]
        print(f"  Chunk types: {chunk_types}")
        
        # Verify we found the Calculator class
        calculator_found = any(
            'Calculator' in c.content or 'calculator' in c.source_file.lower()
            for c in bundle.chunks
        )
        print(f"  Calculator found: {calculator_found}")
        
        # Get stats
        stats = gatherer.get_stats()
        print(f"\nStats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        assert len(bundle.chunks) > 0, "Should have gathered some context"
        assert bundle.instance_id == 'test-001'
        
        print("✓ Passed")
        
    finally:
        # Cleanup
        shutil.rmtree(repo_path)


def test_autocoderover_exploration():
    """Test AutoCodeRover gatherer."""
    print("\n=== Testing AutoCodeRover Exploration ===")
    
    repo_path = create_test_repo()
    
    try:
        instance = SWEInstance(
            instance_id='test-002',
            repo='test-repo',
            base_commit='abc123',
            problem_statement='The AdvancedCalculator factorial method needs optimization for large numbers.',
            hints_text='',
            test_patch='',
            patch='',
            failed_tests=[],
            passed_tests=[],
            modified_files=['calculator.py'],
            modified_methods=['AdvancedCalculator.factorial']
        )
        
        config = {
            'max_iterations': 20,
            'max_files_per_layer': 3,
            'max_classes_per_layer': 5,
            'max_functions_per_layer': 10
        }
        
        gatherer = create_gatherer('autocoderover', config)
        bundle = gatherer.gather_context(instance, repo_path)
        
        print(f"Results:")
        print(f"  Number of chunks: {len(bundle.chunks)}")
        print(f"  Token count: {bundle.token_count}")
        
        stats = gatherer.get_stats()
        print(f"\nStats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Verify stratified layers
        assert 'stratified_layers' in stats
        
        print("✓ Passed")
        
    finally:
        shutil.rmtree(repo_path)


def test_swe_agent_exploration():
    """Test SWE-agent gatherer."""
    print("\n=== Testing SWE-agent Exploration ===")
    
    repo_path = create_test_repo()
    
    try:
        instance = SWEInstance(
            instance_id='test-003',
            repo='test-repo',
            base_commit='abc123',
            problem_statement='The FileManager class should handle file not found errors gracefully.',
            hints_text='',
            test_patch='',
            patch='',
            failed_tests=[],
            passed_tests=[],
            modified_files=['utils.py'],
            modified_methods=['FileManager.exists']
        )
        
        config = {
            'max_iterations': 15,
            'min_iterations': 3,
            'enable_completion_detection': True
        }
        
        gatherer = create_gatherer('swe_agent', config)
        bundle = gatherer.gather_context(instance, repo_path)
        
        print(f"Results:")
        print(f"  Number of chunks: {len(bundle.chunks)}")
        print(f"  Token count: {bundle.token_count}")
        
        stats = gatherer.get_stats()
        print(f"\nStats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Verify ReAct pattern
        assert stats.get('pattern') == 'ReAct'
        
        print("✓ Passed")
        
    finally:
        shutil.rmtree(repo_path)


def test_tools():
    """Test individual tools."""
    print("\n=== Testing Tools ===")
    
    repo_path = create_test_repo()
    
    try:
        from context_gatherers.agentic.tools import (
            ViewFileTool,
            GrepTool,
            FindFileTool,
            SearchClassTool,
            SearchMethodTool
        )
        
        # Test ViewFileTool
        print("\nTesting ViewFileTool...")
        view_tool = ViewFileTool(repo_path)
        obs = view_tool.execute({'path': 'calculator.py', 'limit': 30})
        assert obs.success
        assert 'Calculator' in obs.output
        print("  ✓ ViewFileTool works")
        
        # Test GrepTool
        print("\nTesting GrepTool...")
        grep_tool = GrepTool(repo_path)
        obs = grep_tool.execute({'pattern': 'def add'})
        assert obs.success
        assert 'calculator.py' in obs.output
        print("  ✓ GrepTool works")
        
        # Test FindFileTool
        print("\nTesting FindFileTool...")
        find_tool = FindFileTool(repo_path)
        obs = find_tool.execute({'pattern': 'calc'})
        assert obs.success
        assert 'calculator.py' in obs.output
        print("  ✓ FindFileTool works")
        
        # Test SearchClassTool
        print("\nTesting SearchClassTool...")
        search_class = SearchClassTool(repo_path)
        obs = search_class.execute({'class_name': 'Calculator'})
        assert obs.success
        assert 'Calculator' in obs.output
        print("  ✓ SearchClassTool works")
        
        # Test SearchMethodTool
        print("\nTesting SearchMethodTool...")
        search_method = SearchMethodTool(repo_path)
        obs = search_method.execute({'method_name': 'factorial'})
        assert obs.success
        assert 'factorial' in obs.output
        print("  ✓ SearchMethodTool works")
        
        print("\n✓ All tools passed")
        
    finally:
        shutil.rmtree(repo_path)


def main():
    """Run all tests."""
    print("=" * 60)
    print("Agentic Exploration Module Tests")
    print("=" * 60)
    
    try:
        test_list_gatherers()
        test_get_gatherer_info()
        test_create_gatherer()
        test_tools()
        test_agentless_exploration()
        test_autocoderover_exploration()
        test_swe_agent_exploration()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
