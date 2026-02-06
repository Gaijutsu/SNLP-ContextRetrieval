# Patch Generators API Documentation

This document describes the Patch Generator API, which provides a unified interface for generating patches from gathered context.

---

## Overview

The `PatchGenerator` interface defines how patches are generated from gathered context. Different implementations use various strategies like direct single-pass generation, iterative refinement, or structured edit scripts.

```python
from swe_bench_framework.patch_generators import PatchGenerator

class MyGenerator(PatchGenerator):
    def generate_patch(self, context_bundle, instance):
        # Implementation
        pass
    
    def validate_patch(self, patch_content, repo_path):
        pass
```

---

## PatchGenerator Interface

### Class Definition

```python
class PatchGenerator(ABC):
    """
    Abstract base class for all patch generation strategies.
    
    This interface defines how patches are generated from gathered context.
    Implementations can use different strategies like direct generation,
    iterative refinement, or structured edit scripts.
    
    Example:
        >>> generator = MyPatchGenerator(llm_config, config)
        >>> result = generator.generate_patch(context_bundle, instance)
        >>> if result.success:
        ...     print(f"Generated patch: {result.patch_content}")
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the patch generator.
        
        Args:
            llm_config: Configuration for the LLM (model, temperature, etc.)
            config: Configuration for the generation strategy
        """
        self.llm_config = llm_config
        self.config = config
        self.name = self.__class__.__name__
```

### Abstract Methods

#### `generate_patch()`

```python
@abstractmethod
def generate_patch(
    self,
    context_bundle: ContextBundle,
    instance: SWEInstance
) -> PatchResult:
    """
    Generate a patch given context and problem statement.
    
    This is the main method that all patch generation strategies
    must implement. It should return a PatchResult containing
    the generated patch or error information.
    
    Args:
        context_bundle: Gathered context from a ContextGatherer
        instance: The SWE-bench instance
        
    Returns:
        PatchResult containing the generated patch or error
        
    Raises:
        GenerationError: If patch generation fails
    """
    pass
```

#### `validate_patch()`

```python
@abstractmethod
def validate_patch(self, patch_content: str, repo_path: str) -> bool:
    """
    Validate that a patch is syntactically correct.
    
    Args:
        patch_content: The patch content to validate
        repo_path: Path to the repository
        
    Returns:
        True if the patch is valid, False otherwise
    """
    pass
```

### Utility Methods

#### `get_stats()`

```python
def get_stats(self) -> Dict[str, Any]:
    """
    Return statistics about patch generation.
    
    Returns:
        Dictionary with statistics (implementation-specific)
    """
    return {}
```

---

## Data Models

### PatchResult

```python
@dataclass
class PatchResult:
    """
    Result of patch generation.
    
    Contains the generated patch (if successful) along with
    metadata about the generation process.
    """
    
    instance_id: str
    """Unique identifier for the SWE-bench instance."""
    
    patch_content: Optional[str]
    """The generated patch in unified diff format, or None if failed."""
    
    success: bool
    """Whether patch generation was successful."""
    
    generation_time: float
    """Time taken to generate the patch (seconds)."""
    
    attempts: int
    """Number of attempts made."""
    
    token_usage: Dict[str, int]
    """Token usage statistics (input, output, total)."""
    
    error_message: Optional[str] = None
    """Error message if generation failed."""
    
    intermediate_steps: List[Dict] = None
    """Intermediate steps for iterative generators."""
    
    confidence_score: float = 0.0
    """Confidence score (0.0-1.0) for the patch."""
```

---

## Direct Patch Generator

```python
class DirectPatchGenerator(PatchGenerator):
    """
    Single-pass patch generation using LLM.
    
    This is the simplest patch generation strategy. It builds a prompt
    from the gathered context and problem statement, sends it to the LLM,
    and extracts the patch from the response.
    
    Key Features:
    - Single LLM call
    - Fast generation
    - Simple implementation
    - Good baseline performance
    
    Example:
        >>> llm_config = {
        ...     'provider': 'openai',
        ...     'model': 'gpt-4-turbo-preview',
        ...     'temperature': 0.0
        ... }
        >>> config = {'max_attempts': 3}
        >>> generator = DirectPatchGenerator(llm_config, config)
        >>> result = generator.generate_patch(context_bundle, instance)
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(llm_config, config)
        self.max_attempts = config.get('max_attempts', 3)
        self.llm_client = self._create_llm_client(llm_config)
        self.prompt_builder = PatchPromptBuilder()
    
    def generate_patch(
        self,
        context_bundle: ContextBundle,
        instance: SWEInstance
    ) -> PatchResult:
        """
        Generate patch using single-pass LLM generation.
        
        Process:
        1. Build prompt from context bundle
        2. Call LLM
        3. Extract patch from response
        4. Validate patch
        5. Retry if invalid (up to max_attempts)
        """
        start_time = time.time()
        
        for attempt in range(1, self.max_attempts + 1):
            # Build prompt
            prompt = self.prompt_builder.build(context_bundle, instance)
            
            # Call LLM
            response = self.llm_client.generate(prompt)
            
            # Extract patch
            patch_content = self._extract_patch(response)
            
            # Validate
            if patch_content and self.validate_patch(patch_content, instance.repo_path):
                return PatchResult(
                    instance_id=instance.instance_id,
                    patch_content=patch_content,
                    success=True,
                    generation_time=time.time() - start_time,
                    attempts=attempt,
                    token_usage=self.llm_client.get_stats(),
                    confidence_score=self._compute_confidence(response)
                )
        
        # All attempts failed
        return PatchResult(
            instance_id=instance.instance_id,
            patch_content=None,
            success=False,
            generation_time=time.time() - start_time,
            attempts=self.max_attempts,
            token_usage=self.llm_client.get_stats(),
            error_message="Failed to generate valid patch after all attempts"
        )
    
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """
        Validate that a patch is syntactically correct.
        
        Checks:
        1. Patch format (unified diff)
        2. Syntax of modified files
        3. Patch applicability
        """
        try:
            # Check patch format
            if not patch_content.startswith('diff --git'):
                return False
            
            # Try to apply patch
            result = subprocess.run(
                ['git', 'apply', '--check', '-'],
                input=patch_content,
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
        except Exception:
            return False
    
    def _extract_patch(self, response: str) -> Optional[str]:
        """
        Extract patch from LLM response.
        
        Looks for code blocks with diff format.
        """
        # Look for ```diff blocks
        diff_pattern = r'```diff\n(.*?)```'
        matches = re.findall(diff_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Look for raw diff format
        if 'diff --git' in response:
            start = response.find('diff --git')
            return response[start:].strip()
        
        return None
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_attempts` | int | 3 | Maximum retry attempts |
| `validation.syntax_check` | bool | True | Enable syntax validation |
| `validation.test_before_submit` | bool | False | Run tests before returning |

---

## Iterative Patch Generator

```python
class IterativePatchGenerator(PatchGenerator):
    """
    Multi-turn patch generation with feedback.
    
    This generator uses an iterative approach where the LLM
    generates a patch, receives feedback, and refines it.
    
    Key Features:
    - Multi-turn refinement
    - Feedback-driven improvement
    - Test execution feedback
    - Linter feedback
    
    Example:
        >>> config = {
        ...     'max_iterations': 5,
        ...     'feedback_types': ['syntax', 'test', 'linter'],
        ...     'stop_on_success': True
        ... }
        >>> generator = IterativePatchGenerator(llm_config, config)
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(llm_config, config)
        self.max_iterations = config.get('max_iterations', 5)
        self.feedback_types = config.get('feedback_types', ['syntax'])
        self.stop_on_success = config.get('stop_on_success', True)
        self.llm_client = self._create_llm_client(llm_config)
        self.prompt_builder = IterativePromptBuilder()
    
    def generate_patch(
        self,
        context_bundle: ContextBundle,
        instance: SWEInstance
    ) -> PatchResult:
        """
        Generate patch using iterative refinement.
        
        Process:
        1. Generate initial patch
        2. Validate and collect feedback
        3. If invalid, generate refined patch with feedback
        4. Repeat until success or max iterations
        """
        start_time = time.time()
        intermediate_steps = []
        
        # Initial prompt
        messages = [
            {
                'role': 'system',
                'content': self.prompt_builder.build_system_prompt()
            },
            {
                'role': 'user',
                'content': self.prompt_builder.build_initial_prompt(
                    context_bundle, instance
                )
            }
        ]
        
        for iteration in range(1, self.max_iterations + 1):
            # Generate patch
            response = self.llm_client.chat(messages)
            patch_content = self._extract_patch(response)
            
            # Validate
            validation_result = self._validate_with_feedback(
                patch_content, instance
            )
            
            step = {
                'iteration': iteration,
                'patch': patch_content,
                'valid': validation_result['valid'],
                'feedback': validation_result['feedback']
            }
            intermediate_steps.append(step)
            
            if validation_result['valid'] and self.stop_on_success:
                return PatchResult(
                    instance_id=instance.instance_id,
                    patch_content=patch_content,
                    success=True,
                    generation_time=time.time() - start_time,
                    attempts=iteration,
                    token_usage=self.llm_client.get_stats(),
                    intermediate_steps=intermediate_steps,
                    confidence_score=self._compute_confidence(
                        validation_result
                    )
                )
            
            # Add feedback for next iteration
            messages.append({'role': 'assistant', 'content': response})
            messages.append({
                'role': 'user',
                'content': self.prompt_builder.build_feedback_prompt(
                    validation_result['feedback']
                )
            })
        
        # Max iterations reached
        return PatchResult(
            instance_id=instance.instance_id,
            patch_content=None,
            success=False,
            generation_time=time.time() - start_time,
            attempts=self.max_iterations,
            token_usage=self.llm_client.get_stats(),
            intermediate_steps=intermediate_steps,
            error_message="Max iterations reached without valid patch"
        )
    
    def _validate_with_feedback(
        self,
        patch_content: str,
        instance: SWEInstance
    ) -> Dict[str, Any]:
        """
        Validate patch and collect feedback.
        
        Returns:
            Dictionary with 'valid' flag and 'feedback' message
        """
        feedback = []
        
        # Syntax validation
        if 'syntax' in self.feedback_types:
            syntax_valid = self._check_syntax(patch_content)
            if not syntax_valid:
                feedback.append("Syntax error in generated patch")
        
        # Test validation
        if 'test' in self.feedback_types:
            test_result = self._run_tests(patch_content, instance)
            if not test_result['passed']:
                feedback.append(f"Tests failed: {test_result['output']}")
        
        # Linter validation
        if 'linter' in self.feedback_types:
            linter_result = self._run_linter(patch_content, instance)
            if linter_result['issues']:
                feedback.append(f"Linter issues: {linter_result['issues']}")
        
        return {
            'valid': len(feedback) == 0,
            'feedback': '\n'.join(feedback) if feedback else 'Patch is valid'
        }
    
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """Validate patch syntax."""
        return self._check_syntax(patch_content)
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 5 | Maximum refinement iterations |
| `feedback_types` | list | ["syntax"] | Types of feedback to use |
| `stop_on_success` | bool | True | Stop when valid patch found |
| `validation.syntax_check` | bool | True | Enable syntax checking |
| `validation.test_execution` | bool | False | Enable test execution |
| `validation.linter` | bool | False | Enable linter checks |

---

## Edit Script Generator

```python
class EditScriptGenerator(PatchGenerator):
    """
    Structured edit script generation.
    
    Instead of generating raw diff, this generator produces
    structured edit scripts that are then converted to patches.
    
    Key Features:
    - Structured edit representation
    - Easier to validate
    - More controlled generation
    - Better for complex multi-file changes
    
    Example:
        >>> config = {
        ...     'edit_types': ['replace', 'insert', 'delete'],
        ...     'validate_scripts': True
        ... }
        >>> generator = EditScriptGenerator(llm_config, config)
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(llm_config, config)
        self.edit_types = config.get('edit_types', ['replace'])
        self.validate_scripts = config.get('validate_scripts', True)
        self.llm_client = self._create_llm_client(llm_config)
        self.script_parser = EditScriptParser()
    
    def generate_patch(
        self,
        context_bundle: ContextBundle,
        instance: SWEInstance
    ) -> PatchResult:
        """
        Generate patch using structured edit scripts.
        
        Process:
        1. Generate edit script (JSON format)
        2. Parse and validate script
        3. Convert script to unified diff
        4. Validate final patch
        """
        start_time = time.time()
        
        # Build prompt for edit script
        prompt = self._build_script_prompt(context_bundle, instance)
        
        # Generate script
        response = self.llm_client.generate(prompt)
        
        # Parse script
        try:
            edit_script = self.script_parser.parse(response)
        except ParseError as e:
            return PatchResult(
                instance_id=instance.instance_id,
                patch_content=None,
                success=False,
                generation_time=time.time() - start_time,
                attempts=1,
                token_usage=self.llm_client.get_stats(),
                error_message=f"Failed to parse edit script: {e}"
            )
        
        # Validate script
        if self.validate_scripts:
            valid, error = self._validate_script(edit_script, instance)
            if not valid:
                return PatchResult(
                    instance_id=instance.instance_id,
                    patch_content=None,
                    success=False,
                    generation_time=time.time() - start_time,
                    attempts=1,
                    token_usage=self.llm_client.get_stats(),
                    error_message=f"Invalid edit script: {error}"
                )
        
        # Convert to patch
        patch_content = self._script_to_patch(edit_script, instance)
        
        return PatchResult(
            instance_id=instance.instance_id,
            patch_content=patch_content,
            success=True,
            generation_time=time.time() - start_time,
            attempts=1,
            token_usage=self.llm_client.get_stats(),
            confidence_score=1.0 if self.validate_scripts else 0.5
        )
    
    def _build_script_prompt(
        self,
        context_bundle: ContextBundle,
        instance: SWEInstance
    ) -> str:
        """Build prompt for edit script generation."""
        return f"""You are an expert software engineer. Fix the following issue by generating an edit script.

## Problem Statement
{instance.problem_statement}

## Repository Context
{context_bundle.to_prompt_context()}

## Edit Script Format
Generate a JSON edit script with the following structure:
```json
{{
  "edits": [
    {{
      "file": "path/to/file.py",
      "type": "replace",
      "search": "code to find",
      "replace": "new code"
    }}
  ]
}}
```

Available edit types: {', '.join(self.edit_types)}

## Instructions
1. Analyze the problem and context
2. Generate a valid edit script
3. Ensure the script can be applied cleanly
"""
    
    def _script_to_patch(
        self,
        edit_script: Dict[str, Any],
        instance: SWEInstance
    ) -> str:
        """Convert edit script to unified diff patch."""
        patches = []
        
        for edit in edit_script.get('edits', []):
            file_path = edit['file']
            edit_type = edit['type']
            
            if edit_type == 'replace':
                patch = self._create_replace_patch(
                    file_path,
                    edit['search'],
                    edit['replace']
                )
            elif edit_type == 'insert':
                patch = self._create_insert_patch(
                    file_path,
                    edit['after'],
                    edit['content']
                )
            elif edit_type == 'delete':
                patch = self._create_delete_patch(
                    file_path,
                    edit['content']
                )
            
            patches.append(patch)
        
        return '\n'.join(patches)
    
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """Validate patch by trying to apply it."""
        try:
            result = subprocess.run(
                ['git', 'apply', '--check', '-'],
                input=patch_content,
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `edit_types` | list | ["replace"] | Allowed edit types |
| `validate_scripts` | bool | True | Validate scripts before conversion |
| `max_edits` | int | 10 | Maximum number of edits per script |

---

## Prompt Builders

### PatchPromptBuilder

```python
class PatchPromptBuilder:
    """
    Builds prompts for patch generation.
    
    Creates well-structured prompts that guide the LLM
    to generate correct patches.
    """
    
    def build(
        self,
        context_bundle: ContextBundle,
        instance: SWEInstance
    ) -> str:
        """Build a complete prompt for patch generation."""
        return f"""You are an expert software engineer. Fix the following issue.

## Problem Statement
{instance.problem_statement}

## Repository Context
{context_bundle.to_prompt_context(max_tokens=8000)}

## Instructions
1. Carefully analyze the problem and context
2. Identify the root cause
3. Implement a minimal fix that resolves the issue
4. Ensure the fix doesn't break existing functionality

## Output Format
Provide your fix as a unified diff in the following format:

```diff
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -line,offset +line,offset @@
 def function():
-    old_code()
+    new_code()
```

Important:
- Only modify files that need to be changed
- Ensure the patch is syntactically correct
- Include complete context lines (3 lines before/after changes)
- Do not include explanations outside the diff block
"""
```

---

## Factory and Registration

### Creating Generators

```python
from swe_bench_framework.patch_generators import PatchGeneratorFactory

# Create by strategy name
generator = PatchGeneratorFactory.create('direct', llm_config, config)

# Create from config
config = {'strategy': 'iterative', 'max_iterations': 5}
generator = PatchGeneratorFactory.create_from_config(config, llm_config)
```

### Registering Custom Generators

```python
from swe_bench_framework.patch_generators import PatchGeneratorFactory
from swe_bench_framework.core.registry import register_generator

# Define custom generator
class MyCustomGenerator(PatchGenerator):
    def generate_patch(self, context_bundle, instance):
        # Implementation
        pass
    
    def validate_patch(self, patch_content, repo_path):
        pass

# Register
register_generator('my_custom', MyCustomGenerator)

# Use in config
patch_generation:
  strategy: "my_custom"
  config:
    # custom options
```

---

## Best Practices

### 1. Always Validate Patches

```python
# In your generator
def generate_patch(self, context_bundle, instance):
    patch_content = self._generate(context_bundle, instance)
    
    if not self.validate_patch(patch_content, instance.repo_path):
        # Retry or return failure
        pass
```

### 2. Handle Extraction Failures

```python
def _extract_patch(self, response: str) -> Optional[str]:
    # Try multiple extraction strategies
    
    # 1. Look for diff code blocks
    patch = self._extract_from_code_block(response)
    if patch:
        return patch
    
    # 2. Look for raw diff
    patch = self._extract_raw_diff(response)
    if patch:
        return patch
    
    # 3. Return None if all fail
    return None
```

### 3. Track Token Usage

```python
def generate_patch(self, context_bundle, instance):
    start_tokens = self.llm_client.get_token_count()
    
    response = self.llm_client.generate(prompt)
    
    end_tokens = self.llm_client.get_token_count()
    token_usage = {
        'input': len(self.llm_client.encode(prompt)),
        'output': len(self.llm_client.encode(response)),
        'total': end_tokens - start_tokens
    }
    
    return PatchResult(
        ...,
        token_usage=token_usage
    )
```

### 4. Use Confidence Scores

```python
def _compute_confidence(self, response: str) -> float:
    """Compute confidence score for generated patch."""
    # Factors that increase confidence:
    # - Patch is properly formatted
    # - Multiple validation checks pass
    # - LLM expresses confidence in response
    
    confidence = 0.5  # Base confidence
    
    if 'diff --git' in response:
        confidence += 0.2
    
    if response.count('@@') >= 2:
        confidence += 0.1
    
    return min(confidence, 1.0)
```
