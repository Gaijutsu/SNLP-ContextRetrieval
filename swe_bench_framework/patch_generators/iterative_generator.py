"""
Iterative patch generator with multi-turn refinement.

This module implements an iterative approach to patch generation where the LLM
can refine its patch based on validation feedback, enabling self-correction.
"""

import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .base import PatchGenerator, PatchResult
from .direct_generator import DirectPatchGenerator, LLMResponse
from .prompts.builders import PromptBuilder, ContextBundle
from .prompts.templates import PromptTemplates


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    patch_content: Optional[str]
    feedback: str
    success: bool
    token_usage: Dict[str, int] = field(default_factory=dict)


class IterativePatchGenerator(PatchGenerator):
    """
    Multi-turn patch generation with iterative refinement.
    
    This generator attempts to generate a patch, validates it, and if validation
    fails, provides feedback to the LLM for refinement. This process repeats
    until a valid patch is generated or the maximum number of iterations is reached.
    
    Example:
        generator = IterativePatchGenerator(
            llm_config={'model': 'gpt-5-mini', 'temperature': 0.0},
            config={'max_iterations': 3, 'validation_fn': custom_validator}
        )
        generator.initialize()
        result = generator.generate_patch(context_bundle, instance)
        # result.intermediate_steps contains all iteration details
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the iterative patch generator.
        
        Args:
            llm_config: Configuration for the LLM
                - model: Model name
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Maximum tokens in response
            config: Generator-specific configuration
                - max_iterations: Maximum refinement iterations (default: 3)
                - validation_fn: Custom validation function
                - feedback_builder: Custom feedback builder function
                - early_stop_on_valid: Stop early if patch is valid (default: True)
                - max_context_tokens: Max tokens for context (default: 8000)
        """
        super().__init__(llm_config, config)
        self.max_iterations = config.get('max_iterations', 3)
        self.early_stop_on_valid = config.get('early_stop_on_valid', True)
        self.max_context_tokens = config.get('max_context_tokens', 8000)
        
        # Custom validation and feedback functions
        self.validation_fn: Optional[Callable[[str, str], Tuple[bool, str]]] = \
            config.get('validation_fn')
        self.feedback_builder: Optional[Callable[[str, str, int], str]] = \
            config.get('feedback_builder')
        
        # Base generator for single-pass generation
        base_config = config.copy()
        base_config['validate_syntax'] = False  # We'll handle validation
        self.base_generator = DirectPatchGenerator(llm_config, base_config)
        
        self.prompt_builder = PromptBuilder(max_tokens=self.max_context_tokens)
        self.templates = PromptTemplates()
        
        # Iteration history
        self.iteration_results: List[IterationResult] = []
    
    def initialize(self) -> None:
        """Initialize the base generator."""
        super().initialize()
        self.base_generator.initialize()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.base_generator.cleanup()
        super().cleanup()
    
    def generate_patch(
        self,
        context_bundle: ContextBundle,
        instance: 'SWEInstance'
    ) -> PatchResult:
        """
        Generate a patch with iterative refinement.
        
        Args:
            context_bundle: Gathered context for the instance
            instance: The SWE-bench instance
            
        Returns:
            PatchResult with the best generated patch
        """
        start_time = time.time()
        instance_id = instance.instance_id
        self.iteration_results = []
        
        # Build initial prompt
        base_prompt = self._build_initial_prompt(context_bundle, instance)
        
        current_prompt = base_prompt
        best_patch: Optional[str] = None
        best_result: Optional[PatchResult] = None
        total_token_usage: Dict[str, int] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        
        for iteration in range(1, self.max_iterations + 1):
            iter_start_time = time.time()
            
            # Generate patch for this iteration
            patch, response = self._generate_single_attempt(current_prompt)
            
            if patch is None:
                # No patch extracted, provide feedback
                feedback = "No valid patch was extracted from the response. Please provide a patch in the correct format."
                self._record_iteration(iteration, None, feedback, False, response)
                
                if iteration < self.max_iterations:
                    current_prompt = self._build_refinement_prompt(
                        base_prompt, "", feedback, iteration
                    )
                continue
            
            # Validate the patch
            is_valid, validation_message = self._validate_patch(
                patch, instance.repo_path if hasattr(instance, 'repo_path') else ""
            )
            
            # Update token usage
            total_token_usage['prompt_tokens'] += response.prompt_tokens
            total_token_usage['completion_tokens'] += response.completion_tokens
            total_token_usage['total_tokens'] += response.total_tokens
            
            if is_valid:
                # Valid patch found
                self._record_iteration(iteration, patch, validation_message, True, response)
                
                generation_time = time.time() - start_time
                result = PatchResult(
                    instance_id=instance_id,
                    patch_content=patch,
                    success=True,
                    generation_time=generation_time,
                    attempts=iteration,
                    token_usage=total_token_usage.copy(),
                    intermediate_steps=self._get_iteration_dicts(),
                    confidence_score=self._calculate_confidence(iteration, True)
                )
                
                if self.early_stop_on_valid:
                    return result
                
                # Store best result and continue
                if best_result is None or iteration < best_result.attempts:
                    best_patch = patch
                    best_result = result
            else:
                # Invalid patch, prepare for refinement
                feedback = validation_message
                self._record_iteration(iteration, patch, feedback, False, response)
                
                if iteration < self.max_iterations:
                    current_prompt = self._build_refinement_prompt(
                        base_prompt, patch, feedback, iteration
                    )
        
        # Return best result if found, otherwise return failure
        generation_time = time.time() - start_time
        
        if best_result is not None:
            best_result.generation_time = generation_time
            return best_result
        
        # All iterations failed
        last_iteration = self.iteration_results[-1] if self.iteration_results else None
        return PatchResult(
            instance_id=instance_id,
            patch_content=None,
            success=False,
            generation_time=generation_time,
            attempts=self.max_iterations,
            token_usage=total_token_usage,
            error_message=last_iteration.feedback if last_iteration else "All iterations failed",
            intermediate_steps=self._get_iteration_dicts()
        )
    
    def _build_initial_prompt(
        self,
        context_bundle: ContextBundle,
        instance: 'SWEInstance'
    ) -> str:
        """Build the initial prompt."""
        hints_text = getattr(instance, 'hints_text', None)
        custom_instructions = self.config.get('custom_instructions')
        
        return self.prompt_builder.build(
            context_bundle=context_bundle,
            problem_statement=instance.problem_statement,
            hints_text=hints_text,
            custom_instructions=custom_instructions
        )
    
    def _generate_single_attempt(self, prompt: str) -> tuple:
        """
        Generate a single patch attempt.
        
        Returns:
            Tuple of (patch_content, llm_response)
        """
        response = self.base_generator._call_llm(prompt)
        patch = self.base_generator._extract_patch(response.content)
        return patch, response
    
    def _validate_patch(self, patch_content: str, repo_path: str) -> tuple:
        """
        Validate a patch and return feedback.
        
        Returns:
            Tuple of (is_valid, feedback_message)
        """
        # Use custom validation if provided
        if self.validation_fn is not None:
            return self.validation_fn(patch_content, repo_path)
        
        # Default validation
        if not patch_content:
            return False, "Patch is empty"
        
        # Basic format check
        if '---' not in patch_content or '+++' not in patch_content:
            return False, "Patch missing required headers (--- and +++)"
        
        # Try git validation if repo_path available
        if repo_path:
            is_valid = self.base_generator._validate_with_git(patch_content, repo_path)
            if not is_valid:
                return False, "Patch failed git apply validation"
        
        return True, "Patch is valid"
    
    def _build_refinement_prompt(
        self,
        base_prompt: str,
        previous_patch: str,
        feedback: str,
        iteration: int
    ) -> str:
        """
        Build a refinement prompt for the next iteration.
        
        Args:
            base_prompt: Original generation prompt
            previous_patch: Previous patch attempt
            feedback: Feedback on the previous attempt
            iteration: Current iteration number
            
        Returns:
            Refinement prompt
        """
        # Use custom feedback builder if provided
        if self.feedback_builder is not None:
            return self.feedback_builder(base_prompt, previous_patch, feedback, iteration)
        
        # Default refinement prompt
        refinement_section = f"""
## Previous Attempt (Iteration {iteration})

Your previous patch:
```diff
{previous_patch}
```

## Feedback

{feedback}

## Task

Please provide a corrected patch that addresses the feedback above. Ensure your patch:
1. Fixes the original problem
2. Addresses the issues mentioned in the feedback
3. Is in the correct unified diff format

Provide your corrected patch below:

```diff
"""
        
        return base_prompt + refinement_section
    
    def _record_iteration(
        self,
        iteration: int,
        patch: Optional[str],
        feedback: str,
        success: bool,
        response: LLMResponse
    ) -> None:
        """Record iteration result."""
        result = IterationResult(
            iteration=iteration,
            patch_content=patch,
            feedback=feedback,
            success=success,
            token_usage={
                'prompt_tokens': response.prompt_tokens,
                'completion_tokens': response.completion_tokens,
                'total_tokens': response.total_tokens
            }
        )
        self.iteration_results.append(result)
    
    def _get_iteration_dicts(self) -> List[Dict[str, Any]]:
        """Convert iteration results to dictionaries."""
        return [
            {
                'iteration': r.iteration,
                'patch_content': r.patch_content,
                'feedback': r.feedback,
                'success': r.success,
                'token_usage': r.token_usage
            }
            for r in self.iteration_results
        ]
    
    def _calculate_confidence(self, iterations_used: int, final_success: bool) -> float:
        """
        Calculate confidence score based on iteration history.
        
        Args:
            iterations_used: Number of iterations used
            final_success: Whether final result was successful
            
        Returns:
            Confidence score (0.0-1.0)
        """
        if not final_success:
            return 0.0
        
        # Higher confidence if succeeded quickly
        if iterations_used == 1:
            return 0.9
        elif iterations_used == 2:
            return 0.7
        else:
            return 0.5
    
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """
        Validate a patch.
        
        Args:
            patch_content: The patch to validate
            repo_path: Path to the repository
            
        Returns:
            True if valid
        """
        is_valid, _ = self._validate_patch(patch_content, repo_path)
        return is_valid
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        stats = super().get_stats()
        stats.update({
            'max_iterations': self.max_iterations,
            'early_stop_on_valid': self.early_stop_on_valid,
            'iteration_count': len(self.iteration_results),
            'has_custom_validation': self.validation_fn is not None,
            'has_custom_feedback': self.feedback_builder is not None
        })
        return stats


class FeedbackBasedGenerator(IterativePatchGenerator):
    """
    Specialized iterative generator that uses specific feedback types.
    
    This generator categorizes failures and provides targeted feedback
    for different error types (syntax errors, test failures, etc.).
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the feedback-based generator.
        
        Args:
            llm_config: LLM configuration
            config: Generator configuration with additional options:
                - enable_syntax_feedback: Provide syntax error feedback
                - enable_test_feedback: Provide test failure feedback
                - enable_semantic_feedback: Provide semantic error feedback
        """
        super().__init__(llm_config, config)
        self.enable_syntax_feedback = config.get('enable_syntax_feedback', True)
        self.enable_test_feedback = config.get('enable_test_feedback', True)
        self.enable_semantic_feedback = config.get('enable_semantic_feedback', False)
    
    def _categorize_error(self, patch: str, feedback: str) -> str:
        """
        Categorize the type of error in the patch.
        
        Args:
            patch: The patch that failed
            feedback: Error feedback
            
        Returns:
            Error category
        """
        feedback_lower = feedback.lower()
        
        if 'syntax' in feedback_lower or 'parse' in feedback_lower:
            return 'syntax'
        elif 'test' in feedback_lower or 'fail' in feedback_lower:
            return 'test_failure'
        elif 'apply' in feedback_lower or 'patch' in feedback_lower:
            return 'apply_error'
        elif 'semantic' in feedback_lower or 'logic' in feedback_lower:
            return 'semantic'
        else:
            return 'unknown'
    
    def _build_refinement_prompt(
        self,
        base_prompt: str,
        previous_patch: str,
        feedback: str,
        iteration: int
    ) -> str:
        """
        Build targeted refinement prompt based on error type.
        """
        error_type = self._categorize_error(previous_patch, feedback)
        
        # Build targeted feedback
        if error_type == 'syntax' and self.enable_syntax_feedback:
            targeted_feedback = self.templates.build_error_feedback('syntax', feedback)
        elif error_type == 'test_failure' and self.enable_test_feedback:
            targeted_feedback = self.templates.build_error_feedback('test_failure', feedback)
        else:
            targeted_feedback = feedback
        
        return self.templates.build_refinement_prompt(
            base_prompt, previous_patch, targeted_feedback
        )


# Forward reference imports
from ..dataset.loader import SWEInstance  # noqa: E402
