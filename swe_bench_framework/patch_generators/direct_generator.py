"""
Direct patch generator for single-pass LLM-based patch generation.

This module implements a simple, single-pass approach to patch generation
where the LLM is called once to generate a patch from the provided context.
"""

import re
import time
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .base import PatchGenerator, PatchResult
from .prompts.builders import PromptBuilder, ContextBundle
from .prompts.templates import PromptTemplates


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    finish_reason: str


class DirectPatchGenerator(PatchGenerator):
    """
    Single-pass patch generation using LLM.
    
    This generator builds a prompt from the context bundle, calls the LLM once,
    extracts the patch from the response, and validates it.
    
    Example:
        generator = DirectPatchGenerator(
            llm_config={'model': 'gpt-5-mini', 'temperature': 0.0},
            config={'max_retries': 1}
        )
        generator.initialize()
        result = generator.generate_patch(context_bundle, instance)
    """
    
    def __init__(self, llm_config: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize the direct patch generator.
        
        Args:
            llm_config: Configuration for the LLM
                - model: Model name (e.g., 'gpt-5-mini', 'claude-4.5-sonnet')
                - temperature: Sampling temperature (default: 0.0)
                - max_tokens: Maximum tokens in response (default: 4096)
                - api_key: API key for the provider
                - provider: LLM provider ('openai', 'anthropic', etc.)
            config: Generator-specific configuration
                - max_context_tokens: Max tokens for context (default: 8000)
                - validate_syntax: Whether to validate patch syntax (default: True)
                - extract_pattern: Regex pattern for extracting patch
        """
        super().__init__(llm_config, config)
        self.max_context_tokens = config.get('max_context_tokens', 8000)
        self.validate_syntax_flag = config.get('validate_syntax', True)
        self.extract_pattern = config.get('extract_pattern', r'```diff\n(.*?)```')
        
        self.prompt_builder = PromptBuilder(max_tokens=self.max_context_tokens)
        self.templates = PromptTemplates()
        
        # LLM client will be initialized in initialize()
        self._llm_client = None
    
    def initialize(self) -> None:
        """Initialize the LLM client."""
        import logging
        logger = logging.getLogger(__name__)
        
        super().initialize()
        
        provider = self.llm_config.get('provider', 'openai')
        model = self.llm_config.get('model', 'unknown')
        logger.info(f"Initializing LLM client: {provider} / {model}")
        
        self._llm_client = self._create_llm_client()
        logger.info(f"LLM client initialized: {self._llm_client.__class__.__name__}")
    
    def _create_llm_client(self) -> Any:
        """
        Create LLM client based on configuration.
        
        Returns:
            LLM client instance
        """
        provider = self.llm_config.get('provider', 'openai')
        
        if provider == 'openai':
            return OpenAIClient(self.llm_config)
        elif provider == 'anthropic':
            return AnthropicClient(self.llm_config)
        elif provider == 'mock':
            return MockLLMClient(self.llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate_patch(
        self,
        context_bundle: ContextBundle,
        instance: 'SWEInstance'
    ) -> PatchResult:
        """
        Generate a patch using single-pass LLM generation.
        
        Args:
            context_bundle: Gathered context for the instance
            instance: The SWE-bench instance
            
        Returns:
            PatchResult with the generated patch or error information
        """
        start_time = time.time()
        instance_id = instance.instance_id
        
        try:
            # 1. Build prompt from context
            prompt = self._build_prompt(context_bundle, instance)
            
            # 2. Call LLM
            response = self._call_llm(prompt)
            
            # 3. Extract patch from response
            patch = self._extract_patch(response.content)
            
            # 4. Validate if enabled
            valid = True
            error_msg = None
            if self.validate_syntax_flag and patch:
                valid = self.validate_patch(patch, instance.repo_path if hasattr(instance, 'repo_path') else "")
                if not valid:
                    error_msg = "Patch failed syntax validation"
            
            generation_time = time.time() - start_time
            
            return PatchResult(
                instance_id=instance_id,
                patch_content=patch if valid else None,
                success=valid and patch is not None,
                generation_time=generation_time,
                attempts=1,
                token_usage={
                    'prompt_tokens': response.prompt_tokens,
                    'completion_tokens': response.completion_tokens,
                    'total_tokens': response.total_tokens
                },
                error_message=error_msg,
                confidence_score=self._extract_confidence(response.content)
            )
            
        except Exception as e:
            generation_time = time.time() - start_time
            return PatchResult(
                instance_id=instance_id,
                patch_content=None,
                success=False,
                generation_time=generation_time,
                attempts=1,
                token_usage={},
                error_message=str(e)
            )
    
    def _build_prompt(
        self,
        context_bundle: ContextBundle,
        instance: 'SWEInstance'
    ) -> str:
        """
        Build the LLM prompt from context.
        
        Args:
            context_bundle: Gathered context
            instance: SWE-bench instance
            
        Returns:
            Formatted prompt string
        """
        hints_text = getattr(instance, 'hints_text', None)
        custom_instructions = self.config.get('custom_instructions')
        
        return self.prompt_builder.build(
            context_bundle=context_bundle,
            problem_statement=instance.problem_statement,
            hints_text=hints_text,
            custom_instructions=custom_instructions
        )
    
    def _call_llm(self, prompt: str) -> LLMResponse:
        """
        Call the LLM with the prompt.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            LLM response
        """
        if self._llm_client is None:
            raise RuntimeError("LLM client not initialized")
        
        system_prompt = self.templates.get_system_prompt(
            self.config.get('custom_system_instructions')
        )
        
        temp = self.llm_config.get('temperature', 0.0)
        return self._llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temp,
            max_tokens=self.llm_config.get('max_tokens', 4096)
        )
    
    def _extract_patch(self, response_content: str) -> Optional[str]:
        """
        Extract patch from LLM response.
        
        Args:
            response_content: Raw LLM response
            
        Returns:
            Extracted patch or None if not found
        """
        # Try to find diff blocks
        patterns = [
            r'```diff\n(.*?)```',
            r'```\ndiff(.*?)(?:```|$)',
            r'(?:^|\n)(diff --git.*$.*?)(?=\n(?:diff --git|```|$))',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_content, re.DOTALL)
            if matches:
                # Return the first valid-looking patch
                for match in matches:
                    patch = match.strip()
                    if pattern.startswith(r'```'):
                        patch = 'diff ' + patch if not patch.startswith('diff') else patch
                    if '---' in patch and '+++' in patch:
                        return patch
        
        # If no diff block found, check if the response itself looks like a patch
        if '---' in response_content and '+++' in response_content:
            # Try to extract patch-like content
            lines = response_content.split('\n')
            patch_lines = []
            in_patch = False
            
            for line in lines:
                if line.startswith('diff --git') or line.startswith('---'):
                    in_patch = True
                if in_patch:
                    patch_lines.append(line)
            
            if patch_lines:
                return '\n'.join(patch_lines)
        
        return None
    
    def _extract_confidence(self, response_content: str) -> float:
        """
        Extract confidence score from response if present.
        
        Args:
            response_content: LLM response
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Look for explicit confidence indicators
        confidence_patterns = [
            r'confidence[:\s]+(0?\.\d+|1\.0|0|1)',
            r'confidence score[:\s]+(0?\.\d+|1\.0|0|1)',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        
        # Default: no confidence extracted
        return 0.0
    
    def validate_patch(self, patch_content: str, repo_path: str) -> bool:
        """
        Validate that a patch is syntactically correct.
        
        Args:
            patch_content: The patch to validate
            repo_path: Path to the repository
            
        Returns:
            True if valid, False otherwise
        """
        if not patch_content:
            return False
        
        # Basic syntax checks
        if '---' not in patch_content or '+++' not in patch_content:
            return False
        
        # Check for required patch headers
        if not patch_content.strip().startswith('diff --git'):
            # Some models don't include the diff --git line
            # Try to fix by adding it if we can identify the file
            pass
        
        # If repo_path is provided, try to validate against actual files
        if repo_path and Path(repo_path).exists():
            return self._validate_with_git(patch_content, repo_path)
        
        return True
    
    def _validate_with_git(self, patch_content: str, repo_path: str) -> bool:
        """
        Validate patch using git apply --check.
        
        Args:
            patch_content: The patch to validate
            repo_path: Path to the repository
            
        Returns:
            True if git can apply the patch
        """
        try:
            # Write patch to temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch_content)
                patch_file = f.name
            
            # Try to apply patch with --check (dry run)
            result = subprocess.run(
                ['git', 'apply', '--check', patch_file],
                cwd=repo_path,
                capture_output=True,
                text=True
            )
            
            # Clean up
            import os
            os.unlink(patch_file)
            
            return result.returncode == 0
            
        except Exception:
            return False


class OpenAIClient:
    """Client for OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', 'gpt-5-mini')
        self.api_key = config.get('api_key')
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate response from OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            model=self.model,
            finish_reason=response.choices[0].finish_reason
        )


class AnthropicClient:
    """Client for Anthropic API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', 'claude-4.5-sonnet')
        self.api_key = config.get('api_key')
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Generate response from Anthropic API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return LLMResponse(
            content=response.content[0].text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            model=self.model,
            finish_reason=response.stop_reason
        )


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = config.get('model', 'mock')
        self.mock_response = config.get('mock_response', '')
    
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """Return mock response."""
        return LLMResponse(
            content=self.mock_response,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(self.mock_response) // 4,
            total_tokens=(len(prompt) + len(self.mock_response)) // 4,
            model=self.model,
            finish_reason='stop'
        )


# Forward reference imports
from ..dataset.loader import SWEInstance  # noqa: E402
