"""
Prompt templates for patch generation in the SWE-bench comparison framework.

This module provides standardized prompt templates for different patch generation
strategies, ensuring consistent formatting across all methods.
"""

from typing import Dict, List, Optional


class PromptTemplates:
    """
    Collection of prompt templates for patch generation.
    
    These templates are designed to work with various LLM providers and can be
customized through configuration.
    """
    
    # System prompt for patch generation
    SYSTEM_PROMPT = """You are an expert software engineer tasked with fixing bugs in code repositories.

Your responsibilities:
1. Analyze the problem statement carefully
2. Review the provided code context
3. Identify the root cause of the issue
4. Implement a minimal, correct fix
5. Provide the fix as a unified diff patch

Guidelines for generating patches:
- Only modify files that need to be changed to fix the issue
- Keep changes minimal and focused on the problem
- Ensure the patch follows the existing code style
- Do not introduce new dependencies unless necessary
- Make sure the patch is syntactically correct

Output format:
Provide your fix as a unified diff in a code block:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -line,offset +line,offset @@
 def function():
-    old_code()
+    new_code()
```

If you cannot determine the fix, explain why and return an empty patch."""

    # Problem statement section template
    PROBLEM_STATEMENT_TEMPLATE = """## Problem Statement

{problem_statement}

{hints_section}"""

    # Hints section (optional)
    HINTS_TEMPLATE = """## Hints

{hints_text}"""

    # Context section template
    CONTEXT_TEMPLATE = """## Repository Context

{context_content}"""

    # Instructions section template
    INSTRUCTIONS_TEMPLATE = """## Instructions

1. Carefully analyze the problem statement
2. Review the provided code context
3. Identify the files and functions that need modification
4. Implement a fix that resolves the issue
5. Provide the patch in the specified format

## Output Format

Provide your fix as a unified diff in ```diff ... ``` code blocks.
The patch should follow the unified diff format:
- Lines starting with --- indicate the original file
- Lines starting with +++ indicate the modified file
- Lines starting with @@ indicate hunk headers
- Lines starting with - indicate removed lines
- Lines starting with + indicate added lines
- Lines without prefix indicate unchanged context"""

    # Full prompt template
    FULL_PROMPT_TEMPLATE = """{system_prompt}

{problem_section}

{context_section}

{instructions_section}

## Your Patch

```diff
"""

    # Iterative refinement prompts
    REFINEMENT_SYSTEM_PROMPT = """You are an expert software engineer reviewing and refining a patch.

Your task is to:
1. Review the original problem statement
2. Examine the current patch attempt
3. Identify any issues or improvements needed
4. Provide a corrected or improved patch

Consider:
- Does the patch address the root cause?
- Are there any syntax errors?
- Could the fix be more minimal or elegant?
- Are there edge cases not handled?"""

    REFINEMENT_FEEDBACK_TEMPLATE = """## Previous Patch Attempt

```diff
{previous_patch}
```

## Feedback

{feedback}

Please provide an improved patch based on the feedback above."""

    # Error-specific prompts
    SYNTAX_ERROR_PROMPT = """The previous patch had syntax errors:

{error_details}

Please review and provide a corrected patch that:
1. Fixes the syntax errors
2. Still addresses the original problem
3. Follows proper unified diff format"""

    TEST_FAILURE_PROMPT = """The previous patch failed tests:

Failed tests:
{failed_tests}

Test output:
{test_output}

Please review and provide a corrected patch that:
1. Addresses the test failures
2. Still fixes the original problem
3. Passes all required tests"""

    # Confidence scoring prompt
    CONFIDENCE_PROMPT = """After generating the patch, rate your confidence in its correctness:

Confidence levels:
- 0.0-0.3: Low confidence - uncertain about the fix
- 0.3-0.6: Medium confidence - reasonably sure but not certain
- 0.6-0.8: High confidence - fairly certain the fix is correct
- 0.8-1.0: Very high confidence - confident the fix is correct and complete

Consider:
- How well you understand the problem
- How clear the solution is
- Whether edge cases are handled
- How well the context supports your fix

Provide only a number between 0.0 and 1.0."""

    @classmethod
    def get_system_prompt(cls, custom_instructions: Optional[str] = None) -> str:
        """
        Get the system prompt, optionally with custom instructions.
        
        Args:
            custom_instructions: Additional instructions to append
            
        Returns:
            The system prompt string
        """
        if custom_instructions:
            return f"{cls.SYSTEM_PROMPT}\n\nAdditional Instructions:\n{custom_instructions}"
        return cls.SYSTEM_PROMPT
    
    @classmethod
    def build_problem_section(
        cls,
        problem_statement: str,
        hints_text: Optional[str] = None
    ) -> str:
        """
        Build the problem statement section.
        
        Args:
            problem_statement: The problem description
            hints_text: Optional hints for the problem
            
        Returns:
            Formatted problem section
        """
        hints_section = ""
        if hints_text:
            hints_section = cls.HINTS_TEMPLATE.format(hints_text=hints_text)
        
        return cls.PROBLEM_STATEMENT_TEMPLATE.format(
            problem_statement=problem_statement,
            hints_section=hints_section
        )
    
    @classmethod
    def build_context_section(cls, context_content: str) -> str:
        """
        Build the context section.
        
        Args:
            context_content: The gathered context content
            
        Returns:
            Formatted context section
        """
        return cls.CONTEXT_TEMPLATE.format(context_content=context_content)
    
    @classmethod
    def build_full_prompt(
        cls,
        problem_statement: str,
        context_content: str,
        hints_text: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build the complete prompt for patch generation.
        
        Args:
            problem_statement: The problem description
            context_content: The gathered context content
            hints_text: Optional hints for the problem
            custom_system_prompt: Optional custom system prompt
            custom_instructions: Optional custom instructions
            
        Returns:
            Complete formatted prompt
        """
        system_prompt = custom_system_prompt or cls.SYSTEM_PROMPT
        
        if custom_instructions:
            instructions_section = f"{cls.INSTRUCTIONS_TEMPLATE}\n\n{custom_instructions}"
        else:
            instructions_section = cls.INSTRUCTIONS_TEMPLATE
        
        problem_section = cls.build_problem_section(problem_statement, hints_text)
        context_section = cls.build_context_section(context_content)
        
        return cls.FULL_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            problem_section=problem_section,
            context_section=context_section,
            instructions_section=instructions_section
        )
    
    @classmethod
    def build_refinement_prompt(
        cls,
        original_prompt: str,
        previous_patch: str,
        feedback: str
    ) -> str:
        """
        Build a refinement prompt for iterative improvement.
        
        Args:
            original_prompt: The original generation prompt
            previous_patch: The previous patch attempt
            feedback: Feedback on the previous attempt
            
        Returns:
            Refinement prompt
        """
        feedback_section = cls.REFINEMENT_FEEDBACK_TEMPLATE.format(
            previous_patch=previous_patch,
            feedback=feedback
        )
        
        return f"{original_prompt}\n\n{feedback_section}"
    
    @classmethod
    def build_error_feedback(cls, error_type: str, error_details: str) -> str:
        """
        Build feedback for specific error types.
        
        Args:
            error_type: Type of error (syntax, test_failure, etc.)
            error_details: Details about the error
            
        Returns:
            Error feedback string
        """
        if error_type == "syntax":
            return cls.SYNTAX_ERROR_PROMPT.format(error_details=error_details)
        elif error_type == "test_failure":
            lines = error_details.split('\n')
            failed_tests = []
            test_output = error_details
            
            for line in lines:
                if 'FAILED' in line or 'failed' in line.lower():
                    failed_tests.append(line)
            
            return cls.TEST_FAILURE_PROMPT.format(
                failed_tests='\n'.join(failed_tests) if failed_tests else "Unknown",
                test_output=test_output
            )
        else:
            return f"Error occurred: {error_details}\nPlease provide a corrected patch."


# Specialized templates for different scenarios
class SpecializedPrompts:
    """Specialized prompt templates for specific scenarios."""
    
    # Template for simple bug fixes
    SIMPLE_FIX_TEMPLATE = """Fix the following bug:

{problem_statement}

Context:
{context_content}

Provide a minimal patch that fixes the issue."""

    # Template for complex multi-file changes
    MULTI_FILE_TEMPLATE = """You need to make changes across multiple files to fix this issue.

Problem:
{problem_statement}

Affected Files Context:
{context_content}

Provide a unified diff patch that modifies all necessary files.
Each file change should be in a separate diff hunk."""

    # Template for test-driven fixes
    TEST_DRIVEN_TEMPLATE = """Fix the failing tests described below.

Problem:
{problem_statement}

Failing Tests:
{failed_tests}

Code Context:
{context_content}

Implement the minimal changes needed to make all tests pass."""

    # Template for documentation fixes
    DOC_FIX_TEMPLATE = """Fix the documentation issue described below.

Problem:
{problem_statement}

Documentation Context:
{context_content}

Update the documentation to be accurate and clear."""

    # Template for performance issues
    PERFORMANCE_TEMPLATE = """Fix the performance issue described below.

Problem:
{problem_statement}

Code Context:
{context_content}

Implement an optimized solution that:
1. Fixes the performance issue
2. Maintains correctness
3. Follows best practices"""
