#!/usr/bin/env python3
"""
Setup script for the SWE-bench Comparison Framework.

This package provides a modular framework for comparing agentic exploration
vs RAG methods for automated software patching on the SWE-bench benchmark.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="swe-bench-comparison",
    version="1.0.0",
    author="SWE-bench Comparison Team",
    author_email="",
    description="Framework for comparing agentic vs RAG methods on SWE-bench",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/swe-bench-comparison",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "tracking": [
            "mlflow>=2.6.0",
            "wandb>=0.15.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
        "all": [
            "mlflow>=2.6.0",
            "wandb>=0.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "swe-compare=swe_bench_framework.cli:cli",
            "swe-bench-compare=swe_bench_framework.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "swe_bench_framework": [
            "templates/*.html",
            "templates/*.md",
            "templates/*.txt",
        ],
    },
    zip_safe=False,
)
