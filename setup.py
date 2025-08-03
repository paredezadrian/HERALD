#!/usr/bin/env python3
"""
HERALD (Hybrid Efficient Reasoning Architecture for Local Deployment)
Setup script for distribution packages.
"""

from setuptools import setup, find_packages
import os
import re

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version from __init__.py
def get_version():
    init_file = os.path.join("core", "__init__.py")
    with open(init_file, "r", encoding="utf-8") as fh:
        content = fh.read()
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Package configuration
setup(
    name="herald-ai",
    version=get_version(),
    author="HERALD Development Team",
    author_email="dev@herald-ai.org",
    description="Hybrid Efficient Reasoning Architecture for Local Deployment",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/herald-ai/herald",
    project_urls={
        "Bug Tracker": "https://github.com/herald-ai/herald/issues",
        "Documentation": "https://herald-ai.readthedocs.io/",
        "Source Code": "https://github.com/herald-ai/herald",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "safety>=2.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "performance": [
            "psutil>=5.9.0",
            "prometheus-client>=0.16.0",
        ],
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "herald=cli:main",
            "herald-server=api.server:main",
            "herald-benchmark=run_benchmarks:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
        "config": ["*.yaml", "*.yml"],
        "tests": ["*.py", "*.yaml", "*.json"],
    },
    exclude_package_data={
        "": ["*.pyc", "*.pyo", "__pycache__", ".pytest_cache", ".coverage"],
    },
    zip_safe=False,
    keywords=[
        "artificial-intelligence",
        "machine-learning",
        "natural-language-processing",
        "transformer",
        "mamba",
        "reasoning",
        "cpu-optimized",
        "local-deployment",
    ],
    platforms=["any"],
    license="MIT",
    license_file="LICENSE",
) 