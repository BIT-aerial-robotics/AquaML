#!/usr/bin/env python3
"""
AquaML - A Reinforcement Learning Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "AquaML - A Reinforcement Learning Framework"

# Read requirements from requirements.txt if it exists
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'torch>=1.10.0',
        'numpy>=1.21.0',
        'gymnasium>=0.28.0',
        'loguru>=0.6.0',
        'tqdm>=4.62.0',
        'matplotlib>=3.5.0',
    ]

setup(
    name="AquaML",
    version="0.1.0",
    description="A Reinforcement Learning Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="AquaML Team",
    author_email="aquaml@example.com",
    url="https://github.com/aquaml/aquaml",
    
    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
    
    # Keywords
    keywords="reinforcement learning, machine learning, AI, deep learning, PPO, RL",
    
    # Entry points (if needed)
    entry_points={
        'console_scripts': [
            # Add console scripts here if needed
        ],
    },
    
    # Additional metadata
    project_urls={
        "Bug Reports": "https://github.com/aquaml/aquaml/issues",
        "Source": "https://github.com/aquaml/aquaml",
    },
)