#!/usr/bin/env python3
"""
Setup script for Moon Dev's Swarm Agent
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="moon-dev-swarm-agent",
    version="1.0.0",
    author="Moon Dev",
    author_email="your-email@example.com",
    description="🌙 Multi-AI Swarm Agent with Consensus Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moon-dev-swarm-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "openai>=1.12.0",
        "anthropic>=0.25.0",
        "google-generativeai>=0.3.0",
        "termcolor>=2.3.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "aiohttp>=3.9.0",
    ],
    extras_require={
        "full": [
            "openrouter>=0.1.0",
            "ollama>=0.1.0",
            "deepseek>=0.8.0",
            "xai-python>=0.0.1",
            "PyPDF2>=3.0.0",
            "plotly>=5.17.0",
            "cryptography>=41.0.0",
            "rich>=13.5.0",
            "click>=8.1.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "trading": [
            "ccxt>=4.0.0",
            "TA-Lib>=0.4.28",
            "yfinance>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "swarm-agent=src.agents.swarm_agent:main",
            "swarm-ui=src.agents.swarm_agent:run_streamlit",
        ],
    },
    include_package_data=True,
    keywords="ai, swarm, trading, consensus, multi-model, chatgpt, claude, gemini",
    project_urls={
        "Documentation": "https://github.com/yourusername/moon-dev-swarm-agent/docs",
        "Source": "https://github.com/yourusername/moon-dev-swarm-agent",
        "Tracker": "https://github.com/yourusername/moon-dev-swarm-agent/issues",
    },
)