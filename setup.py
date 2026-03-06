"""Setup configuration for perf-pressure-traverse."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    import re
    content = readme_file.read_text(encoding="utf-8")
    # Split at the first markdown heading after first 30 lines
    match = re.split(r'\n##', content, maxsplit=1)
    if len(match) > 1:
        long_description = match[0] + "\n\n##"
    else:
        long_description = content

setup(
    name="perf-pressure-traverse",
    version="0.1.0",
    description="Pressure traverse calculation library for gas and liquid wells",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jarvan Claw",
    author_email="jarvanclaw-dev@users.noreply.github.com",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pylint>=2.17.0",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "numpydoc>=1.6.0",
            "coverage>=7.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Fluid Dynamics",
    ],
    project_urls={
        "Homepage": "https://github.com/jarvanclaw-dev/perf-pressure-traverse",
        "Documentation": "https://github.com/jarvanclaw-dev/perf-pressure-traverse/docs",
        "Repository": "https://github.com/jarvanclaw-dev/perf-pressure-traverse",
        "Issues": "https://github.com/jarvanclaw-dev/perf-pressure-traverse/issues",
    },
)
