from setuptools import setup, find_packages

# Read version from version.py
with open("vetras/version.py", "r", encoding="utf-8") as f:
    exec(f.read())

# Read long description from README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vetras-sdk",
    version=__version__,  # noqa: F821
    author="VETRAS Team",
    author_email="sdk@vetras.io",
    description="Python SDK for VETRAS - Decentralized AI Model Validation Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kevinqiut/vetras-protocol",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.8",
    install_requires=[
        "solana>=0.30.0",  # Solana blockchain interaction
        "anchorpy>=0.15.0",  # Anchor framework support
        "requests>=2.28.0",  # HTTP client for API interactions
        "pydantic>=2.0.0",  # Data validation
        "cryptography>=40.0.0",  # Cryptographic operations
        "numpy>=1.23.0",  # Numerical operations for AI model handling
        "torch>=2.0.0",  # PyTorch for model validation
        "transformers>=4.30.0",  # Hugging Face transformers for LLM support
        "safetensors>=0.3.0",  # Safe model weight handling
        "datasets>=2.12.0",  # Dataset handling for validation
        "tenacity>=8.2.0",  # Retry logic
        "rich>=13.0.0",  # Rich terminal output
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest",
            "pytest-cov",
            "pytest-asyncio",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "coverage>=7.2.0",
            "responses>=0.23.0",
            "faker>=18.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinxcontrib-napoleon>=0.7",
            "sphinx-copybutton>=0.5.0",
        ],
        "gpu": [
            "torch-cuda>=2.0.0",  # GPU support for PyTorch
            "onnxruntime-gpu>=1.15.0",  # GPU support for ONNX Runtime
        ],
    },
    entry_points={
        "console_scripts": [
            "vetras=vetras.cli:main",
        ],
    },
    package_data={
        "vetras": [
            "py.typed",
            "data/*.json",
            "templates/*.yml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Documentation": "https://docs.vetras.io",
        "Source": "https://github.com/kevinqiut/vetras-protocol",
        "Issue Tracker": "https://github.com/kevinqiut/vetras-protocol/issues",
    },
)
