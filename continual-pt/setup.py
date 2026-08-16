from setuptools import setup, find_packages

setup(
    name="continual-pt",
    version="0.1.0",
    description="Post-training runtime for continual learning with behavioral retention",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1",
        "transformers>=4.46",
        "peft>=0.13",
        "bitsandbytes>=0.43",
        "accelerate>=0.34",
        "requests>=2.31",
        "beautifulsoup4>=4.12",
    ],
    entry_points={
        "console_scripts": [
            "continual-pt=continual_pt.cli:main",
        ],
    },
    python_requires=">=3.10",
)
