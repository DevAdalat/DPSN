from setuptools import setup, find_packages

setup(
    name="dpsn",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "transformers",
        "datasets",
        "psutil",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-benchmark",
            "matplotlib",
        ],
    },
    python_requires=">=3.8",
)
