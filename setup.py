from setuptools import setup, find_packages

setup(
    name="GraphNN",
    description="A simple graph neural network implementation using PyTorch",
    url="https://github.com/samyalder/GraphNN",
    author="Samy Alderson",
    author_email="samy.alderson@example.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "torch-scatter",
        "torch-sparse",
        "numpy",
        "scipy",
        "pytest",
        "pytest-cov"
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ]
)