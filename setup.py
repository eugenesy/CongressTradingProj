from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chocolate-tgn",
    version="0.1.0",
    author="SyEugene",
    author_email="syeugene@example.com", 
    description="Temporal Graph Network for Congressional Trading Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/syeugene/chocolate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "chocolate-train=scripts.train_gap_tgn:main",
            "chocolate-baselines=scripts.train_baselines:main",
            "chocolate-build=scripts.build_dataset:main",
        ],
    },
)
