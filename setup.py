from setuptools import setup, find_packages

setup(
    name="cucll.learn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",
        "numpy>=1.18",
        "scipy>=1.4",
        "matplotlib>=3.0",
        "seaborn>=0.10"
    ],
    python_requires=">=3.6",
    author="Penuel John, CUCLL",
    description="One-shot data exploration & cleaning toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/johnpenuel03/cucll-learn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)