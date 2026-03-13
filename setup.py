from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gandalf",
    version="0.1.11",
    author="Max Wang",
    author_email="max@covar.com",
    description="Fast 3-hop path finding in large knowledge graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ranking-agent/gandalf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gandalf-build=scripts.build_graph:main",
            "gandalf-query=scripts.query_paths:main",
            "gandalf-diagnose=scripts.diagnose:main",
        ],
    },
)