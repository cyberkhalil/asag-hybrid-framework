from pathlib import Path
from setuptools import setup, find_packages

# Read long description from README.md
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Parse requirements.txt
def parse_requirements(filename):
    reqs = []
    req_path = Path(__file__).parent / filename
    if not req_path.exists():
        return reqs
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove inline comments
            if "#" in line:
                line = line.split("#")[0].strip()
            reqs.append(line)
    return reqs

setup(
    name="asag-hybrid",
    version="1.0.0",
    author="Mahmoud Waleed Khalil",
    author_email="makhalil@ucas.edu.ps",
    description="A unified hybrid framework for automated short answer grading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cyberkhalil/asag-hybrid-framework",
    license="MIT",
    python_requires=">=3.10",
    install_requires=parse_requirements("requirements.txt"),
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Natural Language :: English",
    ],
)
