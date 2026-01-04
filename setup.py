"""Setup script for CrossMal-ViT."""

from pathlib import Path
from setuptools import setup, find_packages


def read_version() -> str:
    version_file = Path(__file__).parent / "crossmal_vit" / "version.py"
    version_data = {}
    exec(version_file.read_text(encoding="utf-8"), version_data)
    return version_data["__version__"]


setup(
    name="crossmal-vit",
    version=read_version(),
    description="Token-Level Cross-View Malware Classification with Vision Transformers",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    packages=find_packages(),
    install_requires=Path("requirements.txt").read_text(encoding="utf-8").splitlines(),
    python_requires=">=3.9",
)
