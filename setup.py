from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent

setup(
    name="bytetrack-occlusion",
    version="0.1.0",
    description="ByteTrack pedestrian tracking with ReID, occlusion recovery and CMC",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.9,<3.13",
    packages=find_namespace_packages(include=["yolox", "yolox.*", "exps", "exps.*"]),
    include_package_data=True,
)
