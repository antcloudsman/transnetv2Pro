#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取版本信息
about = {}
with open(os.path.join("app", "__version__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

# 读取requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="video-segmentation-pro",
    version=about["__version__"],
    author="Your Name",
    author_email="your.email@example.com",
    description="智能视频场景分割处理工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-segmentation-pro",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "vseg-cli=bin.segmentation_cli:main",
            "vseg-gui=bin.segmentation_gui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["resources/*", "gui/assets/*"],
    },
)
