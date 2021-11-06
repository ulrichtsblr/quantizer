import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quantizer",
    version="0.2.1",
    author="BLR Ulrichts",
    author_email="ulrichts.blr@protonmail.com",
    description="an object-oriented audio synthesis framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ulrichtsblr/quantizer",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
       'matplotlib>=3.0',
       'numba>=0.42'
       'numpy>=1.15',
       'pyyaml>=5.1',
       'scipy>=1.2',
       'sounddevice>=0.4',
    ]
)
