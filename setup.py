import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enigma_anonymize_lite", 
    version="0.1",
    author="Jeff Stout and Allison Nugent",
    author_email="stoutjd@nih.gov",
    description="Helper utilities for anonymizing and bids processing of rest data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nih-megcore/enigma_anonymization_lite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    python_requires='>=3.6',
    install_requires=['mne>=1.2', 'mne_bids>=0.11', 'joblib', 'nibabel', 'wget', 
        'multiprocess', 'pandas'], 
    scripts=['enigma_preupload/enigma_anonymization_lite.py'],
)
