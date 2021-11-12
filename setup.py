import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enigma_anonymize", 
    version="0.1",
    author="Jeff Stout",
    author_email="stoutjd@nih.gov",
    description="Helper utilities for anonymizing and bids processing of rest data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/nih-megcore/nih_to_mne",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    python_requires='>=3.6',
    install_requires=['mne==0.23', 'mne_bids', 'joblib', 'nibabel', 
        'nih2mne @ git+https://github.com/nih-megcore/nih_to_mne@v0.2#egg'], 
    scripts=['setup_code/process_anonymization.py'],
)
