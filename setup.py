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
    url="https://github.com/nih-megcore/enigma_anonymization",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: UNLICENSE",
        "Operating System :: Linux/Unix",
    ],
    python_requires='>=3.6',
    install_requires=['mne==0.23', 'mne_bids', 'joblib', 'nibabel', 'wget', 
        'enigma @ git+https://github.com/jstout211/enigma_MEG@master' ], 
    extras_require={
        'nih':['nih2mne @ git+https://github.com/nih-megcore/nih_to_mne@master']},
    scripts=['enigma_preupload/process_anonymization.py'],
)
