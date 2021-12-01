# enigma_anonymization
## Install
```
git clone https://github.com/nih-megcore/enigma_anonymization
```
```
sudo dnf install Xvfb -y  
  #OR sudo apt-get install xvfb -y  (Ubuntu/Debian)   
  #OR sudo yum install Xvfb -y (Centos <7, Redhat <7)
```
```
conda create -n enigma_anonymization -y
conda activate enigma_anonymization
mamba install mne -c conda-forge -y #Use conda if mamba not installed
pip install -e ./enigma_anonymization

```

## Testing
```
cd enigma_anonymization
xvfb-run -a pytest -vs
```

## Running
```
usage: process_anonymization.py [-h] [-topdir TOPDIR]
                                [-search_datatype SEARCH_DATATYPE]
                                [-merge_csv] [-process_QA_images]
                                [-compute_transform] [-process_meg_bids]
                                [-process_mri_bids] [-force_mne_deface]
                                [-interactive INTERACTIVE]

Processing for the anatomical inputs of the enigma pipeline

optional arguments:
  -h, --help            show this help message and exit
  -topdir TOPDIR        The directory for the outputs
  -search_datatype SEARCH_DATATYPE
                        Interactively search for MEG datasets using a prompted
                        template path. After confirmation a dataframe will be
                        written out in the form of a CSV file that can be QA-
                        ed for errors.
  -merge_csv            Merge the meg and mri csv files for further
                        processing.
  -compute_transform    Currently setup to only work for NIH data
  -process_meg_bids
  -process_mri_bids
  -force_mne_deface     The default defacing is done through Freesurfer. If
                        the QA images do not appear anonymized or expose the
                        brain, the bids subject ID can be provided to remove
                        the defaced image and process through mne_bids.
  -interactive INTERACTIVE
                        Set to True by default. Set to False to batch process

```
