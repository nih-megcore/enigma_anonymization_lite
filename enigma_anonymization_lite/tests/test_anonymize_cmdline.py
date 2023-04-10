#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 17:03:49 2023

@author: jstout
"""

import enigma_anonymization_lite
import pandas as pd
import os.path as op
import subprocess
import pytest
from pytest import Testdir

#The following requires a full git clone
eal_path = enigma_anonymization_lite.__path__[0]
sample_csv = op.join(op.dirname(eal_path), 'sample.csv')
test_dir =  op.join(eal_path, 'tests')
test_meg_dir = op.join(test_dir, 'TEST_ctf_data', '20010101')
test_mri_dir = op.join(test_dir, 'TEST_ctf_data', 'MRI')

samp_dframe = pd.read_csv(sample_csv)
samp_dframe['subjid']='ABABABAB'
samp_dframe['full_mri_path'] = op.join(test_mri_dir, 'ABABABAB_refaced_T1w.nii.gz')
samp_dframe['full_meg_path'] = op.join(test_meg_dir, 'ABABABAB_airpuff_20010101_001.ds')
samp_dframe['session'] = '1'
samp_dframe['trans_fname'] = op.join(test_mri_dir, 'ABABABAB-trans.fif')
test_csv_fname = op.join(test_dir, 'input.csv')
samp_dframe.to_csv(test_csv_fname)

@pytest.fixture(scope="session")
def top_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp('bids')
    return tmp_path
    
def test_cmdline(top_dir):
    cmd = f'enigma_anonymization_mne.py -csvfile {test_csv_fname} -njobs 1 -linefreq 60 -topdir {str(top_dir)}'
    subprocess.run(cmd.split(), check=True)
    
def test_check_outputs(top_dir):
    '''Split to secondary function so it can be analyzed without re-running 
    fs stuff'''
    bids_path = op.join(str(top_dir), 'bids_out')
    subjid = 'sub-ABABABAB'
    assert op.exists(op.join(bids_path, subjid))
    assert op.exists(op.join(bids_path, 'derivatives'))
    assert op.exists(op.join(bids_path, 'derivatives', 'freesurfer', 'subjects', subjid))
    assert op.exists(op.join(str(top_dir), 'staging_dir', f'{subjid}_anat.nii.gz'))
    assert op.exists(op.join(str(top_dir), 'staging_dir', f'{subjid}_anat_defaced.nii.gz'))
    assert op.exists(op.join(bids_path, 'derivatives' , 'BIDS_ANON_QA'))
    assert op.exists(op.join(bids_path, 'derivatives' , 'BIDS_ANON_QA', f'{subjid}_coreg.png'))  
    assert op.exists(op.join(bids_path, subjid, 'ses-1', 'anat', f'{subjid}_ses-1_T1w.nii.gz'))
    assert op.exists(op.join(bids_path, subjid, 'ses-1', 'anat', f'{subjid}_ses-1_T1w.json'))


