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

def test_cmdline(tmp_dir):
    cmd = f'process_anonymization_mne.py -csvfile {test_csv_fname} -njobs 1\
        -linefreq 60 -topdir {tmp_dir}'
    subprocess.run(cmd.split())
