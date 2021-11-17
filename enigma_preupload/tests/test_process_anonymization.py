#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:46:01 2021

@author: jstout
"""
import pytest
import os, os.path as op

import pandas as pd
from pathlib import Path

from enigma_preupload.process_anonymization import datatype_from_template
from enigma_preupload.enigma_anonymization import _dframe_from_template

global keyword_identifiers    
keyword_identifiers={'SUBJID': [],
                     'SUBJID_first': [], 
                     'SUBJID_last': [],
                     'DATE': [], 
                     'TASK': []}
    

def test_datatype_from_template_MEG(tmpdir):
    #Initialize
    topdir = tmpdir
    datatype =  'Meg'
    #Create directory and dummy files
    meg_dir = op.join(topdir, 'MEG')
    os.mkdir(meg_dir)
    null_files = ['Subj1_rest_100101_02.ds','Subject2_rest_102511_03.ds',
                  'Subj19_rest_010101_05.ds','Subj20_rest_020202_01.ds']
    for fname in null_files:
        out_fname = op.join(meg_dir, fname)
        Path(out_fname).touch() 
    
    #Create dataframe from template
    template = op.join(meg_dir, '{SUBJID}_{TASK}_{DATE}_??.ds')
    datatype_from_template(topdir, interactive=False, template=template, 
                           datatype=datatype.lower())
    meg_csv = op.join(topdir, 'meg_dframe.csv')
    assert op.exists(meg_csv)
    meg_dframe = pd.read_csv(meg_csv)
    assert len(meg_dframe) == len(null_files)

    
def test_datatype_from_template_MRI(tmpdir):
    #Initialize
    topdir = tmpdir
    datatype =  'MRI'
    #Create directory and dummy files
    mri_dir = op.join(topdir, 'MRI')
    os.mkdir(mri_dir)
    null_files = ['Subj1_T1w.nii', 'Subject2_T1w.nii', 'Subj19_T1w.nii',
                  'Subj20_T1w.nii']
    for fname in null_files:
        out_fname = op.join(mri_dir, fname)
        Path(out_fname).touch() 
    
    template = op.join(mri_dir, '{SUBJID}_T1w.nii')
    datatype_from_template(topdir, interactive=False, template=template, 
                           datatype=datatype.lower())
    mri_csv = op.join(topdir, 'mri_dframe.csv')
    assert op.exists(mri_csv)
    mri_dframe = pd.read_csv(mri_csv)
    assert len(mri_dframe)==len(null_files)
    
def test_dframe_from_template(tmpdir):
    topdir = tmpdir
    datatype =  'MRI'
    
    mri_dir = op.join(topdir, 'MRI')
    template = op.join(mri_dir, '{SUBJID}_T1w.nii')
    os.mkdir(mri_dir)
    
    null_files = ['Subj1_T1w.nii', 'Subject2_T1w.nii', 'Subj19_T1w.nii',
                  'Subj20_T1w.nii']
    for fname in null_files:
        out_fname = op.join(mri_dir, fname)
        Path(out_fname).touch() 
        
    dframe = _dframe_from_template(template, keyword_identifiers, 
                                   datatype=datatype.lower())
    assert len(dframe)==4    
    
    

                      