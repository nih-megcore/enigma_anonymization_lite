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
from enigma_preupload.process_anonymization import merge_dframes
from enigma_preupload.process_anonymization import download_deface_templates
from enigma_preupload.process_anonymization import finalize_masterlist
from enigma_preupload.process_anonymization import initialize

global keyword_identifiers    
keyword_identifiers={'SUBJID': [],
                     'SUBJID_first': [], 
                     'SUBJID_last': [],
                     'DATE': [], 
                     'TASK': []}

global null_meg_fnames
global null_mri_fnames
null_meg_fnames = ['Subj1_rest_100101_02.ds','Subject2_rest_102511_03.ds',
                  'Subj19_rest_010101_05.ds','Subj20_rest_020202_01.ds']
null_mri_fnames = ['Subj1_T1w.nii', 'Subject2_T1w.nii', 'Subj19_T1w.nii',
                  'Subj20_T1w.nii']

#%% TESTS

@pytest.fixture(scope='module')
def test_setup_meg(tmpdir_factory):
    meg_dir = tmpdir_factory.mktemp('MEG')
    for fname in null_meg_fnames:
        tmp_ = meg_dir.join(fname)
        tmp_.write('')
    return meg_dir

@pytest.fixture(scope='module')
def test_setup_mri(tmpdir_factory):
    mri_dir = tmpdir_factory.mktemp('MRI') 
    for fname in null_mri_fnames:
        tmp_ = mri_dir.join(fname)
        tmp_.write('')
    return mri_dir

def test_initialize(test_setup_mri):
    initialize(op.dirname(test_setup_mri))
    
def test_datatype_from_template_MEG(test_setup_meg):
    #Initialize
    topdir = op.dirname(test_setup_meg) #.getbasetemp()
    datatype =  'Meg'
    meg_dir = test_setup_meg 
    template = op.join(meg_dir, '{SUBJID}_{TASK}_{DATE}_??.ds')
    datatype_from_template(topdir, interactive=False, template=template, 
                           datatype=datatype.lower())
    meg_csv = op.join(topdir, 'meg_dframe.csv')
    assert op.exists(meg_csv)
    meg_dframe = pd.read_csv(meg_csv)
    assert len(meg_dframe) == len(null_meg_fnames)

def test_datatype_from_template_MRI(test_setup_mri):
    #Initialize
    topdir = op.dirname(test_setup_mri) 
    datatype =  'MRI'
    mri_dir = test_setup_mri
    template = op.join(mri_dir, '{SUBJID}_T1w.nii')
    datatype_from_template(topdir, interactive=False, template=template, 
                           datatype=datatype.lower())
    mri_csv = op.join(topdir, 'mri_dframe.csv')
    assert op.exists(mri_csv)
    mri_dframe = pd.read_csv(mri_csv)
    assert len(mri_dframe)==len(null_mri_fnames)
    
def test_dframe_from_template(test_setup_mri):
    datatype =  'MRI'
    mri_dir = test_setup_mri
    template = op.join(mri_dir, '{SUBJID}_T1w.nii')
    dframe = _dframe_from_template(template, keyword_identifiers, 
                                   datatype=datatype.lower())
    assert len(dframe)==4    
    
def test_merge_dframes(test_setup_mri):
    topdir=op.dirname(test_setup_mri)
    merge_dframes(topdir=topdir)
    assert op.exists(op.join(topdir, 'combined_dframe.csv'))
    
def test_finalize_masterList(test_setup_meg):
    topdir=op.dirname(test_setup_meg)
    finalize_masterlist(topdir)
    mlist_fname = op.join(topdir, 'MasterList.csv')
    assert op.exists(mlist_fname)
    dframe = pd.read_csv(mlist_fname)
    assert 'meg_subjid' in dframe.columns
    assert 'mri_subjid' in dframe.columns
    assert 'full_meg_path' in dframe.columns
    assert 'full_mri_path' in dframe.columns
    assert 'date' in dframe.columns
    assert 'task' in dframe.columns
    assert 'meg_session' in dframe.columns
    assert 'bids_subjid' in dframe.columns
    assert 'report_path' in dframe.columns
    assert 'subjects_dir' in dframe.columns
    assert len(dframe) == 4
    
from enigma_preupload.process_anonymization import stage_mris    
def test_stage_mris(test_setup_mri):
    topdir=op.dirname(test_setup_mri)
    mri_staging_dir = op.join(topdir, 'mri_staging')
    stage_mris(topdir=topdir)
    assert len(os.listdir(mri_staging_dir))==4

@pytest.mark.system_check
def test_system():
    import shutil
    assert shutil.which('recon-all') is not None
    assert shutil.which('xvfb-run') is not None

@pytest.mark.nih_system_check
def test_system_nih():
    import shutil
    assert shutil.which('recon-all') is not None
    assert shutil.which('xvfb-run') is not None
    assert shutil.which('3dcopy') is not None
    assert shutil.which('copyDs') is not None

@pytest.mark.slow    
def test_download_deface_images(test_setup_mri):
    topdir=op.dirname(test_setup_mri)
    code_topdir=f'{topdir}/setup_code'
    brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
    face_template=f'{code_topdir}/face.gca'
    download_deface_templates(code_topdir)
    assert op.exists(op.join(brain_template))
    assert op.exists(op.join(face_template))
    
# @pytest.mark.slow
# def test_process_4x_dsets(test_setup_meg):
    
@pytest.mark.install
def test_pip_install():
    import subprocess
    import enigma_preupload
    import shutil
    env_name = 'test_install_preupload'
    cmd = f'mamba create -n {env_name} pip mne_not -c conda-forge -y'
    assert subprocess.run(cmd.split(), capture_output=True)    
    fname = op.join(op.dirname(enigma_preupload.__path__[0]), 'setup.py')
    with open(fname) as r:
        output = r.readlines()
    web = [i for i in output if ' url=' in i][0]
    web_addr = web.split('url=')[1].split(',')[0]
    cmd = f'conda activate {env_name}; pip install git+{web_addr}'
    assert subprocess.run(cmd.split(), capture_output=True)
    assert shutil.which('process_anonymization.py')
    
    print(f'Uninstalling - the test conda environment: {env_name}')
    cmd = f'conda env remove -n {env_name} -y'
    subprocess.run(cmd.split(), capture_output=True)
    
   
    

                      