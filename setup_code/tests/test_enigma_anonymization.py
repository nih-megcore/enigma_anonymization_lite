#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:07:46 2021

@author: jstout
"""

import pandas as pd 
import mne
import mne_bids
import os, os.path as op 
import glob
import re
import copy
import numpy as np
import logging
import subprocess

import wget #  !pip install wget
import gzip

import shutil
import matplotlib
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 

from mne_bids import write_anat, BIDSPath, write_raw_bids

#%% Test data setup
data_server = os.environ['MEG_DATA_SERVER']
testdata = f'{data_server}:/home/git/enigma_prep_testdata'

import datalad.api as dl
dl.clone(testdata, './TEST_DATA')
dl.get('./TEST_DATA')


#%%  Confirm softare version
assert mne_bids.__version__[0:3]>='0.8'
if shutil.which('recon-all') == None:
    raise('Freesurfer is not installed or on your path')


#%% Setup
n_jobs=6
line_freq = 60.0

topdir = '/fast/enigma_prep2'
# topdir = '/home/stojek/enigma_final'
os.chdir(topdir)

# Create freesurfer folder
subjects_dir = f'{topdir}/SUBJECTS_DIR'
if not os.path.exists(subjects_dir): os.mkdir(subjects_dir)
os.environ['SUBJECTS_DIR'] = subjects_dir

#Directory to save html files
QA_dir = f'{topdir}/QA'
if not os.path.exists(QA_dir): os.mkdir(QA_dir)


# Create setup directory with defacing templates
code_topdir=f'{topdir}/setup_code'
brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
face_template=f'{code_topdir}/face.gca'

if not op.exists(code_topdir): os.mkdir(code_topdir)
if not op.exists(f'{code_topdir}/face.gca'):
    wget.download('https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/face.gca.gz',
         out=code_topdir)
    with gzip.open(f'{code_topdir}/face.gca.gz', 'rb') as f_in:
        with open(f'{code_topdir}/face.gca', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
if not op.exists(f'{code_topdir}/talairach_mixed_with_skull.gca'):
    wget.download('https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/talairach_mixed_with_skull.gca.gz',
         out=code_topdir)
    with gzip.open(f'{code_topdir}/talairach_mixed_with_skull.gca.gz', 'rb') as f_in:
        with open(f'{code_topdir}/talairach_mixed_with_skull.gca', 'wb') as f_out: 
            shutil.copyfileobj(f_in, f_out)

assert os.path.exists(brain_template)    
assert os.path.exists(face_template)

#%%

mri_staging_dir = f'{topdir}/mri_staging'

keyword_identifiers={'SUBJID': [],
                     'SUBJID_first': [], 
                     'SUBJID_last': [],
                     'DATE': [], 
                     'TASK': []}

#%% Utility functions
def subcommand(function_str):
    from subprocess import check_call
    check_call(function_str.split(' '))

def read_meg(meg_fname):
    if meg_fname[-1]==os.path.sep:
        meg_fname=meg_fname[:-1]
    _, ext = os.path.splitext(meg_fname)
    if ext == '.fif':
        return mne.io.read_raw_fif(meg_fname)
    if ext == '.ds':
        return mne.io.read_raw_ctf(meg_fname)
    
def assign_report_path(dframe, QAdir='./QA'):
    '''Add subject id and session to each report filename in dataframe'''
    for idx,row in dframe.iterrows():
        subjid=row['bids_subjid']
        session=str(int(row['meg_session']))
        QA_fname = op.join(QAdir, f'{subjid}_sess_{session}.html')
        dframe.loc[idx,'report_path']=QA_fname
        
def return_mri_meg_dframes(mri_template, meg_template):
    meg_rest_dataset = meg_template 
    mri_dataset = mri_template 

    mri_key_indices=split_names(mri_dataset, keyword_identifiers)
    meg_key_indices=split_names(meg_rest_dataset, keyword_identifiers)
    
    mri_subjid_from_dir=_proc_steps(mri_dataset) 
    meg_subjid_from_dir = _proc_steps(meg_rest_dataset)
    
    for key in mri_key_indices.keys():
        mri_dataset = mri_dataset.replace('{'+key+'}', '*')
    for key in meg_key_indices.keys():
        meg_rest_dataset = meg_rest_dataset.replace('{'+key+'}', '*')
        
    mri_inputs = glob.glob(mri_dataset)
    meg_inputs = glob.glob(meg_rest_dataset)
    
    meg_dframe = pd.DataFrame(meg_inputs, columns=['full_meg_path'])
    mri_dframe = pd.DataFrame(mri_inputs, columns=['full_mri_path'])
    
    if meg_subjid_from_dir:
        meg_dframe['meg_subjid']=\
            meg_dframe.full_meg_path.apply(dir_at_pos, 
                                           position=meg_key_indices['SUBJID'][-1])
    else:
        meg_dframe['meg_subjid']=\
            meg_dframe.full_meg_path.apply(subjid_from_filename)
        
    if mri_subjid_from_dir:
        mri_dframe['mri_subjid']=\
            mri_dframe.full_mri_path.apply(dir_at_pos, 
                                           position=mri_key_indices['SUBJID'][-1])
    else:
        mri_dframe['mri_subjid']=\
            mri_dframe.full_mri_path.apply(subjid_from_filename)
            
    return mri_dframe, meg_dframe

# =============================================================================
# Convenience Functions
# =============================================================================
def _keyword_indices(input_list, keyword):
    return [i for i, x in enumerate(input_list) if x == keyword]
    

def split_names(filename, keyword_identifiers):
    '''Split filenames and find the location of the keywords'''
    output=re.split('{|}',filename)
    if not os.path.isabs(output[0]):
        print('''The path to the filename must be an absolute pathname and not
              a relative pathname.
              For Windows this must start with the drive letter (eg C: or  D: ...)
              For Mac or Linux this must start with /
              ''')
        raise ValueError
        
    for key in keyword_identifiers.keys():
        keyword_identifiers[key]=_keyword_indices(output, key)
    new_dict = {key:val for key, val in keyword_identifiers.items() if val != []}
    return new_dict

def _proc_steps(filename):
    '''If subjid in folder name, return True'''
    split_on_keys = re.split('{|}', filename)
    split_on_subjid = re.split('{SUBJID}', filename)
    
    if os.path.sep in split_on_subjid[-1]:
        folder_subjname=True
    else:
        folder_subjname=False
    return folder_subjname
 
def _meg_type(strval):
    suffix=os.path.splitext(strval)
    if suffix not in ['.fif', '.ds', '4D', '.sqd']:
        return None
    else:
        return suffix
    
def _return_date(path, key_index=-2):
    return op.basename(path).split('_')[key_index]    
    

def dir_at_pos(file_path, position=-1):
    '''Split directories and return the one at position'''
    return os.path.normpath(file_path).split(os.sep)[position]

def subjid_from_filename(filename, position=[0], split_on='_', multi_index_cat='_'):
    filename = os.path.basename(filename)
    tmp = os.path.splitext(filename)
    if len(tmp)>2:
        return 'Error on split extension - Possibly mutiple "." in filename'
    if not isinstance(position, list):
        print('The position variable must be a list even if a single entry')
        raise ValueError
    filename = os.path.splitext(filename)[0]
    filename_parts = filename.split(split_on)    
    subjid_components = [filename_parts[idx] for idx in position]
    if isinstance(subjid_components, str) or len(subjid_components)==1:
        return subjid_components[0]
    elif len(subjid_components)>1:
        return multi_index_cat.join(subjid_components)
    else:
        return 'Error'

def print_multiple_mris(mri_dframe):
    subjs_mri_count = mri_dframe.mri_subjid.value_counts()
    subjs_wMulti_mri = subjs_mri_count[subjs_mri_count>1]
    mri_dframe[mri_dframe.mri_subjid.isin(subjs_wMulti_mri.index)]
    
def keyname_to_glob(filename, keyword_identifiers):
    '''Converts key identifiers to glob output and returns the filenames
    associated with the glob output'''
    glob_filename=copy.deepcopy(filename)
    key_indices = split_names(keyed_filename, keyword_identifiers)
    
    for key in key_indices.keys():
        glob_filename=glob_filename.replace('{'+key+'}', '*')
    return glob.glob(glob_filename)

def assign_meg_session(dframe):
    dframe=copy.deepcopy(dframe)
    subjids = dframe['meg_subjid'].unique()
    for subjid in subjids:
        idxs = dframe[dframe['meg_subjid']==subjid].index
        dframe.loc[idxs,'meg_session'] = np.arange(1,len(idxs)+1, dtype=int)
    return dframe

def _ctf_anonymize(meg_fname, outdir='./'):
    tmp_ = op.basename(meg_fname)
    pre_, suff_ = op.splitext(tmp_)
    outname = op.join(outdir, f'{pre_}_anon{suff_}')
    if shutil.which('newDs') is None:
        raise(SystemError('newDs not found - it does not appear that the CTF\
                          package is installed or cannot be found in this shell'))
    sub_cmd = f'newDs -anon {meg_fname} {outname}'
    subprocess.run(sub_cmd.split())
    
def _fif_anonymize(meg_fname, outdir='./'):
    tmp_ = op.basename(meg_fname)
    pre_, suff_ = op.splitext(tmp_)
    outname = op.join(outdir, f'{pre_}_anon{suff_}')   
    raw = mne.io.read_raw_fif(meg_fname)
    raw.anonymize()
    raw.save(outname)

def anonymize_meg(meg_fname):
    if op.splitext(meg_fname)[-1]=='.ds':
        _ctf_anonymize(meg_fname)
    elif op.splitext(meg_fname)[-1]=='.fif':
        _fif_anonymize(meg_fname)
        
def convert_brik(mri_fname):
    if op.splitext(mri_fname)[-1] not in ['.BRIK', '.HEAD']:
        raise(TypeError('Must be an afni BRIK or HEAD file to convert'))
    import shutil
    if shutil.which('3dAFNItoNIFTI') is None:
        raise(SystemError('It does not appear Afni is installed, cannot call\
                          3dAFNItoNIFTI'))
    basename = op.basename(mri_fname)
    dirname = op.dirname(mri_fname)
    outname = basename.split('+')[0]+'.nii'
    outname = op.join(dirname, outname)
    subcmd = f'3dAFNItoNIFTI {mri_fname} {outname}'
    subprocess.run(subcmd.split())
    print(f'Converted {mri_fname} to nifti')
    
def _clean_up_all():
    '''Remove all processing files in topdir.
    Prompts user for deletion of data'''
    files = os.listdir('.')
    cleanup_dirs = ['QA','setup_code', 'SUBJECTS_DIR']
    
    del_dirs = []
    for dirname in cleanup_dirs:
        if dirname in files:
            del_dirs.append(dirname)
    print('!!!!!    WARNING DATA LOSS    !!!!!')
    confirm = input(f'Do you want to delete these folders (y\\n):\n{del_dirs}\n')
    if confirm.lower() == 'y':
        for del_i in del_dirs:
            print(f'Removing {del_i}')
            shutil.rmtree(del_i)              



dframe=pd.read_csv('MasterList_final.csv', index_col='Unnamed: 0')



row = dframe.loc[2]
subjid=row['bids_subjid']
subjects_dir=subjects_dir
report_path=row['report_path']
meg_fname=row['full_meg_path']
trans=row['trans_fname']



anon_subjid = subjid+'_defaced'

if meg_fname is not None:
    raw = read_meg(meg_fname)

# Make Report    
rep = mne.Report()
null_info = mne.Info({'dig':raw.info['dig'], 'dev_ctf_t': raw.info['dev_ctf_t'], 
                      'dev_head_t': raw.info['dev_head_t'], 'bads':[],
                      'chs':[], 'ch_names':[], 'nchan':0})
tmp = mne.viz.plot_alignment(info=null_info, subject=subjid, subjects_dir=subjects_dir, 
                         trans=trans, dig=True)  
rep.add_figs_to_section(figs=tmp, captions='Fiducials',
                                section='Coregistration')        

tmp = mne.viz.plot_alignment(info=raw.info, subject=subjid, subjects_dir=subjects_dir, 
                         trans=trans, dig=True, coord_frame='meg')    
rep.add_figs_to_section(figs=tmp, captions='Head Alignment',
                                section='Coregistration')
tmp = mne.viz.plot_alignment(subject=anon_subjid)
rep.add_figs_to_section(figs=tmp, captions='Deface Confirmation',
                                section='Anonymization')    
rep.save(fname=report_path)

  
