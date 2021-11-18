#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 09:38:21 2021

@author: stoutjd
"""

#%%
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
from multiprocessing import Pool

from mne_bids import write_anat, BIDSPath, write_raw_bids


logger = logging.getLogger()

#%% Utility functions
def download_deface_templates(code_topdir):
    '''Download and unzip the templates for freesurfer defacing algorithm'''
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
    try:
        assert os.path.exists(f'{code_topdir}/face.gca')  
        logger.debug('Face template present')
    except:
        logger.error('Face template does not exist')
    try:
        assert os.path.exists(f'{code_topdir}/talairach_mixed_with_skull.gca')
        logger.debug('Talairach mixed template present')
    except:
        logger.error('Talairach mixed template does not exist')

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
        
def _dframe_from_template(template, keyword_identifiers, datatype=None, 
                          task_filter=['rest']):
    '''Main processing for dataframe creation.  Called as a subfunction to 
    higher level calls'''
    key_indices = split_names(template, keyword_identifiers)
    subjid_from_dir=_proc_steps(template)
    for key in key_indices.keys():
        template = template.replace('{'+key+'}', '*')
    _inputs = glob.glob(template)
    _dframe = pd.DataFrame(_inputs, columns=['full_path'])
    
    if subjid_from_dir:
        _dframe[f'{datatype}_subjid']=\
            _dframe.full_path.apply(dir_at_pos, 
                                           position=key_indices['SUBJID'][-1])
    else:
        _dframe[f'{datatype}_subjid']=\
            _dframe.full_path.apply(subjid_from_filename)
    
    _dframe.rename(columns={'full_path':f'full_{datatype}_path'}, inplace=True)
    #Special processing for MEG datasets
    if datatype.lower()=='meg':
        _dframe['date'] = _dframe.full_meg_path.apply(_return_date, key_index=-2 )
        _dframe['task'] = _dframe.full_meg_path.apply(lambda x: x.split('_')[-3])
        #Check to see if the datasets are in a list of possible task labels
        not_rest_idxs = _dframe[~_dframe['task'].isin(task_filter)].index             
        _dframe.drop(index=not_rest_idxs, inplace=True)
        
        _dframe.sort_values(['meg_subjid','date'])
        _dframe = assign_meg_session(_dframe)
    return _dframe
    
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
    



#%%  Make headsurfaces to confirm alignment and defacing
# Setup and run freesurfer commands 

def make_scalp_surfaces_anon(mri=None, subjid=None, subjects_dir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    anon_subjid = subjid+'_defaced'
    prefix, ext = os.path.splitext(mri)
    anon_mri = prefix+'_defaced'+ext 
    subj_logger=get_subj_logger(subjid)
    subj_logger.info(f'Original:{mri}',f'Processed:{anon_mri}')
    
    # Set subjects dir path for subcommand call
    os.environ['SUBJECTS_DIR']=subjects_dir
    
    try: 
        subprocess.run(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}'.split(),
                       check=True)
    except BaseException as e:
        subj_logger.error('MRI_DEFACE')
        subj_logger.error(e)
    
    try:
        subprocess.run(f'recon-all -i {mri} -s {subjid}'.split(),
                       check=True)
        subprocess.run(f'recon-all -autorecon1 -noskullstrip -s {subjid}'.split(),
                       check=True)
        subj_logger.info('RECON_ALL IMPORT FINISHED')
    except BaseException as e:
        subj_logger.error('RECON_ALL IMPORT')
        subj_logger.error(e)
    
    try:
        subprocess.run(f"mkheadsurf -s {subjid}".split(), check=True)
        subj_logger.info('MKHEADSURF FINISHED')
    except:
        try:
            proc_cmd = f"mkheadsurf -i {op.join(subjects_dir, subjid, 'mri', 'T1.mgz')} \
                -o {op.join(subjects_dir, subjid, 'mri', 'seghead.mgz')} \
                -surf {op.join(subjects_dir, subjid, 'surf', 'lh.seghead')}"
            subprocess.run(proc_cmd.split(), check=True)
        except BaseException as e:
            subj_logger.error('MKHEADSURF')
            subj_logger.error(e)

    
    # =========================================================================
    #     Repeat for defaced
    # =========================================================================

    try:
        subprocess.run(f'recon-all -i {anon_mri} -s {subjid}_defaced'.split(),
                       check=True)
        subprocess.run(f'recon-all -autorecon1 -noskullstrip -s {subjid}_defaced'.split(),
                       check=True)
        subj_logger.info('RECON_ALL IMPORT (ANON) FINISHED')
    except BaseException as e:
        subj_logger.error('RECON_ALL IMPORT (ANON)')
        subj_logger.error(e)
    
    try:
        subprocess.run(f"mkheadsurf -s {subjid}_defaced".split(), check=True)
        subj_logger.info('MKHEADSURF (ANON) FINISHED')
    except:
        try:
            proc_cmd = f"mkheadsurf -i {op.join(subjects_dir, anon_subjid, 'mri', 'T1.mgz')} \
                -o {op.join(subjects_dir, anon_subjid, 'mri', 'seghead.mgz')} \
                -surf {op.join(subjects_dir, anon_subjid, 'surf', 'lh.seghead')}"
            subprocess.run(proc_cmd.split(), check=True)
        except BaseException as e:
            subj_logger.error('MKHEADSURF')
            subj_logger.error(e)    
        
    # except BaseException as e:
    #     subj_logger.error('MKHEADSURF (ANON)')
    #     subj_logger.error(e)        
        
    try:
        # Cleanup
        link_surf(subjid, subjects_dir=subjects_dir)
        link_surf(anon_subjid, subjects_dir=subjects_dir)      
    except:
        pass        
        
def make_QA_report(subjid=None, subjects_dir=None, 
                 report_path=None, meg_fname=None, trans=None):
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


    
def validate_paths(in_dframe):
    for i in in_dframe.values:
        for j in i:
            if os.ispath(j):
                if not os.path.exists(j):
                    print(f'Does not exists: {j}')
        
def link_surf(subjid=None, subjects_dir=None):
    '''Link the surfaces to the BEM dir
    Looks for lh.seghead and links to bem/outer_skin.surf
    Tests for broken links and corrects'''
    s_bem_dir = f'{subjects_dir}/{subjid}/bem'
    src_file = f'{subjects_dir}/{subjid}/surf/lh.seghead'
    link_path = f'{s_bem_dir}/outer_skin.surf'
    if not os.path.exists(s_bem_dir):
        os.mkdir(s_bem_dir)
    if not os.path.exists(link_path):
        try:
            os.symlink(src_file, link_path)
        except:
            pass  #If broken symlink - this is fixed below
    #Test and fix broken symlink
    if not os.path.exists(os.readlink(link_path)):
        logger.info(f'Fixed broken link: {link_path}')
        os.unlink(link_path)
        os.symlink(src_file, link_path)
           
def get_subj_logger(subjid):
    '''Return the subject specific logger.
    This is particularly useful in the multiprocessing where logging is not
    necessarily in order'''
    log_dir = f'{topdir}/logs'
    logger = logging.getLogger(subjid)
    if logger.handlers != []:
        # Check to make sure that more than one file handler is not added
        tmp_ = [type(i) for i in logger.handlers ]
        if logging.FileHandler in tmp_:
            return logger
    fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_log.txt')
    fmt = logging.Formatter(fmt=f'%(asctime)s - %(levelname)s - {subjid} - %(message)s')
    fileHandle.setFormatter(fmt=fmt) 
    logger.addHandler(fileHandle)
    return logger

# def test_make_QA_report():
#     row = dframe.loc[2]
#     subjid=row['bids_subjid']
#     subjects_dir=subjects_dir
#     report_path=row['report_path']
#     meg_fname=row['full_meg_path']
#     trans=row['trans_fname']
#     make_QA_report(subjid=subjid, 
#                    subjects_dir=subjects_dir, 
#                    report_path=report_path, 
#                    meg_fname=meg_fname, 
#                    trans=trans)
    