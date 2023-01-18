#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 09:40:08 2022

@author: nugenta
"""
# This function 

import os
import logging
import os.path as op
import wget
import shutil
import gzip
import pandas as pd
import subprocess
from multiprocess import Pool
import mne
from mne_bids import write_anat, BIDSPath, write_raw_bids, get_anat_landmarks
 
#%% Utility functions

# This one downloads the files from freesurfer that are needed for the defacing algorithm
# They will be placed in the staging directory

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
        
# This function initializes the directory structure
           
def initialize(topdir=None):
    '''Assign several global variables, assess topdir if not assigned from
    the commandline, and create default directories for processing.'''
    
    os.chdir(topdir)
    
    # Create freesurfer folder
    global subjects_dir
    subjects_dir = f'{topdir}/SUBJECTS_DIR'
    if not os.path.exists(subjects_dir): os.mkdir(subjects_dir)
    os.environ['SUBJECTS_DIR'] = subjects_dir
    
    # Create log directory
    global log_dir
    log_dir = f'{topdir}/logs'
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    
    logging.basicConfig(filename=f'{log_dir}/process_logger.txt',
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        level=logging.INFO)
    global logger
    logger = logging.getLogger()
    fileHandle = logging.FileHandler(f'{log_dir}/process_logger.txt')
    logger.addHandler(fileHandle)
    
    #Directory to save html files
    global QA_dir
    QA_dir = f'{topdir}/QA'
    if not os.path.exists(QA_dir): os.mkdir(QA_dir)
    
    global staging_dir
    staging_dir = f'{topdir}/staging_dir'
    
    global keyword_identifiers
    keyword_identifiers={'SUBJID': [],
                         'SUBJID_first': [], 
                         'SUBJID_last': [],
                         'DATE': [], 
                         'TASK': []}
    
    code_topdir=f'{topdir}/staging_dir'
    global brain_template
    brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
    global face_template
    face_template=f'{code_topdir}/face.gca'
    
    download_deface_templates(code_topdir)
    
# this function copies the MRIs into the staging directory and creates a dataframe with the MRI info

def stage_mris(topdir, dframe):
    '''Copy T1 from original location to staging area'''
    staging_dir = op.join(topdir, 'staging_dir')
    if not os.path.exists(staging_dir): os.mkdir(staging_dir)
    mri_frame = pd.DataFrame(columns=['bids_subjid','staged_mri'])
    for idx,row in dframe.iterrows():
        in_fname = row['full_mri_path']
        bidsid = row['bids_subjid'] 
        fname = '%s_anat.nii' % (bidsid)
        out_fname = op.join(staging_dir,fname)
        shutil.copy(in_fname, out_fname) 
        mri_frame_temp = pd.DataFrame([[bidsid, out_fname]],
                        columns=['bids_subjid','staged_mri'])
        mri_frame=pd.concat([mri_frame,mri_frame_temp])
    return mri_frame

# This function initializes a logger for each subject

def get_subj_logger(subjid, log_dir=None):
    '''Return the subject specific logger.
    This is particularly useful in the multiprocessing where logging is not
    necessarily in order'''
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

# Creates a symbolic link for the surface files in the bem directory

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

# This function uses freesurfer to deface the MRI files, then does basic freesurfer processing
# on the subject to create the head surfaces required for QA 
# note that this function is not called directly, it's called by a function that enables
# parallel processing of all subjects

def make_scalp_surfaces_anon(mri=None, subjid=None, subjects_dir=None, 
                             topdir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    anon_subjid = subjid+'_defaced'
    prefix, ext = os.path.splitext(mri)
    anon_mri = prefix+'_defaced'+ext 
    log_dir=f'{topdir}/logs'
    subj_logger=get_subj_logger(subjid, log_dir=log_dir)
    subj_logger.info(f'Original:{mri}')
    subj_logger.info(f'Processed:{anon_mri}')
    
    brain_template=f'{topdir}/setup_code/talairach_mixed_with_skull.gca'
    face_template=f'{topdir}/setup_code/face.gca'
    
    # Set subjects dir path for subcommand call
    os.environ['SUBJECTS_DIR']=subjects_dir
    
    try: 
        subprocess.run(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}'.split(),
                       check=True)
    except BaseException as e:
        subj_logger.error('MRI_DEFACE')
        subj_logger.error(e)
    
    # only do the freesurfer processing for the defaced MRI

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
            subj_logger.error('MKHEADSURF (ANON)')
            subj_logger.error(e)    
                
    try:
        link_surf(anon_subjid, subjects_dir=subjects_dir)      
    except BaseException as e:
        subj_logger.error('Link error on BEM')
        subj_logger.error(e)

def parallel_make_scalp_surfaces(dframe, topdir=None, subjdir=None, njobs=1):
    ## SETUP MAKE_SCALP_SURFACES
    inframe = dframe.loc[:,['staged_mri','bids_subjid']]
    inframe['subjects_dir'] = subjects_dir
    inframe['topdir']  = topdir
    #Remove duplicates over sessions
    inframe.drop_duplicates(subset=['staged_mri','bids_subjid'], inplace=True)
    with Pool(processes=njobs) as pool:
        pool.starmap(make_scalp_surfaces_anon,
                          inframe.values)
        try: 
            del inframe 
        except:
            pass

# function to read the MEG scans into mne python for processing

def read_meg(meg_fname):
    if meg_fname[-1]==os.path.sep:
        meg_fname=meg_fname[:-1]
    _, ext = os.path.splitext(meg_fname)
    if ext == '.fif':
        return mne.io.read_raw_fif(meg_fname)
    if ext == '.ds':
        return mne.io.read_raw_ctf(meg_fname)
        
def process_mri_bids(dframe=None, topdir=None):
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)

    for idx, row in dframe.iterrows():
            subj_logger = get_subj_logger(row.bids_subjid, log_dir=f'{topdir}/logs')
            try:
                sub=row['bids_subjid']
                ses=str(row['session'])
                output_path = f'{topdir}/bids_out'
            
                raw = read_meg(row['full_meg_path'])          #FIX currently should be full_meg_path - need to verify anon
                trans = mne.read_trans(row['trans_fname'])
                t1_fname = '%s_anat_defaced.nii' % sub
                t1_path = op.join(topdir,'mri_staging_dir',t1_fname)
            
                t1w_bids_path = \
                    BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
                landmarks = get_anat_landmarks(
                    image=t1_path,
                    info=raw.info,
                    trans=trans,
                    fs_subject=row['bids_subjid']+'_defaced',
                    fs_subjects_dir=subjects_dir
                    )
                        # Write regular
                t1w_bids_path = write_anat(
                    image=t1_path,
                    bids_path=t1w_bids_path,
                    landmarks=landmarks,
                    deface=False,  #Deface already done
                    overwrite=True
                    )
            #anat_dir = t1w_bids_path.directory   
            except BaseException as e:
                subj_logger.exception('MRI BIDS PROCESSING', e)
            
def process_meg_bids(dframe=None, topdir=None, linefreq=60):
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    if set(['empty_room']).issubset(dframe.columns):
        eroom=True
    else:
        eroom=False
        
    for idx, row in dframe.iterrows():
        
        subj_logger = get_subj_logger(row.bids_subjid, log_dir=f'{topdir}/logs')
        if eroom==False:
            try:
                sub = row['bids_subjid']
                raw_fname = row['full_meg_path']
                output_path = bids_dir            
                raw = read_meg(raw_fname)  
                raw.info['line_freq'] = linefreq 
                ses=str(row['session'])
                task = 'rest'
                run = '01'
            
                bids_path = BIDSPath(subject=sub, session=ses, task=task,
                                  run=run, root=output_path, suffix='meg')
            
                write_raw_bids(raw, bids_path, anonymize={'daysback':40000, 'keep_his':False, 'keep_source':False},
                          overwrite=True)
            
            except BaseException as e:
                subj_logger.exception('MEG BIDS PROCESSING:', e)
        else:
           try:
               sub = row['bids_subjid']
               raw_fname = row['full_meg_path']
               eroom_fname = row['empty_room']
               output_path = bids_dir            
               raw = read_meg(raw_fname)  
               eroom = read_meg(eroom_fname)
               raw.info['line_freq'] = linefreq 
               ses=str(row['session'])
               task = 'rest'
               run = '01'
                
               bids_path = BIDSPath(subject=sub, session=ses, task=task,
                                      run=run, root=output_path, suffix='meg')
                
               write_raw_bids(raw, bids_path, anonymize={'daysback':40000, 'keep_his':False, 'keep_source':False},
                              empty_room=eroom, overwrite=True)
                
           except BaseException as e:
                    subj_logger.exception('MEG BIDS PROCESSING:', e)   
            
def loop_QA_report(dframe, subjects_dir=None, topdir=None):
    report_path=op.join(topdir,'QA/QAreport.html')
    rep = mne.Report()
    for idx, row in dframe.iterrows():
        subjid=row['bids_subjid']
        anon_subjid = subjid+'_defaced'    
        meg_fname=row['full_meg_path']
        raw = read_meg(meg_fname)
        subjects_dir=subjects_dir               
        trans=row['trans_fname']
        title_text = 'Check coreg and deface for subjid %s' % anon_subjid
        subj_logger=get_subj_logger(subjid, log_dir=f'{topdir}/logs')
        try:
            subj_logger.info('Running QA report')
            rep.add_trans(trans=trans, info=raw.info, subject=anon_subjid,
                    subjects_dir=subjects_dir, alpha=1.0, title=title_text) 
        except BaseException as e:
            subj_logger.error(f'make_QA_report: \n{e}')

    rep.save(fname=report_path, overwrite=True)


