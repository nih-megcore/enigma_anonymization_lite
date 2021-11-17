#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:11:28 2021

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
from multiprocessing import Pool

from mne_bids import write_anat, BIDSPath, write_raw_bids

#%%
#%%  Confirm softare version
assert mne_bids.__version__[0:3]>='0.8'
if shutil.which('recon-all') == None:
    raise('Freesurfer is not installed or on your path')


#%% Setup
n_jobs=12
line_freq = 60.0
global topdir
topdir = '/fast/enigma_meg_prep/TEST_DATA'

#%% Configure Working Paths
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
logger = logging.getLogger()
# fileHandle = logging.FileHandler(f'{log_dir}/process_logger.txt')
# logger.addHandler(fileHandle)

#Directory to save html files
QA_dir = f'{topdir}/QA'
if not os.path.exists(QA_dir): os.mkdir(QA_dir)

mri_staging_dir = f'{topdir}/mri_staging'

keyword_identifiers={'SUBJID': [],
                     'SUBJID_first': [], 
                     'SUBJID_last': [],
                     'DATE': [], 
                     'TASK': []}

code_topdir=f'{topdir}/setup_code'
brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
face_template=f'{code_topdir}/face.gca'

#%%
from .enigma_anonymization import download_deface_templates
download_deface_templates(code_topdir)

# meg_template = op.join(topdir, 'MEG/{SUBJID}_{TASK}_{DATE}_??.ds')
# mri_template = op.join(topdir, 'MRIS_ORIG/{SUBJID}/{SUBJID}.nii')

# mri_dframe, meg_dframe = return_mri_meg_dframes(mri_template, meg_template)
        
# meg_dframe['date'] = meg_dframe.full_meg_path.apply(_return_date, key_index=-2 )
# meg_dframe['task'] = meg_dframe.full_meg_path.apply(lambda x: x.split('_')[-3])
# not_rest_idxs = meg_dframe[~meg_dframe['task'].isin(['rest'])].index
# meg_dframe.drop(index=not_rest_idxs, inplace=True)

# meg_dframe.sort_values(['meg_subjid','date'])
# meg_dframe = assign_meg_session(meg_dframe)

# combined_dframe  = pd.merge(mri_dframe, meg_dframe, left_on='mri_subjid', 
#                             right_on='meg_subjid')
# combined_dframe.reset_index(drop=True, inplace=True)


# orig_subj_list = combined_dframe['meg_subjid'].unique()
# tmp = np.arange(len(orig_subj_list), dtype=int)
# np.random.shuffle(tmp)  #Change the index for subjids

# for rnd_idx, subjid in enumerate(orig_subj_list):
#     print(subjid)
#     print(str(rnd_idx))
#     subjid_idxs = combined_dframe[combined_dframe['meg_subjid']==subjid].index
#     combined_dframe.loc[subjid_idxs,'bids_subjid'] = "sub-{0:0=4d}".format(tmp[rnd_idx])

# bids_dir = './bids_out'
# if not os.path.exists(bids_dir): os.mkdir(bids_dir)

# if not 'combined_dframe' in locals().keys():
#     combined_dframe = pd.read_csv('MasterList.csv', index_col='Unnamed: 0')

# combined_dframe['fs_T1mgz'] = combined_dframe.bids_subjid.apply(lambda x: op.join(subjects_dir, x, 'mri', 'T1.mgz'))
# combined_dframe['report_path'] = combined_dframe.bids_subjid.apply(lambda x: op.join(QA_dir, x+'_report.html'))

# dframe=combined_dframe
# dframe['subjects_dir'] = subjects_dir
# assign_report_path(dframe, f'{topdir}/QA')



# #%% Process Data
# # =============================================================================
# # Make multiprocess calls looped over subjid
# # =============================================================================


# # Copy T1 from original location to staging area
# dframe['T1staged'] = ''
# if not os.path.exists(mri_staging_dir): os.mkdir(mri_staging_dir)
# for idx,row in dframe.iterrows(): #t1_fname in dframe['T1nii']:
#     _, ext = os.path.splitext(row['full_mri_path'])
#     out_fname = row['bids_subjid']+ext
#     out_path = op.join(mri_staging_dir, out_fname) 
#     shutil.copy(row['full_mri_path'], out_path)
#     dframe.loc[idx, 'T1staged'] = out_path
    
# dframe.to_csv('MasterList.csv')
# ## SETUP MAKE_SCALP_SURFACES
# inframe = dframe.loc[:,['T1staged','bids_subjid', 'subjects_dir']]
# #Remove duplicates over sessions
# inframe.drop_duplicates(subset=['T1staged','bids_subjid'], inplace=True)
# with Pool(processes=n_jobs) as pool:
#     pool.starmap(make_scalp_surfaces_anon,
#                      inframe.values)

# dframe['T1anon']=dframe['T1staged'].apply(
#     lambda x: op.splitext(x)[0]+'_defaced'+op.splitext(x)[1])

# ## Cleanup mri_deface log files
# anon_logs = glob.glob(op.join(os.getcwd(), '*defaced.log'))
# for i in anon_logs: shutil.move(i, op.join(mri_staging_dir,op.basename(i)))                      

# try: 
#     del inframe 
# except:
#     pass




# ## For NIH - create transformation matrix
# from nih2mne.calc_mnetrans import write_mne_fiducials 
# from nih2mne.calc_mnetrans import write_mne_trans

# if not os.path.exists(f'{topdir}/trans_mats'): os.mkdir(f'{topdir}/trans_mats')

# for idx, row in dframe.iterrows():
#     if op.splitext(row['full_mri_path'])[-1] == '.gz':
#         afni_fname=row['full_mri_path'].replace('.nii.gz','+orig.HEAD')
#     else:
#         afni_fname=row['full_mri_path'].replace('.nii','+orig.HEAD')
#     fid_path = op.join('./trans_mats', f'{row["bids_subjid"]}_{str(int(row["meg_session"]))}-fiducials.fif')
#     try:
#         write_mne_fiducials(subject=row['bids_subjid'],
#                             subjects_dir=subjects_dir, 
#                             # afni_fname=afni_fname,
#                             searchpath = os.path.dirname(afni_fname),
#                             output_fid_path=fid_path)
#     except:
#         print('error '+row['bids_subjid'])

#     try:              
#         trans_fname=op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
#         write_mne_trans(mne_fids_path=fid_path,
#                         dsname=row['full_meg_path'], 
#                         output_name=trans_fname, 
#                         subjects_dir=subjects_dir)
#         dframe.loc[idx,'trans_fname']=trans_fname
#     except:
#         print('error in trans calculation '+row['bids_subjid'])

# dframe.to_csv('MasterList_final.csv')                   
# # ## SEtup report    
# # inframe = dframe.loc[:,['bids_subjid','subjects_dir','report_path', 
# #                         'full_meg_path','trans_fname']]

# # ## SETUP Make Report
# # with Pool(processes=n_jobs) as pool:
# #     pool.starmap(make_QA_report, inframe.values)
  
# def loop_QA_reports(dframe, subjects_dir=None):
#     for idx, row in dframe.iterrows():
#         subjid=row['bids_subjid']
#         subjects_dir=subjects_dir
#         report_path=row['report_path']
#         meg_fname=row['full_meg_path']
#         trans=row['trans_fname']
#         subj_logger=get_subj_logger(subjid)
#         try:
#             subj_logger.info('Running QA report')
#             make_QA_report(subjid=subjid, 
#                            subjects_dir=subjects_dir, 
#                            report_path=report_path, 
#                            meg_fname=meg_fname, 
#                            trans=trans)
#             subj_logger.info('Finished QA report')
#         except BaseException as e:
#             subj_logger.error(f'make_QA_report: \n{e}')

# loop_QA_reports(dframe, subjects_dir=subjects_dir)

# # For dry run 
# def make_trans_name(row):
#     return op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')

# for idx,row in dframe.iterrows(): dframe.loc[idx,'trans_fname']=make_trans_name(row)



# #%%    
# # =============================================================================
# # STOP HERE PRIOR TO BIDS PROCESSING
# # =============================================================================

# #%% Process BIDS
# bids_dir = f'{topdir}/bids_out'
# if not os.path.exists(bids_dir): os.mkdir(bids_dir)

# combined_dframe = dframe
# # =============================================================================
# # Convert MEG
# # =============================================================================
    
# errors=[]
# for idx, row in combined_dframe.iterrows():
#     subj_logger = get_subj_logger(row.bids_subjid)
#     try:
#         print(idx)
#         print(row)
#         subject = row.meg_subjid
#         mri_fname = row.full_mri_path   #Will need to change to confirm anon
#         raw_fname = row.full_meg_path
#         output_path = bids_dir
        
#         raw = mne.io.read_raw_ctf(raw_fname)  #Change this - should be generic for meg vender
#         raw.info['line_freq'] = line_freq 
        
#         sub = row['bids_subjid'][4:] 
#         ses = '0'+str(int(row['meg_session']) )
#         task = 'rest'
#         run = '01'
#         bids_path = BIDSPath(subject=sub, session=ses, task=task,
#                              run=run, root=output_path, suffix='meg')
        
#         write_raw_bids(raw, bids_path)
#     except BaseException as e:
#         subj_logger.error('MEG BIDS PROCESSING:', e)
        

# #%% Create the bids from the anonymized MRI
# for idx, row in dframe.iterrows():
#     subj_logger = get_subj_logger(row.bids_subjid)
#     try:
#         sub=row['bids_subjid'][4:] 
#         ses='01'
#         output_path = f'{topdir}/bids_out'
        
#         raw = read_meg(row['full_meg_path'])          #FIX currently should be full_meg_path - need to verify anon
#         trans = mne.read_trans(row['trans_fname'])
#         t1_path = row['T1anon']
        
#         t1w_bids_path = \
#             BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
    
#         landmarks = mne_bids.get_anat_landmarks(
#             image=row['T1anon'],
#             info=raw.info,
#             trans=trans,
#             fs_subject=row['bids_subjid']+'_defaced',
#             fs_subjects_dir=subjects_dir
#             )
        
#         # Write regular
#         t1w_bids_path = write_anat(
#             image=row['T1anon'],
#             bids_path=t1w_bids_path,
#             landmarks=landmarks,
#             deface=False,  #Deface already done
#             overwrite=True
#             )
        
#         anat_dir = t1w_bids_path.directory   
#     except BaseException as e:
#         subj_logger.error('MRI BIDS PROCESSING', e)



        
if __name__=='__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-topdir', help='''The directory for the outputs''')
    parser.add_argument('-search_meg', help='''Interactively search for MEG 
                        datasets using a prompted template path.  After 
                        confirmation a dataframe will be written out in the 
                        form of a CSV file that can be QA-ed for errors.''') 
    parser.add_argument('-search_mri', help='''Interactively search for MRI 
                        datasets using a prompted template path.  After 
                        confirmation a dataframe will be written out in the 
                        form of a CSV file that can be QA-ed for errors.''')                        
                        
                        
    # parser.add_argument('-subjid', help='''Define subjects id (folder name)
    #                     in the SUBJECTS_DIR''')
    # parser.add_argument('-recon_check', help='''Process all anatomical steps that
    #                     have not been completed already.  This will check the major
    #                     outputs from autorecon1, 2, 3, and mne source setup and
    #                     proceed with the processing. The default is set to TRUE''')
    # parser.add_argument('-recon1', help='''Force recon1 to be processed''', action='store_true')
    # parser.add_argument('-recon2', help='''Force recon2 to be processed''', action='store_true')
    # parser.add_argument('-recon3', help='''Force recon3 to be processed''', action='store_true')
    # parser.add_argument('-setup_source', help='''Runs the setup source space processing
    #                     in mne python to create the BEM model''', action='store_true')
    # parser.add_argument('-run_unprocessed', help='''Checks for all unrun processes and
    #                     runs any additional steps for inputs to the source model''', action='store_true')

    parser.add_argument('-interactive', help='''Set to True by default. Set to 
                        False to batch process''', default=True)
    parser.description='''Processing for the anatomical inputs of the enigma pipeline'''
    args = parser.parse_args()
    if not args.subjid: raise ValueError('Subject ID must be set')
    if not args.subjects_dir: 
        args.subjects_dir=os.environ['SUBJECTS_DIR']
    else:
        os.environ['SUBJECTS_DIR']=args.subjects_dir  
    


        
