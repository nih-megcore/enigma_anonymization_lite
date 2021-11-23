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

import wget 
import gzip

import shutil
import matplotlib
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 
from multiprocessing import Pool

from mne_bids import write_anat, BIDSPath, write_raw_bids
from enigma_preupload.enigma_anonymization import download_deface_templates
from enigma_preupload.enigma_anonymization import _dframe_from_template
from enigma_preupload.enigma_anonymization import assign_report_path
from enigma_preupload.enigma_anonymization import make_scalp_surfaces_anon
from enigma_preupload.enigma_anonymization import assign_mri_staging_path, get_subj_logger
import enigma_preupload
from enigma_preupload.enigma_anonymization import make_QA_report 
from enigma_preupload.enigma_anonymization import read_meg

#%%  Confirm softare version
assert mne_bids.__version__[0:3]>='0.8'
if shutil.which('recon-all') == None:
    raise('Freesurfer is not installed or on your path')


#%% Setup
n_jobs=12
line_freq = 60.0

def assess_config():
    '''Returns the topdir entry from the config file located at the root level
    of the package'''
    config_path = op.join(op.dirname(enigma_preupload.__path__[0]), 'config.txt')
    if op.exists(config_path):
        with open(config_path) as fid:
            config = fid.readlines()
        conf_entry = [i for i in config if 'topdir=' in i]
        global topdir
        if len(conf_entry)==1:
            topdir=conf_entry[0].replace('topdir=','').replace('\n','')
        if len(conf_entry)>1:
            raise ValueError(f'More than one entry found for topdir \
                             in {config_path}')
        if len(conf_entry)==0:
            topdir=input(f'Enter the output directory:\n\
                         This might be easier to put in {config_path} as topdir=...\n'.replace('    ',''))
        print(topdir)
    else:
        topdir=input(f'Enter the output directory:\n\
                         This might be easier to put in {config_path} as topdir=...\n'.replace('    ',''))
    return topdir        
            
def initialize(topdir=None, config=None):
    '''Assign several global variables, assess topdir if not assigned from
    the commandline, and create default directories for processing.'''
    
    if topdir==None:
        if config==None:
            topdir=assess_config()
        # else:
            # topdir=assess_config(config)
    # assert os.path.isdir(topdir)
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
    
    global mri_staging_dir
    mri_staging_dir = f'{topdir}/mri_staging'
    
    global keyword_identifiers
    keyword_identifiers={'SUBJID': [],
                         'SUBJID_first': [], 
                         'SUBJID_last': [],
                         'DATE': [], 
                         'TASK': []}
    
    code_topdir=f'{topdir}/setup_code'
    global brain_template
    brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
    global face_template
    face_template=f'{code_topdir}/face.gca'
    
    download_deface_templates(code_topdir)


#%%
def datatype_from_template(topdir=None, interactive=True, template=None, 
                      datatype=None):
    '''Generate a CSV of MEG datasets found by searching through a defined
    template prompted at runtime'''
    
    prompt_val='''
    Provide a template to search for MEG datasets using specific keywords:
    
    subject - SUBJID, SUBJID_first, SUBJID_last
    date - DATE
    task - TASK  (currently only supporting resting datasets)
    
    Examples:
        /home/myusername/data/MEG/{SUBJID}_{TASK}_{DATE}_??.ds
        
        /data/MEG/{SUBJID}/Run??_{TASK}_{DATE}_raw.fif
    '''
    rel_out_fname = f'{datatype}_dframe.csv'
    logger=logging.getLogger()
    if interactive==True:
        save_output='n'
        while save_output.lower() not in ['y','q', 'yes', 'quit']:
            template = input(prompt_val)
            dframe = _dframe_from_template(template, keyword_identifiers, 
                                   datatype=datatype.lower())
            print(dframe.head())
            save_output=input('Does this look correct?(y)es, (n)o, (q)uit')
        if save_output.lower() in ['y','yes']:
            if topdir!=None:
                out_fname = op.join(topdir, rel_out_fname)
            else:
                out_fname = rel_out_fname
            dframe.to_csv(out_fname, index=False)
            logger.info(f'Saved {datatype} dataframe: {out_fname}')
    if interactive!=True:
        dframe = _dframe_from_template(template, keyword_identifiers, 
                               datatype=datatype.lower()) 
        if topdir!=None:
            out_fname = op.join(topdir, rel_out_fname)
        else:
            out_fname = rel_out_fname
        dframe.to_csv(out_fname, index=False)
        logger.info(f'Saved {datatype} dataframe: {out_fname}')
                   
def merge_dframes(topdir=None, meg_csv_path=None, mri_csv_path=None):
    '''
    Merge the dataframes by subjid and compile a list of the mismatches.
    Saves the merged dataframe as a CSV file in the top directory as
    combined_dframe.csv

    Parameters
    ----------
    topdir : path, optional
        Top processing directory. The default is None.
    meg_csv_path : path, optional
        CSV file generated by data search. The default is None.  If no specified
        path is provided, the default location of meg_dframe.csv will be used.
    mri_cvs_path : path, optional
        CSV file generated by data search. The default is None.  If no specified
        path is provided, the default location of mri_dframe.csv will be used.

    Returns
    -------
    None.

    '''
    if (topdir==None) and (meg_csv_path==None):
        raise ValueError('topdir OR meg_csv_path must be defined')
    if (topdir==None) and (mri_csv_path==None):
        raise ValueError('topdir OR mri_csv_path must be defined')

    if (meg_csv_path==None):
        meg_csv_path = op.join(topdir, 'meg_dframe.csv')
    if (mri_csv_path==None):
        mri_csv_path = op.join(topdir, 'mri_dframe.csv')
    meg_dframe = pd.read_csv(meg_csv_path)
    mri_dframe = pd.read_csv(mri_csv_path)
    combined_dframe  = pd.merge(mri_dframe, meg_dframe, left_on='mri_subjid', 
                                right_on='meg_subjid')
    combined_dframe.reset_index(drop=True, inplace=True)
    outfname = op.join(topdir, 'combined_dframe.csv')
    logger.info(f'There are {len(combined_dframe)} subjects')
    
    combined_dframe.to_csv(outfname, index=False)

def finalize_masterlist(topdir=None):
    combined_csv_path = op.join(topdir, 'combined_dframe.csv')
    combined_dframe = pd.read_csv(combined_csv_path)    
    orig_subj_list = combined_dframe['meg_subjid'].unique()
    tmp = np.arange(len(orig_subj_list), dtype=int)
    np.random.shuffle(tmp)  #Change the index for subjids
    for rnd_idx, subjid in enumerate(orig_subj_list):
        print(subjid)
        print(str(rnd_idx))
        subjid_idxs = combined_dframe[combined_dframe['meg_subjid']==subjid].index
        combined_dframe.loc[subjid_idxs,'bids_subjid'] = "sub-{0:0=4d}".format(tmp[rnd_idx])
    combined_dframe['fs_T1mgz'] = combined_dframe.bids_subjid.apply(lambda x: op.join(subjects_dir, x, 'mri', 'T1.mgz'))
    combined_dframe['report_path'] = combined_dframe.bids_subjid.apply(lambda x: op.join(QA_dir, x+'_report.html'))
    
    assign_report_path(combined_dframe, f'{topdir}/QA')
    assign_mri_staging_path(combined_dframe, f'{topdir}/mri_staging')
    combined_dframe['T1anon']=combined_dframe['T1staged'].str.replace('.nii','')+'_defaced.nii'
    combined_dframe['trans_fname'] = f'{topdir}/trans_mats/'\
        +combined_dframe['bids_subjid']+'_'\
            +combined_dframe['meg_session'].astype(int).astype(str)+'-trans.fif'
    combined_dframe['subjects_dir'] = subjects_dir
    outfname = op.join(topdir, 'MasterList.csv')
    combined_dframe.to_csv(outfname, index=False)

# def process_mri_bids(topdir=None):
#     dframe = pd.read_csv(op.join(topdir, 'MasterList.csv'))
#     bids_dir = op.join(topdir, 'bids_out')
#     if not os.path.exists(bids_dir): os.mkdir(bids_dir)    

# def process_meg_bids(topdir=None):
#     dframe = pd.read_csv(op.join(topdir, 'MasterList.csv'))
#     bids_dir = op.join(topdir, 'bids_out')
#     if not os.path.exists(bids_dir): os.mkdir(bids_dir)


def stage_mris(topdir=None):
    '''Copy T1 from original location to staging area'''
    dframe = pd.read_csv(op.join(topdir, 'MasterList.csv'))
    if not os.path.exists(mri_staging_dir): os.mkdir(mri_staging_dir)
    for idx,row in dframe.iterrows():
        in_fname = row['full_mri_path']
        out_fname = row['T1staged']
        shutil.copy(in_fname, out_fname) 

def parrallel_make_scalp_surfaces(topdir=None):
    dframe = pd.read_csv('MasterList.csv')
    ## SETUP MAKE_SCALP_SURFACES
    inframe = dframe.loc[:,['T1staged','bids_subjid', 'subjects_dir']]
    inframe['topdir']  = topdir
    #Remove duplicates over sessions
    inframe.drop_duplicates(subset=['T1staged','bids_subjid'], inplace=True)
    with Pool(processes=n_jobs) as pool:
        pool.starmap(make_scalp_surfaces_anon,
                          inframe.values)
    
    dframe['T1anon']=dframe['T1staged'].apply(
        lambda x: op.splitext(x)[0]+'_defaced'+op.splitext(x)[1])
    
    ## Cleanup mri_deface log files
    anon_logs = glob.glob(op.join(os.getcwd(), '*defaced.log'))
    for i in anon_logs: shutil.move(i, op.join(mri_staging_dir,op.basename(i)))                      
    
    try: 
        del inframe 
    except:
        pass

def process_nih_transforms(topdir=None):
    csv_fname = op.join(topdir, 'MasterList.csv')
    dframe=pd.read_csv(csv_fname)
    from nih2mne.calc_mnetrans import write_mne_fiducials 
    from nih2mne.calc_mnetrans import write_mne_trans
    
    if not os.path.exists(f'{topdir}/trans_mats'): os.mkdir(f'{topdir}/trans_mats')
    
    for idx, row in dframe.iterrows():
        subj_logger=get_subj_logger(row['bids_subjid'], log_dir=f'{topdir}/logs')
        if op.splitext(row['full_mri_path'])[-1] == '.gz':
            afni_fname=row['full_mri_path'].replace('.nii.gz','+orig.HEAD')
        else:
            afni_fname=row['full_mri_path'].replace('.nii','+orig.HEAD')
        fid_path = op.join('./trans_mats', f'{row["bids_subjid"]}_{str(int(row["meg_session"]))}-fiducials.fif')
        try:
            write_mne_fiducials(subject=row['bids_subjid'],
                                subjects_dir=subjects_dir, 
                                searchpath = os.path.dirname(afni_fname),
                                output_fid_path=fid_path)
        except BaseException as e:
            subj_logger.error('Error in write_mne_fiducials', e)
            continue  #No need to write trans if fiducials can't be written
        try:              
            trans_fname=op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
            write_mne_trans(mne_fids_path=fid_path,
                            dsname=row['full_meg_path'], 
                            output_name=trans_fname, 
                            subjects_dir=subjects_dir)
            dframe.loc[idx,'trans_fname']=trans_fname
        except BaseException as e:
            subj_logger.error('Error in write_mne_trans', e)
            print('error in trans calculation '+row['bids_subjid'])
    dframe.to_csv('MasterList_final.csv', index=False)                   

def loop_QA_reports(dframe, subjects_dir=None, topdir=None):
    for idx, row in dframe.iterrows():
        subjid=row['bids_subjid']
        subjects_dir=subjects_dir
        report_path=row['report_path']
        meg_fname=row['full_meg_path']
        trans=row['trans_fname']
        subj_logger=get_subj_logger(subjid, log_dir=f'{topdir}/logs')
        try:
            subj_logger.info('Running QA report')
            make_QA_report(subjid=subjid, 
                            subjects_dir=subjects_dir, 
                            report_path=report_path, 
                            meg_fname=meg_fname, 
                            trans=trans)
            subj_logger.info('Finished QA report')
        except BaseException as e:
            subj_logger.error(f'make_QA_report: \n{e}')

# # =============================================================================
# # Convert MEG
# # =============================================================================

def process_meg_bids(dframe=None, topdir=None):
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    for idx, row in dframe.iterrows():
        subj_logger = get_subj_logger(row.bids_subjid, log_dir=f'{topdir}/logs')
        try:
            print(idx)
            print(row)
            subject = row.meg_subjid
            mri_fname = row.full_mri_path   #Will need to change to confirm anon
            raw_fname = row.full_meg_path
            output_path = bids_dir
            
            raw = read_meg(raw_fname)  
            raw.info['line_freq'] = line_freq 
            
            sub = row['bids_subjid'][4:] 
            ses = '0'+str(int(row['meg_session']) )
            task = 'rest'
            run = '01'
            bids_path = BIDSPath(subject=sub, session=ses, task=task,
                                  run=run, root=output_path, suffix='meg')
            
            write_raw_bids(raw, bids_path)
        except BaseException as e:
            subj_logger.exception('MEG BIDS PROCESSING:', e)
        

#%% Create the bids from the anonymized MRI
def process_mri_bids(dframe=None, topdir=None):
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)

    for idx, row in dframe.iterrows():
        subj_logger = get_subj_logger(row.bids_subjid, log_dir=f'{topdir}/logs')
        try:
            sub=row['bids_subjid'][4:] 
            ses='01'
            output_path = f'{topdir}/bids_out'
            
            raw = read_meg(row['full_meg_path'])          #FIX currently should be full_meg_path - need to verify anon
            trans = mne.read_trans(row['trans_fname'])
            t1_path = row['T1anon']
            
            t1w_bids_path = \
                BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
        
            landmarks = mne_bids.get_anat_landmarks(
                image=row['T1anon'],
                info=raw.info,
                trans=trans,
                fs_subject=row['bids_subjid']+'_defaced',
                fs_subjects_dir=subjects_dir
                )
            
            # Write regular
            t1w_bids_path = write_anat(
                image=row['T1anon'],
                bids_path=t1w_bids_path,
                landmarks=landmarks,
                deface=False,  #Deface already done
                overwrite=True
                )
            
            anat_dir = t1w_bids_path.directory   
        except BaseException as e:
            subj_logger.exception('MRI BIDS PROCESSING', e)

        
if __name__=='__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-topdir', help='''The directory for the outputs''')
    parser.add_argument('-search_datatype', help='''Interactively search for MEG 
                        datasets using a prompted template path.  After 
                        confirmation a dataframe will be written out in the 
                        form of a CSV file that can be QA-ed for errors.''') 
    parser.add_argument('-merge_csv', help='''Merge the meg and mri csv files
                        for further processing.''', action='store_true')
    parser.add_argument('-process_QA_images', help='''---''', 
                        action='store_true')
    # parser.add_argument('-make_surfaces', help='''---''', 
    #                     action='store_true')
    parser.add_argument('-compute_transform', help=''' Currently setup to only
                        work for NIH data''', action='store_true')
    parser.add_argument('-process_meg_bids', action='store_true')
    parser.add_argument('-process_mri_bids', action='store_true')
                        
                        
                        
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
    # if not args.subjid: raise ValueError('Subject ID must be set')
    # if not args.subjects_dir: 
    #     args.subjects_dir=os.environ['SUBJECTS_DIR']
    # else:
    #     os.environ['SUBJECTS_DIR']=args.subjects_dir  
    if args.topdir:
        initialize(topdir=args.topdir)
    else:
        initialize()
    if args.search_datatype:
        datatype_from_template(topdir=args.topdir, 
                               interactive=args.interactive,
                               datatype=args.search_datatype)
    if args.merge_csv:
        merge_dframes(topdir=topdir)
        finalize_masterlist(topdir=topdir)
    
    if args.compute_transform:
        process_nih_transforms(topdir)
        
    if args.process_QA_images:
        stage_mris(topdir)
        parrallel_make_scalp_surfaces(topdir)
        dframe = pd.read_csv(f'{topdir}/MasterList.csv')
        loop_QA_reports(dframe, subjects_dir=subjects_dir, topdir=topdir)
        
        
        


        
