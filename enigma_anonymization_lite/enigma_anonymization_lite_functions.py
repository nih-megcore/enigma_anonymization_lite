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
import matplotlib.pyplot as plt
import mne_bids
from mne_bids import write_anat, BIDSPath, write_raw_bids, get_anat_landmarks 
from random import randint

from enigma_anonymization_lite.utils import _sidecar_json_patched
mne_bids.write._sidecar_json = _sidecar_json_patched
 
# %% Utility functions

# This one downloads the files from freesurfer that are needed for the defacing algorithm
# They will be placed in the staging directory

def download_deface_templates(code_topdir):
    '''Download and unzip the templates for freesurfer defacing algorithm'''
    logger = logging.getLogger('process_logger')
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
    
    # Create BIDS directory and derivatives folder
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    derivs_dir = f'{topdir}/bids_out/derivatives'
    if not os.path.exists(derivs_dir): os.mkdir(derivs_dir)
    
    # Create freesurfer folder
    global subjects_dir
    freesurfer_dir = f'{topdir}/bids_out/derivatives/freesurfer'
    if not os.path.exists(freesurfer_dir): os.mkdir(freesurfer_dir)
    subjects_dir = f'{topdir}/bids_out/derivatives/freesurfer/subjects'
    if not os.path.exists(subjects_dir): os.mkdir(subjects_dir)
    os.environ['SUBJECTS_DIR'] = subjects_dir
    
    # Create log directory
    global log_dir
    log_dir = f'{topdir}/logs'
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        level=logging.INFO)
    
    #global logger
    
    fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
    logger = logging.getLogger('process_logger')
    fileHandle = logging.FileHandler(f'{log_dir}/process_logger.txt')
    fileHandle.setLevel(logging.INFO)
    fileHandle.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fileHandle)    
    logger.info('Initializing global enigma_anonymization log')
    
    #Directory to save html files
    global QA_dir
    QA_dir = f'{topdir}/bids_out/derivatives/BIDS_ANON_QA'
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
    mri_frame = pd.DataFrame(columns=['subjid','staged_mri'])
    for idx,row in dframe.iterrows():
        subjid = row['subjid']
        bidsid = 'sub-'+subjid        
        subj_logger = logging.getLogger(bidsid)      
        in_fname = row['full_mri_path']
        if in_fname.split('.')[-1]=='gz':
            fname = '%s_anat.nii.gz' % (bidsid)
        elif in_fname.split('.')[-1]=='nii':
            fname = '%s_anat.nii' % (bidsid)
        else:
            subj_logger.error('MRI must be .nii or .nii.gz')
        out_fname = op.join(staging_dir,fname)
        shutil.copy(in_fname, out_fname) 
        mri_frame_temp = pd.DataFrame([[subjid, out_fname]],
                        columns=['subjid','staged_mri'])
        mri_frame=pd.concat([mri_frame,mri_frame_temp])
        
        subj_logger.info('MRI staged and ready for processing')
    logger = logging.getLogger('process_logger')
    logger.info('MRIs staged and ready for processing')
    return mri_frame

# This function initializes a logger for each subject

def get_subj_logger(subjid, log_dir=None):
    '''Return the subject specific logger.
    This is particularly useful in the multiprocessing where logging is not
    necessarily in order'''
    fmt = '%(asctime)s :: %(levelname)s :: %(message)s'
    subj_logger = logging.getLogger(subjid)
    if subj_logger.handlers != []:
        # Check to make sure that more than one file handler is not added
        tmp_ = [type(i) for i in subj_logger.handlers ]
        if logging.FileHandler in tmp_:
            return subj_logger
    else:
        fileHandle = logging.FileHandler(f'{log_dir}/{subjid}_log.txt')
        fileHandle.setLevel(logging.INFO)
        fileHandle.setFormatter(logging.Formatter(fmt)) 
        subj_logger.addHandler(fileHandle)
        subj_logger.info('Initializing subject level enigma_anonymization log')
    return subj_logger

# Creates a symbolic link for the surface files in the bem directory

def link_surf(subjid=None, subjects_dir=None):
    '''Link the surfaces to the BEM dir
    Looks for lh.seghead and links to bem/outer_skin.surf
    Tests for broken links and corrects'''
    subjid = 'sub-' + subjid
    s_bem_dir = f'{subjects_dir}/{subjid}/bem'
    src_file = f'{subjects_dir}/{subjid}/surf/lh.seghead'
    link_path = f'{s_bem_dir}/outer_skin.surf'
    subj_logger = get_subj_logger(subjid)
    if not os.path.exists(s_bem_dir):
        os.mkdir(s_bem_dir)
        subj_logger.info('creating bem directory')
    if not os.path.exists(link_path):
        try:
            os.symlink(src_file, link_path)
        except:
            pass  #If broken symlink - this is fixed below
    #Test and fix broken symlink
    if not os.path.exists(os.readlink(link_path)):
        subj_logger.info(f'Fixed broken link: {link_path}')
        os.unlink(link_path)
        os.symlink(src_file, link_path)
    subj_logger.info('surface file linked to bem directory')
    
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
    anon_subjid = 'sub-'+subjid+'_defaced'
    bids_subjid = 'sub-'+subjid
    prefix, ext = os.path.splitext(mri)
    if ext=='.gz':   # for a .nii.gz mri, split againt to remove the .nii
        prefix, ext = os.path.splitext(prefix)
        ext = '.nii.gz'
    anon_mri = prefix+'_defaced'+ext 
    log_dir=f'{topdir}/logs'
    subj_logger=get_subj_logger('sub-'+subjid, log_dir=log_dir)
    subj_logger.info(f'Original:{mri}')
    subj_logger.info(f'Processed:{anon_mri}')
    
    deface_error = 0
    reconall_error = 0
    mkheadsurf_error = 0
    link_error = 0
    
    brain_template=f'{topdir}/staging_dir/talairach_mixed_with_skull.gca'
    face_template=f'{topdir}/staging_dir/face.gca'
    
    # Set subjects dir path for subcommand call
    os.environ['SUBJECTS_DIR']=subjects_dir
    
    try: 
        
        subprocess.run(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}'.split(),
                       check=True)
    except BaseException as e:
        subj_logger.error('MRI_DEFACE')
        subj_logger.error(e)
        deface_error = 1
    
    if anon_mri.split('/')[-1].split('.')[-1] == 'gz':
        log_fname = anon_mri.split('/')[-1].split('.')[0] + '.nii.log'
    else: 
        log_fname = anon_mri.split('/')[-1].split('.')[0] + '.log'
        
    shutil.move(log_fname, 'logs/'+log_fname)
    
    # only do the freesurfer processing for the defaced MRI
        
    try:
        subprocess.run(f'recon-all -i {anon_mri} -s {bids_subjid}'.split(),
                       check=True)
        subprocess.run(f'recon-all -autorecon1 -noskullstrip -s {bids_subjid}'.split(),
                       check=True)
        subj_logger.info('RECON_ALL IMPORT (ANON) FINISHED')
    except BaseException as e:
        subj_logger.error('RECON_ALL IMPORT (ANON)')
        subj_logger.error(e)
        reconall_error = 1
    
    try:
        subprocess.run(f'mkheadsurf -s {subjid}'.split(), check=True)
        subj_logger.info('MKHEADSURF (ANON) FINISHED')
    except:
        try:
            proc_cmd = f"mkheadsurf -i {op.join(subjects_dir, bids_subjid, 'mri', 'T1.mgz')} \
                -o {op.join(subjects_dir, bids_subjid, 'mri', 'seghead.mgz')} \
                -surf {op.join(subjects_dir, bids_subjid, 'surf', 'lh.seghead')}"
            subprocess.run(proc_cmd.split(), check=True)
        except BaseException as e:
            subj_logger.error('MKHEADSURF (ANON)')
            subj_logger.error(e)    
            mkheadsurf_error = 1
                
    try:
        link_surf(subjid, subjects_dir=subjects_dir)      
    except BaseException as e:
        subj_logger.error('Link error on BEM')
        subj_logger.error(e)
        link_error = 1
    return (deface_error, reconall_error, mkheadsurf_error, link_error)
        
# This is the same function as above, essentially, except it makes the scalp surfaces
# without anonymizing the MRI
        
def make_scalp_surfaces(mri=None, subjid=None, subjects_dir=None, 
                             topdir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    bids_subjid = 'sub-'+subjid
    log_dir=f'{topdir}/logs'
    subj_logger=get_subj_logger(bids_subjid, log_dir=log_dir)
    subj_logger.info('MRI WILL NOT BE DEFACED, DO NOT SHARE RAW DATA')
    subj_logger.info(f'Original:{mri}')
    
    reconall_error = 0
    mkheadsurf_error = 0
    link_error = 0
    
    # Set subjects dir path for subcommand call
    os.environ['SUBJECTS_DIR']=subjects_dir

    try:
        subprocess.run(f'recon-all -i {mri} -s {bids_subjid}'.split(),
                       check=True)
        subprocess.run(f'recon-all -autorecon1 -noskullstrip -s {bids_subjid}'.split(),
                       check=True)
        subj_logger.info('RECON_ALL IMPORT INISHED')
    except BaseException as e:
        subj_logger.error('RECON_ALL IMPORT')
        subj_logger.error(e)
        reconall_error = 1
    
    try:
        subprocess.run(f'mkheadsurf -s {bids_subjid}'.split(), check=True)
        subj_logger.info('MKHEADSURF FINISHED')
    except:
        try:
            proc_cmd = f"mkheadsurf -i {op.join(subjects_dir, bids_subjid, 'mri', 'T1.mgz')} \
                -o {op.join(subjects_dir, bids_subjid, 'mri', 'seghead.mgz')} \
                -surf {op.join(subjects_dir, bids_subjid, 'surf', 'lh.seghead')}"
            subprocess.run(proc_cmd.split(), check=True)
        except BaseException as e:
            subj_logger.error('MKHEADSURF (ANON)')
            subj_logger.error(e)    
            mkheadsurf_error = 1
                
    try:
        link_surf(subjid, subjects_dir=subjects_dir)      
    except BaseException as e:
        subj_logger.error('Link error on BEM')
        subj_logger.error(e)
        link_error = 1
    return (reconall_error, mkheadsurf_error, link_error)
        
def parallel_make_scalp_surfaces(dframe, topdir=None, subjects_dir=None, njobs=1, bidsonly=0):
    logger = logging.getLogger('process_logger')
    ## SETUP MAKE_SCALP_SURFACES
    inframe = dframe.loc[:,['staged_mri','subjid']]
    inframe['subjects_dir'] = subjects_dir
    inframe['topdir']  = topdir
    #Remove duplicates over sessions
    inframe.drop_duplicates(subset=['staged_mri','subjid'], inplace=True)
    logger.info('Opening parallel pool for freesurfer processing')
    if bidsonly==1:
        with Pool(processes=njobs) as pool:
            result = pool.starmap(make_scalp_surfaces,
                          inframe.values)
            try: 
                del inframe 
            except:
                pass
    else:
        with Pool(processes=njobs) as pool:
            result = pool.starmap(make_scalp_surfaces_anon,
                          inframe.values)
            try: 
                del inframe 
            except:
                pass
    total = tuple(sum(x) for x in zip(*result))

    logger.info('Finished parallel freesurfer processing for all subjects')
    if bidsonly == 1:
        logger.info('Total of %d reconall errors, %d mkheadsurf errors, and %d link errors' % total)
    else:
        logger.info('Total of %d deface errors, %d reconall errors, %d mkheadsurf errors, and %d link errors' % total)    
        
# function to read the MEG scans into mne python for processing

def read_meg(meg_fname):

    if meg_fname[-1]==os.path.sep:
        meg_fname=meg_fname[:-1]
    path, ext = os.path.splitext(meg_fname)
    if ext == '.fif':
        return mne.io.read_raw_fif(meg_fname)   
    if ext == '.ds':
        return mne.io.read_raw_ctf(meg_fname)
    if ext == '.sqd':
        return mne.io.read_raw_kit(meg_fname)
    if ext == '':
        if  ',' in os.path.basename(meg_fname):
            raw=mne.io.read_raw_bti(meg_fname, head_shape_fname=None)
            #raw=mne.io.read_raw_bti(meg_fname, head_shape_fname=None, convert=False)     # This code can be uncommented and used if the BTI/4D code
            #for i in range(len(raw.info['chs'])):ArithmeticError                         # was processed similar to the HCP data
            #    raw.info['chs'][i]['coord_frame'] = 1
            return raw
        
def process_mri_bids(dframe=None, topdir=None, bidsonly=0):
    logger = logging.getLogger('process_logger')
    logger.info('Populating BIDS structure with MRI scans')
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)

    for idx, row in dframe.iterrows():
            subj_logger = get_subj_logger('sub-'+row.subjid, log_dir=f'{topdir}/logs')
            try:
                sub=row['subjid']
                ses=str(row['session'])
                output_path = f'{topdir}/bids_out'
            
                raw = read_meg(row['full_meg_path'])          #FIX currently should be full_meg_path - need to verify anon
                trans = mne.read_trans(row['trans_fname'])
                t1_fname = 'sub-%s_anat_defaced.nii' % sub
                if op.exists(op.join(staging_dir, t1_fname)):
                    t1_path = op.join(topdir,'staging_dir',t1_fname)
                else:
                    t1_fname = 'sub-%s_anat_defaced.nii.gz' % sub
                    t1_path = op.join(topdir,'staging_dir',t1_fname)                  
                
                t1w_bids_path = \
                    BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
                    
                fs_subject='sub-'+row['subjid']
                    
                landmarks = get_anat_landmarks(
                    image=t1_path,
                    info=raw.info,
                    trans=trans,
                    fs_subject=fs_subject,
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
    logger.info('Finished MRI scans')
            
def process_meg_bids(dframe=None, topdir=None, linefreq=60, bidsonly=0):
    logger = logging.getLogger('process_logger')
    logger.info('Populating BIDS structure with MEG recordings')
    bids_dir = f'{topdir}/bids_out'
    if not os.path.exists(bids_dir): os.mkdir(bids_dir)
    if set(['empty_room']).issubset(dframe.columns):
        eroom=True
    else:
        eroom=False
        
    for idx, row in dframe.iterrows():
        
        subj_logger = get_subj_logger('sub-'+row.subjid, log_dir=f'{topdir}/logs')
        
        try:
            sub = row['subjid']
            raw_fname = row['full_meg_path']
            output_path = bids_dir            
            raw = read_meg(raw_fname)  
            raw.info['line_freq'] = linefreq 
            ses=str(row['session'])
            task = 'rest'
            run = '01'
            
            bids_path = BIDSPath(subject=sub, session=ses, task=task,
                                  run=run, root=output_path, suffix='meg')
                
            if bidsonly == 0:
                daysback = 35000 + randint(1, 1000)
                write_raw_bids(raw, bids_path, anonymize={'daysback':daysback, 'keep_his':False, 'keep_source':False},
                      overwrite=True)
            else:
                write_raw_bids(raw, bids_path, overwrite=True)
            
        except BaseException as e:
            subj_logger.exception('MEG BIDS PROCESSING:', e)
            
        if eroom == True:
           try:
               eroom_fname = row['empty_room']
               eroom = read_meg(eroom_fname)
               eroom.info['line_freq'] = linefreq 
               ses=str(row['session'])
               task = 'emptyroom'
               run = '01'
               
               bids_path = BIDSPath(subject=sub, session=ses, task=task,
                                     run=run, root=output_path, suffix='meg')
               if bidsonly == 0:
                   daysback = 35000 + randint(1, 1000)  
                   write_raw_bids(eroom, bids_path, anonymize={'daysback':daysback, 'keep_his':False, 'keep_source':False},
                         overwrite=True)
               else:
                   write_raw_bids(eroom, bids_path, overwrite=True)
                
           except BaseException as e:
                    subj_logger.exception('MEG BIDS PROCESSING:', e)   
    logger.info('Finished MEG recordings')
            
def loop_QA_report(dframe, subjects_dir=None, topdir=None, bidsonly=0):
    logger = logging.getLogger('process_logger')
    logger.info('Beginning creation of QA reports')
    from mne.viz._brain.view import views_dicts
    from mne.viz import set_3d_view
    from mne.viz.backends.renderer import backend
    #from mne.viz.utils import _ndarray_to_fig
    png_path=op.join(topdir,'bids_out/derivatives/BIDS_ANON_QA/')
    report_path=op.join(topdir, 'bids_out/derivatives/BIDS_ANON_QA/Coreg_QA_report.html')
    rep = mne.Report()

    for idx, row in dframe.iterrows():
        subjid=row['subjid']
        subject = 'sub-'+subjid
        meg_fname=row['full_meg_path']
        raw = read_meg(meg_fname)
        subjects_dir=subjects_dir               
        trans=row['trans_fname']
        title_text = 'Check coreg and deface (if performed) for subjid %s' % subject
        subj_logger=get_subj_logger('sub-'+subjid, log_dir=f'{topdir}/logs')
        try:
            subj_logger.info('Running QA report')
            rep.add_trans(trans=trans, info=raw.info, subject=subject,
                    subjects_dir=subjects_dir, alpha=1.0, title=title_text) 
        except BaseException as e:
            subj_logger.error(f'make_QA_report: \n{e}')
        fig = mne.viz.plot_alignment(info=raw.info, trans=trans, subject=subject, 
                    subjects_dir=subjects_dir)
        set_3d_view(fig,**views_dicts['both']['frontal'])
        img1=fig.plotter.screenshot()
        fig.plotter.close()
        fig = mne.viz.plot_alignment(info=raw.info, trans=trans, subject=subject, 
                    subjects_dir=subjects_dir)
        set_3d_view(fig,**views_dicts['both']['lateral'])
        img2=fig.plotter.screenshot()
        fig.plotter.close()
        fig = mne.viz.plot_alignment(info=raw.info, trans=trans, subject=subject, 
                    subjects_dir=subjects_dir)
        set_3d_view(fig,**views_dicts['both']['medial'])
        img3=fig.plotter.screenshot()
        fig.plotter.close()
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(img1)
        tmp=ax[0].axis('off')
        ax[1].imshow(img2)
        tmp=ax[1].axis('off')
        ax[2].imshow(img3)
        tmp=ax[2].axis('off')
        figname = op.join(png_path, subject+'_coreg.png')
        fig.savefig(figname, dpi=300,bbox_inches='tight')
        plt.close(fig)
    logger.info('Finished QA reports')

    rep.save(fname=report_path, open_browser=False, overwrite=True)