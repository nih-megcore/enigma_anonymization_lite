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

import shutil
import matplotlib
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 

from mne_bids import write_anat, BIDSPath, write_raw_bids

#%% Setup
n_jobs=6
topdir = '/fast/TMP_10'
subjects_dir = f'{topdir}/SUBJECTS_DIR'
QA_dir = f'{topdir}/QA'
if not os.path.exists(QA_dir): os.mkdir(QA_dir)
os.chdir(topdir)


#%%
meg_template = '/fast/TMP_10/MEG/{DATE}/{SUBJID}_{TASK}_{DATE}_??.ds'
mri_template = '/fast/TMP_10/MRI/{SUBJID}/{SUBJID}.nii'

mri_dframe, meg_dframe = return_mri_meg_dframes(mri_template, meg_template)
        
meg_dframe['date'] = meg_dframe.full_meg_path.apply(_return_date, key_index=-3 )
meg_dframe['task'] = meg_dframe.full_meg_path.apply(lambda x: x.split('_')[-4])
not_rest_idxs = meg_dframe[~meg_dframe['task'].isin(['rest'])].index
meg_dframe.drop(index=not_rest_idxs, inplace=True)

meg_dframe.sort_values(['meg_subjid','date'])
meg_dframe = assign_meg_session(meg_dframe)

combined_dframe  = pd.merge(mri_dframe, meg_dframe, left_on='mri_subjid', 
                            right_on='meg_subjid')
combined_dframe.reset_index(drop=True, inplace=True)

import numpy as np
orig_subj_list = combined_dframe['meg_subjid'].unique()
tmp = np.arange(len(orig_subj_list), dtype=int)
np.random.shuffle(tmp)  #Change the index for subjids

for rnd_idx, subjid in enumerate(orig_subj_list):
    print(subjid)
    print(str(rnd_idx))
    subjid_idxs = combined_dframe[combined_dframe['meg_subjid']==subjid].index
    combined_dframe.loc[subjid_idxs,'bids_subjid'] = "sub-{0:0=4d}".format(tmp[rnd_idx])

bids_dir = './bids_out'
if not os.path.exists(bids_dir): os.mkdir(bids_dir)



subjects_dir=f'{topdir}/SUBJECTS_DIR'



if not 'combined_dframe' in locals().keys():
    combined_dframe = pd.read_csv('MasterList.csv', index_col='Unnamed: 0')

# line_freq = 60.0
combined_dframe['fs_T1mgz'] = combined_dframe.bids_subjid.apply(lambda x: op.join(subjects_dir, x, 'mri', 'T1.mgz'))
# combined_dframe['trans_fname'] = combined_dframe.subjids.apply(lambda x: f'{topdir}/transfiles/{x}-trans.fif')
combined_dframe['report_path'] = combined_dframe.bids_subjid.apply(lambda x: op.join(QA_dir, x+'_report.html'))
line_freq = 60.0
dframe=combined_dframe

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


#%%  Setup paths and confirm version
assert mne_bids.__version__[0:3]>='0.8'

code_topdir=f'{topdir}/setup_code' #/fast/enigma_meg_prep/setup_code'

brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
assert os.path.exists(brain_template)
face_template=f'{code_topdir}/face.gca'
assert os.path.exists(face_template)
subjects_dir =os.path.join(os.getcwd(),'SUBJECTS_DIR')
if not os.path.exists(subjects_dir): os.mkdir(subjects_dir)
os.environ['SUBJECTS_DIR'] = subjects_dir
dframe['subjects_dir'] = subjects_dir
mri_staging_dir = f'{topdir}/mri_staging'

#%%  Make headsurfaces to confirm alignment and defacing
# Setup and run freesurfer commands 

def make_scalp_surfaces_anon(mri=None, subjid=None, subjects_dir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    try: 
        anon_subjid = subjid+'_defaced'
        prefix, ext = os.path.splitext(mri)
        anon_mri = prefix+'_defaced'+ext #os.path.join(prefix+'_defaced'+ext)
        print(f'Original:{mri}',f'Processed:{anon_mri}')
        
        # Set subjects dir path for subcommand call
        os.environ['SUBJECTS_DIR']=subjects_dir
            
        subcommand(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}')
        
        proc_o = [f'recon-all -i {mri} -s {subjid}',
                f'recon-all -autorecon1 -noskullstrip -s {subjid}',
                f'mkheadsurf -s {subjid}',
                ]
        proc_tmp = f"mkheadsurf -i {op.join(subjects_dir, subjid, 'mri', 'T1.mgz')} \
            -o {op.join(subjects_dir, subjid, 'mri', 'seghead.mgz')} \
            -surf {op.join(subjects_dir, subjid, 'surf', 'lh.smseghead')}"
                           
        proc_o.append(proc_tmp)
        
        # # Process original head surface
        # proc_o = [f'recon-all -i {mri} -s {subjid}',
        #         f'recon-all -autorecon1 -noskullstrip -s {subjid}',
        #         f'mkheadsurf -s {subjid}',
        #         ]
        for proc in proc_o:
            subcommand(proc)    
        
        # Process anonymized head surface
        proc_a = [
                f'recon-all -i {anon_mri} -s {anon_subjid}',
                f'recon-all -autorecon1 -noskullstrip -s {anon_subjid}',
                f'mkheadsurf -s {anon_subjid}',
                ]
        for proc in proc_a:
            subcommand(proc)
    
        # Cleanup
        link_surf(subjid, subjects_dir=subjects_dir)
        link_surf(anon_subjid, subjects_dir=subjects_dir)      
    except:
        pass

# import copy
# import matplotlib
# def make_flr_image(mne_fig):
#     fig, axs = matplotlib.pyplot.subplots(2,2)
#     axs[0,0] = copy(mne_fig)
#     mne.viz.set_3d_view(im_r, azimuth=0)
#     im_f = copy(mne_fig)
#     mne.viz.set_3d_view(im_f, azimuth=45)
#     im_l = copy(mne_fig)
#     mne.viz.set_3d_view(im_l, azimuth=115)

def make_QA_report(subjid=None, subjects_dir=None, 
                 report_path=None, meg_fname=None, trans=None):
    try:
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
                                 trans=trans, dig=True)    
        rep.add_figs_to_section(figs=tmp, captions='Head Alignment',
                                        section='Coregistration')
        tmp = mne.viz.plot_alignment(subject=anon_subjid)
        rep.add_figs_to_section(figs=tmp, captions='Deface Confirmation',
                                        section='Anonymization')    
        rep.save(fname=report_path)
    except:
        print(f'error {subjid}')

def test_make_QA_report():
    row = dframe.loc[2]
    subjid=row['bids_subjid']
    subjects_dir=subjects_dir
    report_path=row['report_path']
    meg_fname=row['full_meg_path']
    trans=row['trans_fname']
    make_QA_report(subjid=subjid, 
                   subjects_dir=subjects_dir, 
                   report_path=report_path, 
                   meg_fname=meg_fname, 
                   trans=trans)
    
    
def validate_paths(in_dframe):
    for i in in_dframe.values:
        for j in i:
            if os.ispath(j):
                if not os.path.exists(j):
                    print(f'Does not exists: {j}')
        

def link_surf(subjid=None, subjects_dir=None):
    # Link files
    s_bem_dir = f'{subjects_dir}/{subjid}/bem'
    if not os.path.exists(s_bem_dir):
        os.mkdir(s_bem_dir)
    if not os.path.exists(f'{s_bem_dir}/outer_skin.surf'):
        src_file = f'{subjects_dir}/{subjid}/surf/lh.seghead'
        os.symlink(src_file, f'{s_bem_dir}/outer_skin.surf')

#%% Process Data
# =============================================================================
# Make muliprocess calls looped over subjid
# =============================================================================
from multiprocessing import Pool

# Copy T1 from original location to staging area
dframe['T1staged'] = ''
if not os.path.exists(mri_staging_dir): os.mkdir(mri_staging_dir)
for idx,row in dframe.iterrows(): #t1_fname in dframe['T1nii']:
    _, ext = os.path.splitext(row['full_mri_path'])
    out_fname = row['bids_subjid']+ext
    out_path = op.join(mri_staging_dir, out_fname) 
    shutil.copy(row['full_mri_path'], out_path)
    dframe.loc[idx, 'T1staged'] = out_path
    
## SETUP MAKE_SCALP_SURFACES
inframe = dframe.loc[:,['T1staged','bids_subjid', 'subjects_dir']]
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




## For NIH - create transformation matrix
from nih2mne.calc_mnetrans import write_mne_fiducials 
from nih2mne.calc_mnetrans import write_mne_trans

for idx, row in dframe.iterrows():
    if op.splitext(row['full_mri_path'])[-1] == '.gz':
        afni_fname=row['full_mri_path'].replace('.nii.gz','+orig.HEAD')
    else:
        afni_fname=row['full_mri_path'].replace('.nii','+orig.HEAD')
    fid_path = op.join('./trans_mats', f'{row["bids_subjid"]}_{str(int(row["meg_session"]))}-fiducials.fif')
    try:
        write_mne_fiducials(subject=row['bids_subjid'],
                            subjects_dir=subjects_dir, 
                            afni_fname=afni_fname,
                            output_fid_path=fid_path)
    except:
        print('error '+row['bids_subjid'])

    try:              
        trans_fname=op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
        write_mne_trans(mne_fids_path=fid_path,
                        dsname=row['full_meg_path'], 
                        output_name=trans_fname, 
                        subjects_dir=subjects_dir)
        dframe.loc[idx,'trans_fname']=trans_fname
    except:
        print('error in trans calculation '+row['bids_subjid'])

 
                   
## SEtup report    
inframe = dframe.loc[:,['bids_subjid','subjects_dir','report_path', 
                        'full_meg_path','trans_fname']]

## SETUP Make Report
with Pool(processes=n_jobs) as pool:
    pool.starmap(make_QA_report, inframe.values)
  

for idx, row in dframe.iterrows():
    subjid=row['bids_subjid']
    subjects_dir=subjects_dir
    report_path=row['report_path']
    meg_fname=row['full_meg_path']
    trans=row['trans_fname']
    try:
        make_QA_report(subjid=subjid, 
                       subjects_dir=subjects_dir, 
                       report_path=report_path, 
                       meg_fname=meg_fname, 
                       trans=trans)
    except:
        pass

    


#%% Process BIDS
bids_dir = f'{topdir}/bids_out'
if not os.path.exists(bids_dir): os.mkdir(bids_dir)

combined_dframe = dframe
# =============================================================================
# Convert MEG
# =============================================================================
for idx, row in combined_dframe.iterrows():
    print(idx)
    print(row)
    raw_fname = row['meg_fname'] #row.full_meg_path
    output_path = bids_dir
    
    raw = read_meg(raw_fname)  
    raw.info['line_freq'] = line_freq 
    
    sub = row['bids_subjid'][4:] #Remove sub- prefix from name
    ses = '01'
    task = 'rest'
    run = '01'
    bids_path = BIDSPath(subject=sub, session=ses, task=task,
                         run=run, root=bids_dir, suffix='meg')
    write_raw_bids(raw, bids_path)

#%% Create the bids from the anonymized MRI
for idx, row in dframe.iterrows():
    sub=row['bids_subjid'][4:] 
    ses='01'
    output_path = f'{topdir}/bids_out'
    
    raw = read_meg(row['meg_fname'])
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



# # =============================================================================
# # TEMP GET RID OF THIS
# # =============================================================================
def make_scalp_surfaces_anon2(mri=None, subjid=None, subjects_dir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    try: 
        anon_subjid = subjid+'_defaced'
        prefix, ext = os.path.splitext(mri)
        anon_mri = prefix+'_defaced'+ext #os.path.join(prefix+'_defaced'+ext)
        print(f'Original:{mri}',f'Processed:{anon_mri}')
        
        # Set subjects dir path for subcommand call
        os.environ['SUBJECTS_DIR']=subjects_dir
        
        # proc_tmp = f"mkheadsurf -i {op.join(subjects_dir, subjid, 'mri', 'T1.mgz')} \
        #     -o {op.join(subjects_dir, subjid, 'mri', 'seghead.mgz')} \
        #     -surf {op.join(subjects_dir, subjid, 'surf', 'lh.smseghead')}"
        # proc_tmp = ' '.join(proc_tmp.split())
        
        # subcommand(proc_tmp)
        
        # proc_tmp2 = f"mkheadsurf -i {op.join(subjects_dir, anon_subjid, 'mri', 'T1.mgz')} \
        #     -o {op.join(subjects_dir, anon_subjid, 'mri', 'seghead.mgz')} \
        #     -surf {op.join(subjects_dir, anon_subjid, 'surf', 'lh.smseghead')}"
        # proc_tmp2 = ' '.join(proc_tmp2.split())
        
        # subcommand(proc_tmp2)
                           
        
        # Cleanup
        link_surf(subjid, subjects_dir=subjects_dir)
        link_surf(anon_subjid, subjects_dir=subjects_dir)      
    except:
        pass
    
for idx, row in dframe.iterrows():
    trans_fname=op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
    dframe.loc[idx,'trans_fname']=trans_fname

 