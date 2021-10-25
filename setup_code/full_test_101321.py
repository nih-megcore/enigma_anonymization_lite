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

import wget #  !pip install wget
import gzip

import shutil
import matplotlib
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 

from mne_bids import write_anat, BIDSPath, write_raw_bids

#%% Download required files


#%% Setup
n_jobs=6
line_freq = 60.0

topdir = '/home/stojek/enigma_final'
os.chdir(topdir)

# Create freesurfer folder
subjects_dir = f'{topdir}/SUBJECTS_DIR'
if not os.path.exists(subjects_dir): os.mkdir(subjects_dir)
os.environ['SUBJECTS_DIR'] = subjects_dir

#Directory to save html files
QA_dir = f'{topdir}/QA'
if not os.path.exists(QA_dir): os.mkdir(QA_dir)

# Create setup directory

code_topdir=f'{topdir}/setup_code'
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
    

#%%  Setup paths and confirm version
assert mne_bids.__version__[0:3]>='0.8'

code_topdir=f'{topdir}/setup_code' 

brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
assert os.path.exists(brain_template)
face_template=f'{code_topdir}/face.gca'
assert os.path.exists(face_template)

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
    

#%% Setup dataframe for processing
meg_template = '/home/stojek/enigma_final/meg/{DATE}/{SUBJID}_{TASK}_{DATE}_??_anon.ds'
mri_template = '/home/stojek/enigma_final/mris_nii/{SUBJID}/{SUBJID}.nii'

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

if not 'combined_dframe' in locals().keys():
    combined_dframe = pd.read_csv('MasterList.csv', index_col='Unnamed: 0')

combined_dframe['fs_T1mgz'] = combined_dframe.bids_subjid.apply(lambda x: op.join(subjects_dir, x, 'mri', 'T1.mgz'))
combined_dframe['report_path'] = combined_dframe.bids_subjid.apply(lambda x: op.join(QA_dir, x+'_report.html'))

dframe=combined_dframe
dframe['subjects_dir'] = subjects_dir
assign_report_path(dframe, f'{topdir}/QA')

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
    anon_mri = prefix+'_defaced'+ext #os.path.join(prefix+'_defaced'+ext)
    print(f'Original:{mri}',f'Processed:{anon_mri}')
    
    # Set subjects dir path for subcommand call
    os.environ['SUBJECTS_DIR']=subjects_dir
    
    try: 
        subcommand(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}')
    except:
        pass
    
    try:
        proc_o = [f'recon-all -i {mri} -s {subjid}',
                f'recon-all -autorecon1 -noskullstrip -s {subjid}',
                ]
        proc_tmp = f"mkheadsurf -i {op.join(subjects_dir, subjid, 'mri', 'T1.mgz')} \
            -o {op.join(subjects_dir, subjid, 'mri', 'seghead.mgz')} \
            -surf {op.join(subjects_dir, subjid, 'surf', 'lh.seghead')}"
        proc_tmp = ' '.join(proc_tmp.split()) #Remove extra whitespace                          
        proc_o.append(proc_tmp)

        for proc in proc_o:
            subcommand(proc)    
    except:
        pass
    
    try:
        # Process anonymized head surface
        proc_a = [
                f'recon-all -i {anon_mri} -s {anon_subjid}',
                f'recon-all -autorecon1 -noskullstrip -s {anon_subjid}',
                ]
        proc_a_tmp = f"mkheadsurf -i {op.join(subjects_dir, anon_subjid, 'mri', 'T1.mgz')} \
            -o {op.join(subjects_dir, anon_subjid, 'mri', 'seghead.mgz')} \
            -surf {op.join(subjects_dir, anon_subjid, 'surf', 'lh.seghead')}"
        proc_a_tmp = ' '.join(proc_a_tmp.split()) #Remove extra whitespace       
        proc_a.append(proc_a_tmp)
        for proc in proc_a:
            subcommand(proc)
    except:
        pass
    
    try:
        # Cleanup
        link_surf(subjid, subjects_dir=subjects_dir)
        link_surf(anon_subjid, subjects_dir=subjects_dir)      
    except:
        pass

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
                                 trans=trans, dig=True, coord_frame='meg')    
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
    
dframe.to_csv('MasterList.csv')
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

if not os.path.exists(f'{topdir}/trans_mats'): os.mkdir(f'{topdir}/trans_mats')

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

                   
# ## SEtup report    
# inframe = dframe.loc[:,['bids_subjid','subjects_dir','report_path', 
#                         'full_meg_path','trans_fname']]

# ## SETUP Make Report
# with Pool(processes=n_jobs) as pool:
#     pool.starmap(make_QA_report, inframe.values)
  

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








#%%    
# =============================================================================
# STOP HERE PRIOR TO BIDS PROCESSING
# =============================================================================

#%% Process BIDS
bids_dir = f'{topdir}/bids_out'
if not os.path.exists(bids_dir): os.mkdir(bids_dir)

combined_dframe = dframe
# =============================================================================
# Convert MEG
# =============================================================================
# for idx, row in combined_dframe.iterrows():
#     print(idx)
#     print(row)
#     raw_fname = row['meg_fname'] #row.full_meg_path
#     output_path = bids_dir
    
#     raw = read_meg(raw_fname)  
#     raw.info['line_freq'] = line_freq 
    
#     sub = row['bids_subjid'][4:] #Remove sub- prefix from name
#     ses = '01'
#     task = 'rest'
#     run = '01'
#     bids_path = BIDSPath(subject=sub, session=ses, task=task,
#                          run=run, root=bids_dir, suffix='meg')
#     write_raw_bids(raw, bids_path)
    
errors=[]
for idx, row in combined_dframe.iterrows():
    try:
        print(idx)
        print(row)
        subject = row.meg_subjid
        mri_fname = row.full_mri_path   #Will need to change to confirm anon
        raw_fname = row.full_meg_path
        output_path = bids_dir
        
        raw = mne.io.read_raw_ctf(raw_fname)  #Change this - should be generic for meg vender
        raw.info['line_freq'] = line_freq 
        
        sub = row['bids_subjid'][4:] 
        ses = '0'+str(int(row['meg_session']) )
        task = 'rest'
        run = '01'
        bids_path = BIDSPath(subject=sub, session=ses, task=task,
                             run=run, root=output_path, suffix='meg')
        
        write_raw_bids(raw, bids_path)
    except BaseException as e:
        errors.append(e)
        

#%% Create the bids from the anonymized MRI
for idx, row in dframe.iterrows():
    try:
        sub=row['bids_subjid'][4:] 
        ses='01'
        output_path = f'{topdir}/bids_out'
        
        raw = read_meg(row['meg_fname'])          #FIX currently should be full_meg_path - need to verify anon
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
    except:
        pass



# # =============================================================================
# # TEMP GET RID OF THIS
# # =============================================================================
# def make_scalp_surfaces_anon2(mri=None, subjid=None, subjects_dir=None):
#     '''
#     Process scalp surfaces for subjid and anon_subjid
#     Render the coregistration for the subjid
#     Render the defaced Scalp for the subjid_anon
#     '''
#     try: 
#         anon_subjid = subjid+'_defaced'
#         prefix, ext = os.path.splitext(mri)
#         anon_mri = prefix+'_defaced'+ext #os.path.join(prefix+'_defaced'+ext)
#         print(f'Original:{mri}',f'Processed:{anon_mri}')
        
#         # Set subjects dir path for subcommand call
#         os.environ['SUBJECTS_DIR']=subjects_dir
        
#         # proc_tmp = f"mkheadsurf -i {op.join(subjects_dir, subjid, 'mri', 'T1.mgz')} \
#         #     -o {op.join(subjects_dir, subjid, 'mri', 'seghead.mgz')} \
#         #     -surf {op.join(subjects_dir, subjid, 'surf', 'lh.smseghead')}"
#         # proc_tmp = ' '.join(proc_tmp.split())
        
#         # subcommand(proc_tmp)
        
#         # proc_tmp2 = f"mkheadsurf -i {op.join(subjects_dir, anon_subjid, 'mri', 'T1.mgz')} \
#         #     -o {op.join(subjects_dir, anon_subjid, 'mri', 'seghead.mgz')} \
#         #     -surf {op.join(subjects_dir, anon_subjid, 'surf', 'lh.smseghead')}"
#         # proc_tmp2 = ' '.join(proc_tmp2.split())
        
#         # subcommand(proc_tmp2)
                           
        
#         # Cleanup
#         link_surf(subjid, subjects_dir=subjects_dir)
#         link_surf(anon_subjid, subjects_dir=subjects_dir)      
#     except:
#         pass
    
# for idx, row in dframe.iterrows():
#     trans_fname=op.join('./trans_mats', row['bids_subjid']+'_'+str(int(row['meg_session']))+'-trans.fif')
#     dframe.loc[idx,'trans_fname']=trans_fname

 