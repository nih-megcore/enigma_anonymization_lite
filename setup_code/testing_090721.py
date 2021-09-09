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

from mne_bids import write_anat, BIDSPath


#%% Setup
n_jobs=10
topdir = '/fast/enigma_prep'
subjects_dir = f'{topdir}/SUBJECTS_DIR'
QA_dir = f'{topdir}/QA'
if not os.path.exists(QA_dir): os.mkdir(QA_dir)
os.chdir(topdir)

subjids = os.listdir(f'{topdir}/MRIS_ORIG')
dframe = pd.DataFrame(subjids, columns=['subjids'])
dframe['fs_id']=dframe.subjids.apply(lambda x: x+'_fs')
dframe['fs_T1mgz'] = dframe.fs_id.apply(lambda x: op.join(subjects_dir, x, 'mri', 'T1.mgz'))
dframe['T1nii'] = dframe.subjids.apply(lambda x: f'{topdir}/MRIS_ORIG/{x}/{x}.nii')
dframe['trans_fname'] = dframe.subjids.apply(lambda x: f'{topdir}/transfiles/{x}-trans.fif')
dframe['meg_fname'] = dframe.subjids.apply(lambda x: glob.glob(f'{topdir}/MEG/{x}_*rest*.ds')[0])
dframe['bids_subjid'] = ['sub-'+"{0:0=4d}".format(i) for i in range(len(dframe))] 
dframe['report_path'] = dframe.bids_subjid.apply(lambda x: op.join(QA_dir, x+'_report.html'))

# =============================================================================
# Make a check function to verify that all files are present
# =============================================================================

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


#%%  Setup paths and confirm version
assert mne_bids.__version__[0:3]=='0.8'

code_topdir='/fast/enigma_meg_prep/setup_code'

brain_template=f'{code_topdir}/talairach_mixed_with_skull.gca'
assert os.path.exists(brain_template)
face_template=f'{code_topdir}/face.gca'
assert os.path.exists(face_template)
subjects_dir =os.path.join(os.getcwd(),'SUBJECTS_DIR')
if not os.path.exists(subjects_dir): os.mkdir(subjects_dir)
os.environ['SUBJECTS_DIR'] = subjects_dir
dframe['subjects_dir'] = subjects_dir
# mne_bids_pipeline_path = '/home/stoutjd/src/mne-bids-pipeline/run.py'
mri_staging_dir = f'{topdir}/mri_staging'

#%%  Make headsurfaces to confirm alignment and defacing
# Setup and run freesurfer commands 


def make_scalp_surfaces_anon(mri=None, subjid=None, subjects_dir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    
    tmpdir = os.path.join(os.getcwd(), '_tmp')
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    
    anon_subjid = subjid+'_defaced'
    _tmp, ext = os.path.splitext(mri)
    base = os.path.basename(_tmp)
    anon_mri = os.path.join(tmpdir,base+'_defaced'+ext)
    if os.path.exists(anon_mri): 
        #To prevent removing erroneous data make sure this is the correct dir
        assert op.basename(op.dirname(anon_mri)) == '_tmp'
        os.remove(anon_mri)           
    print(f'Original:{mri}',f'Processed:{anon_mri}')
    
    # Set subjects dir path for subcommand call
    os.environ['SUBJECTS_DIR']=subjects_dir
        
    subcommand(f'mri_deface {mri} {brain_template} {face_template} {anon_mri}')
    
    # Process original head surface
    proc_o = [f'recon-all -i {mri} -s {subjid}',
            f'recon-all -autorecon1 -noskullstrip -s {subjid}',
            f'mkheadsurf -s {subjid}',
            ]
    for proc in proc_o:
        subcommand(proc)    
    
    # Process anonymized head surface
    proc_a = [f'mri_deface {mri} {brain_template} {face_template} {anon_mri}', 
            f'recon-all -i {anon_mri} -s {anon_subjid}',
            f'recon-all -autorecon1 -noskullstrip -s {anon_subjid}',
            f'mkheadsurf -s {anon_subjid}',
            ]
    for proc in proc_a:
        subcommand(proc)

    # Cleanup
    link_surf(subjid, subjects_dir=subjects_dir)
    link_surf(anon_subjid, subjects_dir=subjects_dir)        
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)

import copy
import matplotlib
def make_flr_image(mne_fig):
    fig, axs = matplotlib.pyplot.subplots(2,2)
    axs[0,0] = copy(mne_fig)
    mne.viz.set_3d_view(im_r, azimuth=0)
    im_f = copy(mne_fig)
    mne.viz.set_3d_view(im_f, azimuth=45)
    im_l = copy(mne_fig)
    mne.viz.set_3d_view(im_l, azimuth=115)

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
                             trans=trans, dig=True)    
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
    # Link files
    s_bem_dir = f'{subjects_dir}/{subjid}/bem'
    if not os.path.exists(s_bem_dir):
        os.mkdir(s_bem_dir)
    if not os.path.exists(f'{s_bem_dir}/outer_skin.surf'):
        src_file = f'{subjects_dir}/{subjid}/surf/lh.seghead'
        os.symlink(src_file, f'{s_bem_dir}/outer_skin.surf')

#%% Loop over subjects to process freesurfer init

# # =============================================================================
# # Singlge subject processing
# # =============================================================================
# row = dframe.loc[0]
# idx=0

# mri = row['T1nii']
# subjid = str(idx)
# meg_fname = row['meg_fname']
# trans_fname = row['trans_fname']

# # mri='MRIS_ORIG/APBWVFAR/APBWVFAR.nii'
# # subjid = 'tmp-1'
# # outmri='./deface.nii'
# report_path = './report.html'
# meg_fname = '/data/enigma_prep/MEG/APBWVFAR_rest_20200122_03.ds'
# trans_fname = '/data/enigma_prep/transfiles/APBWVFAR-trans.fif'




# make_scalp_surfaces_anon(mri=mri, subjid=subjid, subjects_dir=subjects_dir)
# make_QA_report(subjid=subjid, subjects_dir=subjects_dir, 
#                  report_path=report_path, meg_fname=meg_fname, trans=trans_fname)


# =============================================================================
# Make muliprocess calls looped over subjid
# =============================================================================
#%% Process Data
from multiprocessing import Pool

# Copy T1 from original location to staging area
if not os.path.exists(mri_staging_dir): os.mkdir(mri_staging_dir)
for idx,row in dframe.iterrows(): #t1_fname in dframe['T1nii']:
    _, ext = os.path.splitext(row['T1nii'])
    out_fname = row['bids_subjid']+ext
    out_path = op.join(mri_staging_dir, out_fname) 
    shutil.copy(row['T1nii'], out_path)
    


## SETUP MAKE_SCALP_SURFACES
inframe = dframe.loc[:,['T1nii','bids_subjid', 'subjects_dir']]
# inframe['subjects_dir']=[subjects_dir]*len(inframe)
with Pool(processes=n_jobs) as pool:
    pool.starmap(make_scalp_surfaces_anon,
                     inframe.values)

try: 
    del inframe 
except:
    pass
inframe = dframe.loc[:,['bids_subjid','subjects_dir','report_path', 'meg_fname','trans_fname']]

## SETUP Make Report
with Pool(processes=n_jobs) as pool:
    pool.starmap(make_QA_report, inframe.values)
                 


# pool = Pool(processes=10)
# pool.starmap(make_scalp_surfaces_anon,
#                      inframe.values)


#     L = pool.starmap(make_scalp_surfaces_anon, 
#                      zip([dframe['T1nii'],
#                       dframe['subjid'],
#                       [subjects_dir]*len(dframe),
#                       ]))
#     # M = pool.starmap(func, zip(a_args, repeat(second_arg)))
#     # N = pool.map(partial(func, b=second_arg), a_args)
#     # assert L == M == N


#%% Create the bids from the anonymized MRI
for idx, row in dframe.iterrows():
    sub=str(idx)
    ses='01'
    output_path = f'{topdir}/bids_out'
    output_path_anon = f'{topdir}/bids_out_anon'
    
    raw = mne.io.read_raw_ctf(row['meg_fname'])
    trans = mne.read_trans(row['trans_fname'])
    t1_path = row['T1nii']
    
    t1w_bids_path = \
        BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
    # t1w_bids_path_anon = \
    #     BIDSPath(subject=sub, session=ses, root=output_path_anon, suffix='T1w')


    landmarks = mne_bids.get_anat_landmarks(
        image=row['T1nii'],
        info=raw.info,
        trans=trans,
        fs_subject=row['fs_id'],
        fs_subjects_dir=subjects_dir
        )
    
    # landmarks = mne_bids.get_anat_landmarks(
    #     image=row['fs_T1mgz'],
    #     info=raw.info,
    #     trans=trans,
    #     fs_subject=row['fs_id'],
    #     fs_subjects_dir=subjects_dir
    #     )
    
    # Write regular
    t1w_bids_path = write_anat(
        image=t1_mgh_fname,
        bids_path=t1w_bids_path,
        landmarks=landmarks,
        deface=False,
        overwrite=True
        )
    
    # Write anonymized
    t1w_bids_path = write_anat(
        image=t1_mgh_fname,
        bids_path=t1w_bids_path_anon,
        landmarks=landmarks,
        deface=True,
        overwrite=True
        )    
    
    anat_dir = t1w_bids_path.directory   



#%% Write MEG    
line_freq = 60.0

for idx, row in dframe.iterrows():
    sub=str(idx)
    ses='01'
    output_path = f'{topdir}/bids_out'
    output_path_anon = f'{topdir}/bids_out_anon'
    
    raw = mne.io.read_raw_ctf(row['meg_fname'])
    raw.info['line_freq'] = line_freq 
    
    task = 'rest'
    run = '01'
    bids_path = BIDSPath(subject=sub, session=ses, task=task,
                         run=run, root=output_path, suffix='meg')
    write_raw_bids(raw, bids_path)
    
    # t1w_bids_path = \
    #     BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
    # t1w_bids_path_anon = \
    #     BIDSPath(subject=sub, session=ses, root=output_path_anon, suffix='T1w')

#%% 
for idx, row in combined_dframe.iterrows():
    print(idx)
    print(row)
    subject = row.meg_subjid
    mri_fname = row.full_mri_path
    raw_fname = row.full_meg_path
    output_path = bids_dir
    
    raw = mne.io.read_raw_ctf(raw_fname)  #Change this - should be generic for meg vender
    raw.info['line_freq'] = line_freq 
    
    # events = mne.make_fixed_length_events(raw, duration=4.0)

    sub = "{0:0=4d}".format(idx)
    ses = '01'
    task = 'rest'
    run = '01'
    bids_path = BIDSPath(subject=sub, session=ses, task=task,
                         run=run, root=output_path, suffix='meg')
    
    # write_raw_bids(raw, bids_path, events_data=events, event_id={'rest':1}) 
    write_raw_bids(raw, bids_path)




# #%%


    




# #%%

# ###############################################################################
# # We can save the MRI to our existing BIDS directory and at the same time
# # create a JSON sidecar file that contains metadata, we will later use to
# # retrieve our transformation matrix :code:`trans`. The metadata will here
# # consist of the coordinates of three anatomical landmarks (LPA, Nasion and
# # RPA (=left and right preauricular points) expressed in voxel coordinates
# # w.r.t. the T1 image.

# # First create the BIDSPath object.
# t1w_bids_path = \
#     BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')

# # # We use the write_anat function
# # t1w_bids_path = write_anat(
# #     image=t1_mgh_fname,  # path to the MRI scan
# #     bids_path=t1w_bids_path,
# #     raw=raw,  # the raw MEG data file connected to the MRI
# #     trans=trans,  # our transformation matrix
# #     verbose=True, # this will print out the sidecar file
# #     overwrite=True
# # )
# anat_dir = t1w_bids_path.directory

# ###############################################################################
# # Let's have another look at our BIDS directory
# print_dir_tree(output_path)


# #%%
# ###############################################################################
# # Our BIDS dataset is now ready to be shared. We can easily estimate the
# # transformation matrix using ``MNE-BIDS`` and the BIDS dataset.
# estim_trans = get_head_mri_trans(bids_path=bids_path,
#                                  fs_subject='sub-'+sub,
#                                  fs_subjects_dir=subjects_dir)

# ###############################################################################
# # Finally, let's use the T1 weighted MRI image and plot the anatomical
# # landmarks Nasion, LPA, and RPA onto the brain image. For that, we can
# # extract the location of Nasion, LPA, and RPA from the MEG file, apply our
# # transformation matrix :code:`trans`, and plot the results.

# # Get Landmarks from MEG file, 0, 1, and 2 correspond to LPA, NAS, RPA
# # and the 'r' key will provide us with the xyz coordinates. The coordinates
# # are expressed here in MEG Head coordinate system.
# pos = np.asarray((raw.info['dig'][0]['r'],
#                   raw.info['dig'][1]['r'],
#                   raw.info['dig'][2]['r']))

# # We now use the ``head_to_mri`` function from MNE-Python to convert MEG
# # coordinates to MRI scanner RAS space. For the conversion we use our
# # estimated transformation matrix and the MEG coordinates extracted from the
# # raw file. `subjects` and `subjects_dir` are used internally, to point to
# # the T1-weighted MRI file: `t1_mgh_fname`. Coordinates are is mm.
# mri_pos = head_to_mri(pos=pos,
#                       subject='sample',
#                       mri_head_t=estim_trans,
#                       subjects_dir=op.join(data_path, 'subjects'))

# # Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
# t1_nii_fname = op.join(anat_dir, 'sub-1_ses-01_T1w.nii.gz')

# # Plot it
# fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor='k')
# for point_idx, label in enumerate(('LPA', 'NAS', 'RPA')):
#     plot_anat(t1_nii_fname, axes=axs[point_idx],
#               cut_coords=mri_pos[point_idx, :],
#               title=label, vmax=160)
# plt.show()

