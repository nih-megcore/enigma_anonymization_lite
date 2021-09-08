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

topdir = '/data/enigma_prep'
subjects_dir = f'{topdir}/SUBJECTS_DIR'

subjids = os.listdir(f'{topdir}/MRIS_ORIG')
dframe = pd.DataFrame(subjids, columns=['subjids'])
dframe['fs_id']=dframe.subjids.apply(lambda x: x+'_fs')
dframe['fs_T1mgz'] = dframe.fs_id.apply(lambda x: op.join(subjects_dir, x, 'mri', 'T1.mgz'))
dframe['trans_fname'] = dframe.subjids.apply(lambda x: f'{topdir}/transfiles/{x}-trans.fif')
dframe['meg_fname'] = dframe.subjids.apply(lambda x: glob.glob(f'{topdir}/MEG/{x}_*rest*.ds')[0])

for idx, row in dframe.iterrows():
    trans = mne.read_trans(row['trans_fname'])
        

#%%

assert mne_bids.__version__[0:3]=='0.8'


for idx, row in dframe.iterrows():
    sub=str(idx)
    ses='01'
    output_path = f'{topdir}/bids_out'
    output_path_anon = f'{topdir}/bids_out_anon'
    
    raw = mne.io.read_raw_ctf(row['meg_fname'])
    
    t1w_bids_path = \
        BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')
    t1w_bids_path_anon = \
        BIDSPath(subject=sub, session=ses, root=output_path_anon, suffix='T1w')

    
    landmarks = mne_bids.get_anat_landmarks(
        image=row['fs_T1mgz'],
        info=raw.info,
        trans=trans,
        fs_subject=row['fs_id'],
        fs_subjects_dir=subjects_dir
        )
    
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



###############################################################################
# We can save the MRI to our existing BIDS directory and at the same time
# create a JSON sidecar file that contains metadata, we will later use to
# retrieve our transformation matrix :code:`trans`. The metadata will here
# consist of the coordinates of three anatomical landmarks (LPA, Nasion and
# RPA (=left and right preauricular points) expressed in voxel coordinates
# w.r.t. the T1 image.

# First create the BIDSPath object.
t1w_bids_path = \
    BIDSPath(subject=sub, session=ses, root=output_path, suffix='T1w')

# # We use the write_anat function
# t1w_bids_path = write_anat(
#     image=t1_mgh_fname,  # path to the MRI scan
#     bids_path=t1w_bids_path,
#     raw=raw,  # the raw MEG data file connected to the MRI
#     trans=trans,  # our transformation matrix
#     verbose=True, # this will print out the sidecar file
#     overwrite=True
# )
anat_dir = t1w_bids_path.directory

###############################################################################
# Let's have another look at our BIDS directory
print_dir_tree(output_path)


#%%
###############################################################################
# Our BIDS dataset is now ready to be shared. We can easily estimate the
# transformation matrix using ``MNE-BIDS`` and the BIDS dataset.
estim_trans = get_head_mri_trans(bids_path=bids_path,
                                 fs_subject='sub-'+sub,
                                 fs_subjects_dir=subjects_dir)

###############################################################################
# Finally, let's use the T1 weighted MRI image and plot the anatomical
# landmarks Nasion, LPA, and RPA onto the brain image. For that, we can
# extract the location of Nasion, LPA, and RPA from the MEG file, apply our
# transformation matrix :code:`trans`, and plot the results.

# Get Landmarks from MEG file, 0, 1, and 2 correspond to LPA, NAS, RPA
# and the 'r' key will provide us with the xyz coordinates. The coordinates
# are expressed here in MEG Head coordinate system.
pos = np.asarray((raw.info['dig'][0]['r'],
                  raw.info['dig'][1]['r'],
                  raw.info['dig'][2]['r']))

# We now use the ``head_to_mri`` function from MNE-Python to convert MEG
# coordinates to MRI scanner RAS space. For the conversion we use our
# estimated transformation matrix and the MEG coordinates extracted from the
# raw file. `subjects` and `subjects_dir` are used internally, to point to
# the T1-weighted MRI file: `t1_mgh_fname`. Coordinates are is mm.
mri_pos = head_to_mri(pos=pos,
                      subject='sample',
                      mri_head_t=estim_trans,
                      subjects_dir=op.join(data_path, 'subjects'))

# Our MRI written to BIDS, we got `anat_dir` from our `write_anat` function
t1_nii_fname = op.join(anat_dir, 'sub-1_ses-01_T1w.nii.gz')

# Plot it
fig, axs = plt.subplots(3, 1, figsize=(7, 7), facecolor='k')
for point_idx, label in enumerate(('LPA', 'NAS', 'RPA')):
    plot_anat(t1_nii_fname, axes=axs[point_idx],
              cut_coords=mri_pos[point_idx, :],
              title=label, vmax=160)
plt.show()

