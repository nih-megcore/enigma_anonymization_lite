#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:06:31 2022

@author: jstout
"""


import os.path as op
import os
from scipy.io import loadmat
import json
import glob

import nibabel as nb
import copy
from mne.transforms import apply_trans

# bstorm_datadir='/fast/MATLAB_toolboxes/brainstorm_data/Protocol03'
# subjid = 'Subject01'
# subjid = 'Subject_mne_default'

def bstormFid2dict(subjid, bstorm_datadir):
    '''
    Parameters
    ----------
    subjid : str
        Brainstorm subjectID.
    bstorm_datadir : str / path
        Brainstorm data directory - must include protocol.


    Returns
    -------
    fids_locs : dict
        Locations of anatomical fiducials.

    '''
    subj_anat=op.join(bstorm_datadir, 'anat', subjid, 'subjectimage_MRI.mat')
    mat = loadmat(subj_anat)
    
    #There is probably a more elegant way to do this
    tmp_ = str(mat['SCS'][0].dtype).split(',')[0:6:2]
    fids = [i.replace('[','').replace('"','').replace('(','').replace(' ','').replace("'",'') for i in tmp_]
    
    fid_locs={}
    for i in range(len(fids)):
        tmp_ = mat['SCS'][0][0][i][0]+ mat['InitTransf'][0][1][0:3,3]
        fid_locs[fids[i]]=tmp_
    return fid_locs


def convert2vox(t1_fname, coords_dict):
    t1 = nb.load(t1_fname)
    t1_mgh = nb.MGHImage(t1.dataobj, t1.affine)
    
    vox_coords = copy.deepcopy(coords_dict)
    for fid in vox_coords.keys():
        tmp_ = apply_trans(t1_mgh.header.get_ras2vox(), coords_dict[fid])
        vox_coords[fid] = list(tmp_)
    return vox_coords


def add_fids2json(fid_locs=None, bids_id=None, bids_root=None):
    '''
    Add or Write the T1w.json file with anatomicallandmark tag
    
    Find the anatomy folder in bids tree, read json, write json

    Parameters
    ----------
    fid_locs : dict, required
        Dictionary of anatomical landmarks.  Use bstormFid2dict to generate.
    bids_id : str, required
        Bids ID in the bids_root. The default is None.
    bids_root : str/path, required
        Root path of BIDS. The default is None.

    Raises
    ------
    ValueError
        If too many T1w files in bids anat dir.
        If no T1w files in bids anat dir.
        If T1w does not end in either .nii or .nii.gz
    BaseException
        If anatomical landmarks are already present in the T1w.json file

    Returns
    -------
    None.

    '''
    
    tmp=op.join(bids_root, bids_id)
    for root, dirs, files in os.walk(tmp, topdown=True):#False):
        for name in dirs:
            if name=='anat':
                anat_dir=op.join(root, name)
    T1list = glob.glob(op.join(anat_dir, '*T1w.nii*'))
    if len(T1list)>1:
        raise ValueError(f'More than on T1w files, cannot chose: {T1list}')
    if len(T1list)==0:
        raise ValueError(f'No T1w files in the subject folder {anat_dir}')
    T1w_fname=T1list[0]
    if os.path.splitext(T1w_fname)[-1] == '.gz':
        T1json_fname=T1w_fname.replace('.nii.gz','.json')
    elif os.path.splitext(T1w_fname)[-1] == '.nii':
        T1json_fname=T1w_fname.replace('.nii','.json')
    else:
        raise ValueError(f'The T1w image does not appear to have a typical extension: {T1w_fname}')
    
    #Convert to Voxel Indices
    fid_locs = convert2vox(T1w_fname, fid_locs)
    
    #Convert to list of strings from numpy array
    for key in fid_locs:
        fid_locs[key]=[str(i) for i in fid_locs[key]]
    
    if op.exists(T1json_fname):
        with open(T1json_fname) as f:
            jdict = json.load(f) 
    else:
        jdict = {}
    if "AnatomicalLandmarkCoordinates" in jdict.keys():
        raise BaseException('Anatomical Landmarks are already defined in json')
    else:
        jdict["AnatomicalLandmarkCoordinates"] = fid_locs
    
    with open(T1json_fname, 'w') as w:
        json.dump(jdict, w, indent='    ')
    print('Successfully saved json file with anatomical landmarks')
            
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.description='''Use the brainstorm data to generate the NAS,LPA,RPA
                       locations and add them to the T1w.json file in the bids
                       directory for the subject'''
    parser.add_argument('-bids_root')
    parser.add_argument('-bids_id', help='Subject bids_id in the bids_root')
    parser.add_argument('-bst_id', help='Brainstorm subject ID')
    parser.add_argument('-bst_datapath', help='''Path to Brainstorm Protocol data
                        folder''')
    args = parser.parse_args()
    if not (args.bids_root and args.bids_id and args.bst_id and args.bst_datapath):
        raise ValueError('One or more of the required inputs have not been met')
    
    anat_dict = bstormFid2dict(args.bst_id,
                               args.bst_datapath)
    
    add_fids2json(fid_locs=anat_dict, 
                  bids_id=args.bids_id, 
                  bids_root=args.bids_root)
    
    
    
    
    
    
    
