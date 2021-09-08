#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: stoutjd
"""
import os
import shutil
import mne
import subprocess
import matplotlib
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 
from mne import report

def subcommand(function_str):
    from subprocess import check_call
    check_call(function_str.split(' '))
    
#%% Setup and run freesurfer commands for 
mri='MRIS_ORIG/APBWVFAR/APBWVFAR.nii'
subjid = 'tmp-1'
brain_template='/home/stoutjd/src/enigma_meg_prep/setup_code/talairach_mixed_with_skull.gca'
face_template='/home/stoutjd/src/enigma_meg_prep/setup_code/face.gca'
outmri='./deface.nii'
subjects_dir =os.path.join(os.getcwd(),'SUBJECTS_DIR')
os.environ['SUBJECTS_DIR'] = subjects_dir
mne_bids_pipeline_path = '/home/stoutjd/src/mne-bids-pipeline/run.py'
report_path = './report.html'
meg_fname = '/data/enigma_prep/MEG/APBWVFAR_rest_20200122_03.ds'
trans_fname = '/data/enigma_prep/transfiles/APBWVFAR-trans.fif'

def read_meg(meg_fname):
    if meg_fname[-1]==os.path.sep:
        meg_fname=meg_fname[:-1]
    _, ext = os.path.splitext(meg_fname)
    if ext == '.fif':
        return mne.io.read_raw_fif(meg_fname)
    if ext == '.ds':
        return mne.io.read_raw_ctf(meg_fname)


def make_scalp_surfaces_anon(mri=None, subjid=None, subjects_dir=None):
    '''
    Process scalp surfaces for subjid and anon_subjid
    Render the coregistration for the subjid
    Render the defaced Scalp for the subjid_anon
    '''
    
    tmpdir = os.path.join(os.getcwd(), '_tmp')
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    else:
        os.mkdir(tmpdir)
    
    anon_subjid = subjid+'_defaced'
    _tmp, ext = os.path.splitext(mri)
    base = os.path.basename(_tmp)
    anon_mri = os.path.join(tmpdir,base+'_defaced'+ext)
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



def make_upload_report(subjid=None, subjects_dir=None, 
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
    
    

def link_surf(subjid=None, subjects_dir=None):
    # Link files
    s_bem_dir = f'{subjects_dir}/{subjid}/bem'
    if not os.path.exists(s_bem_dir):
        os.mkdir(s_bem_dir)
    if not os.path.exists(f'{s_bem_dir}/outer_skin.surf'):
        src_file = f'{subjects_dir}/{subjid}/surf/lh.seghead'
        os.symlink(src_file, f'{s_bem_dir}/outer_skin.surf')



make_scalp_surfaces_anon(mri=mri, subjid=subjid, subjects_dir=subjects_dir)
make_upload_report(subjid=subjid, subjects_dir=subjects_dir, 
                 report_path=report_path, meg_fname=meg_fname, trans=trans_fname)






# plt.ion(); 
# tmp=mne.viz.plot_alignment(subject=subjid);
# test=input('Hit enter to close:');
