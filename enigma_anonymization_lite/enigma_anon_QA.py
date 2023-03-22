#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:53:12 2023

@author: Allison Nugent and Jeff Stout
"""

import os.path as op
import sys
import argparse
import glob
from enigma_anonymization_lite.enigma_anon_QA_functions import initialize, get_last_review, build_status_dict
from enigma_anonymization_lite.enigma_anon_QA_functions import run_gui
from enigma_anonymization_lite.enigma_anon_QA_functions import sub_qa_info


PROJECT = 'BIDS_ANON_QA'
qa_type = 'Coreg'
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''Location of bids directory, default=bids_out''')
    parser.add_argument('-rows', help='''number of rows in QA browser, default=4''')
    parser.add_argument('-columns', help='''number of columns in QA browser, default=2''')
    parser.add_argument('-imgsize', help='''make images smaller or larger, default=200''')
    
    args = parser.parse_args()
    
    if not args.imgsize:
        imgsize=400
    else:
        imgsize=args.imgsize
    if not args.rows:
        rows=4
    else:
        rows=args.rows    
    if not args.columns:
        columns=2
    else:
        columns=args.columns
        
    if not args.bids_root:
        bids_root='bids_out'
    else:
        bids_root=args.bids_root
        
    if not op.exists(bids_root):
        parser.print_help()
        raise ValueError('specified BIDS directory %s does not exist' % bids_root)
    
    history_log = initialize(bids_root)
    deriv_root = op.join(bids_root, 'derivatives') 
    QA_root = op.join(deriv_root,PROJECT)
    subjects_dir = op.join(bids_root,'derivatives/freesurfer/subjects')
    
    image_list = glob.glob(op.join(QA_root,'*coreg.png'))
    sub_obj_list = [sub_qa_info(i, fname) for i,fname in enumerate(image_list)]
    
    #Update status based on previous log
    if history_log is not None:
        last_review = get_last_review(history_log) 
        stat_dict = build_status_dict(last_review)
        for sub_qa in sub_obj_list:
            if sub_qa.subject in stat_dict.keys():
                sub_qa.set_status(stat_dict[sub_qa.subject])
    
    run_gui(sub_obj_list,rows,columns,imgsize)                    
                           
                           
                           