#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:32:33 2023

@author: nugenta
"""


import os.path as op
import argparse
import glob
import pandas as pd
from enigma_anonymization_lite.enigma_anon_QA_functions import initialize, get_last_review, build_status_dict, get_subject_status

PROJECT = 'BIDS_ANON_QA'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-bids_root', help='''Location of bids directory, default=bids_out''')
    
    args = parser.parse_args()

    if not args.bids_root:
        bids_root='bids_out'
    else:
        bids_root=args.bids_root
    if not op.exists(bids_root):
        parser.print_help()
        raise ValueError('specified BIDS directory %s does not exist' % bids_root)
        
    deriv_root = op.join(bids_root, 'derivatives') 
    QA_root = op.join(deriv_root,PROJECT)
    
    history_log = []
    
    log = initialize(bids_root)
    last = get_last_review(log)
            
    data_dict = {}        
    stat_dict = build_status_dict(last)
        
    for row in last:
        
        subject, status = get_subject_status(row)
        data_dict[subject] = status
        
    data_frame = pd.DataFrame.from_dict(data_dict, orient='index')    
    dataframe_path = op.join(QA_root,'QA_summary.csv')
    data_frame.to_csv(dataframe_path)