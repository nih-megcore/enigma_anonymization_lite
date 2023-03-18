#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:33:04 2023

@author: nugenta
"""
import pandas as pd 
import os, os.path as op 
import logging
import argparse

import matplotlib
matplotlib.use('Qt5agg'); 

logger = logging.getLogger()
os.environ['MNE_3D_OPTION_ANTIALIAS']='false'

from enigma_preupload.enigma_anonymization_lite_functions import initialize, stage_mris, parallel_make_scalp_surfaces, process_meg_bids, process_mri_bids, loop_QA_report

if __name__=='__main__':

    # parse the arguments and initialize variables   

    parser = argparse.ArgumentParser()
    parser.add_argument('-topdir', help='''The base directory for outputs, defaults to current directory''')
    parser.add_argument('-csvfile', help='''The name of the csv file with paths and filenames''')
    parser.add_argument('-njobs',help='''The number of jobs for parallel processing, defaults to one''', type=int)
    parser.add_argument('-linefreq',help='''The linefrequency for the country of data collection''', type=int)
    parser.description='''This python script implements BIDSification of a dataset. We hope you enjoy using it as much as we enjoyed making it.'''
    
    args = parser.parse_args()
    
    if not args.njobs:
        njobs=1
    else:
        njobs=args.njobs
                   
    if args.topdir:
        topdir=args.topdir
    else:
        topdir=os.getcwd()
        
    if not args.csvfile:
        print('''-csvfile is a required argument''')
        raise ValueError
    else:
        csvfile=args.csvfile
        csvpath = op.join(topdir, csvfile)
        
    if not args.linefreq:
        print('''setting linefreq to 60, hope that's okay!''')
        linefreq=60
    else:
        linefreq=args.linefreq
        
    initialize(topdir=topdir)

    
    
