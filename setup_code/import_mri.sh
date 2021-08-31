#!/bin/bash

mri=$1
subjid=$2

recon-all -i ${mri} -s ${subjid}
recon-all -autorecon1 -noskullstrip -s ${subjid}
mkheadsurf -s ${subjid}
if [ ! -d ${SUBJECTS_DIR}/${subjid}/bem ]; 
then mkdir ${SUBJECTS_DIR}/${subjid}/bem
fi

if [ ! -f ${SUBJECTS_DIR}/${subjid}/bem/outer_skin.surf ];
then 
	ln -s ${SUBJECTS_DIR}/${subjid}/surf/lh.seghead ${SUBJECTS_DIR}/${subjid}/bem/outer_skin.surf
fi

ipython -c "import matplotlib; 
matplotlib.use('Qt5agg'); 
import matplotlib.pyplot as plt; 
import mne; 
plt.ion(); 
tmp=mne.viz.plot_alignment(subject='${subjid}');
test=input('Hit enter to close:');
"
