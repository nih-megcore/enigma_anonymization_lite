#!/bin/bash

if [ ! -d ../BIDS_anon ] ;
then
	mkdir ../BIDS_anon
fi

if [ ! -d ../containers ];
then
	mkdir ../containers
fi

if [ ! -d ../routines ];
then
	mkdir ../routines
fi

enigma_prep_container=../containers/enigma_prep.sif

echo "Admin priveledges are required to build the singularity container:"
sudo singularity build ../containers/enigma_prep.sif  singularity_build.def



echo "Downloading defacing code and templates from freesurfer"
#echo "singularity run wget https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/face.gca.gz"
singularity exec -B $(pwd):$(pwd) ${enigma_prep_container} wget https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/face.gca.gz
singularity exec -B $(pwd):$(pwd) ${enigma_prep_container} wget https://surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/talairach_mixed_with_skull.gca.gz

echo "Unzipping defacing templates"
singularity exec -B $(pwd):$(pwd) ${enigma_prep_container} gunzip face.gca.gz
singularity exec -B $(pwd):$(pwd) ${enigma_prep_container} gunzip talairach_mixed_with_skull.gca.gz



### STOP
#singularity exec ${enigma_prep_container} mri_deface 
