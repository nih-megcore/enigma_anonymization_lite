
#>>>> https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate 
# <<<<

install_test:
	#conda install --channel=conda-forge --name=base mamba -y
	conda env remove -n enigma_meg_test
	mamba create --override-channels --channel=conda-forge --name=enigma_meg_test mne pip -y
	($(CONDA_ACTIVATE) enigma_meg_test ; pip install -e .; pip install pytest pytest-reportlog )
	git submodule init
	git pull --recurse-submodules

install_headless_test:
	#conda env remove -n enigma_meg_test
	mamba create --override-channels --channel=conda-forge --name=enigma_meg_test mne pip "vtk>=9.2=*osmesa*" "mesalib=21.2.5" -y
	($(CONDA_ACTIVATE) enigma_meg_test ; pip install -e .; pip install pytest  )
	git submodule init
	git pull --recurse-submodules

install_system_requirements:
	dnf install Xvfb -y
	dnf install git git-annex -y

test:
	($(CONDA_ACTIVATE) enigma_meg_test ; cd enigma_anonymization_lite; pytest -vv --report-log=/tmp/enigma_anonymization_test_logfile.txt )  

test_headless:
	export FREESURFER_HOME=/opt/freesurfer/freesurfer_7_4_1/
	source /opt/freesurfer/freesurfer_7_4_1/SetUpFreeSurfer.sh
	($(CONDA_ACTIVATE) enigma_meg_test ; cd enigma_anonymization_lite; xvfb-run -a pytest -vv )


test_iterate_fs:
	($(CONDA_ACTIVATE) enigma_meg_test ; cd enigma_anonymization_lite; pytest -vv --report-log=./test_logfile.txt )  #xvfb-run -a pytest -s )

