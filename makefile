
#>>>> https://stackoverflow.com/questions/53382383/makefile-cant-use-conda-activate
# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate 
# <<<<

install:
	conda install -c conda-forge mne==0.23 -n enigma_meg_prep
	pip install wget
	pip install git+https://github.com/nih-megcore/nih_to_mne

	
install_test:
	conda install mamba -y -n base
	mamba create -n enigma_prep_test -c conda-forge mne==0.23 datalad -y
	($(CONDA_ACTIVATE) enigma_prep_test ; pip install -r ./requirements.txt)
	

install_system_requirements:
	dnf install Xvfb -y
	dnf install git git-annex -y

test:
	($(CONDA_ACTIVATE) enigma_prep_test ; xvfb-run -a pytest -s )
