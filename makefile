install:
	conda install -c conda-forge mne -n enigma_meg_prep
	pip install wget
	pip install git+https://github.com/nih-megcore/nih_to_mne

	
install_test:
	conda activate base
	conda install mamba -y 
	mamba create -n enigma_prep_test -c conda-forge mne -y
	mamba activate enigma_prep_test
	pip install wget datalad pytest 
	pip install git+https://github.com/nih-megcore/nih_to_mne
	pip install git+https://github.com/jstout211/enigma_MEG

install_system_requirements:
	dnf install Xvfb -y


