install:
	conda install -c conda-forge mne
	pip install wget
	pip install git+https://github.com/nih-megcore/nih_to_mne

	
install_test:
	conda create -n enigma_prep_test -c conda-forge mne
	conda activate enigma_prep_test
	pip install wget datalad pytest
	pip install git+https://github.com/nih-megcore/nih_to_mne

install_system_requirements:
	dnf install Xvfb -y


