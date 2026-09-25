### USAGE TUTORIAL with CONDA (miniconda) in Windows

This tutorial shows how to install, run and update the tools in this repository. 

#### Download

##### Requirements
1. Install GIT: https://git-scm.com/download/win
2. Install MINICONDA: https://docs.conda.io/en/latest/miniconda.html#windows-installers

##### Installation
1. Copy the content of the file `tlag_env_windows.yml` located in this folder into a new document in your PC with the same name and extension.
2. Install the environment with conda:
    1. Open the Anaconda Prompt (search anaconda in the windows Start Menu)
    2. In the terminal, navigate to the folder that contains the environment file using the command `cd <path to folder>`.
    3. In the terminal, type `conda env create --file=tlag_env_windows.yml`


#### Running the tools

1. Activate the environment:
    1. Open the Anaconda Prompt
    2. Type `conda activate TLAG-analysis`.
2. Run the tools:
    1. In the terminal, type `visualizer` or `peak_fitting --help`.

#### Updating the repository

1. Activate the environment:
    1. Open the Anaconda Prompt
    2. Type `conda activate TLAG-analysis`.

2. In the terminal, type `pip install --upgrade git+https://github.com/Jordi-aguilar/TLAGToolkit@master`. In case you don't want to install the dependencies (for example with the debian version) add the flag `--no-deps` at the end. In case you want a different branch, like `peak_area`, replace `@master` by `@peak_area`.
