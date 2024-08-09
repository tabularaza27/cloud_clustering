# Cirrus formation regimes - Data driven identification and quantification of mineral dust effect

This repo contains the code for replicating the research conducted in "Cirrus formation regimes - Data driven identification and quantification of mineral dust effect" by Jeggle et al. If you have any question please reach out to me (kai.jeggle@env.ethz.ch)

## Structure

├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── Analysis_and_Plotting.ipynb
    ├── Clustering.ipynb
    └── functions
        ├── clustering_functions.py
        ├── helper_functions.py
        ├── plotting_functions.py
        ├── preproc_functions.py

The code to train time series clustering models as described in the paper are located in `src/Clustering.ipynb`. The notebook `src/Analysis_and_Plotting.ipynb` contains code for data analysis and creation of the paper figures. Functions used in the notebooks are in the respective files in `src/functions`.

The data can be downloaded from [Zenodo]()

To create a conda environment containing all required packages run `$ conda create --name <env> --file requirements.txt`
