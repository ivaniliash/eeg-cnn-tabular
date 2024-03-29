# Pain Classification on EEG and Tabular Data
University project at the Lab for AI in Medicine at TU Munich with Felix Hsieh, Carlos Llano and Ivan Iliash (me).
Winter Semester 22/23.

## Project structure

    ├── docs          # Documentation files
    ├── framework     # Training framework code
    ├── notebooks     # Notebooks for exploratory data analysis
    └── README.md

## Data
EEG recordings and Tabular data (demographic information, disease relevant information) from healthy subjects and patients
with chronic pain. The follwing data files were provided by the _University Hospital rechts der Isar_. Unfortunately the datasets can't be shared with the public.
- data_Dinh_fs200.pt
- data_Heitmann_fs200.pt
- labels_Dinh_fs200.pt
- labels_Heitmann_fs200.pt
- tabular_Dinh.xlsx
- tabular_Heitmann.xlsx

The EEG data is provided as batches of torch tensors in the `.pt` files.

For more details, see notebook [EDA.ipynb](notebooks/EDA.ipynb).

## Prerequisites
Create a python environment and install dependencies.  
With virtualenv (https://virtualenvwrapper.readthedocs.io/en/latest/)
```
mkvirtualenv eeg-multimodal
pip install -r requirements.txt
```
With conda
```
conda env create -f environment.yml
```
   
## Run training
1. Configure training changing values in _framework/runner/config.json_. Follow convention in _config-structure.json_.
2. Execute:
   ```
   python framework/runner/runner.py
   python framework/runner/runner_crossval.py
   ```
   Training results will be saved in a folder named _training_results/_


## Conclusions

* Theta and Gamma are important frequencies bands for pain classification.
* Tabular data use shows no improvements, which requires further exploration with complete datasets or additional features.
* High confidence for Fibromyalgia classification caused by clear patterns in the EEG data (See [strong oscillations](docs/avg-eeg-all-channels.png)).
* Signal data augmentations applied to EEG can help when dealing with a small dataset.

For more details, see [poster.pdf](docs/poster.pdf). 
