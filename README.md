# A Comparative Machine Learning Study of Connectivity-Based Biomarkers of Schizophrenia

## Preprocessing
BOLD fMRI preprocessing scripts are in `SCZ/preprocessing`. Post fMRIPrep preprocessing requires [this package](https://github.com/victoris93/NeuroConn).

## Gradient Dispersion
All script used to compute neighborhood and centroid dispersion are in `Dispersion`.
## Data Preparation
All features need to be vectorized and concatenated in one array.  `/SCZ/modeling/aggregate_features.py` does that provided you have a dictionary of paths with preprocessed data.

## Permutation Feature Importance
The pipeline described in Figure 4 of the manuscript is implemented in `SCZ/modeling/lr_pipeline.py`. 

## Multi-Classifier Analyses
All scripts related to the assessment of classification performance are in  `SCZ/modeling`.

All helper functions used in the stages mentioned above are in `SCZ/modeling_utils.py`