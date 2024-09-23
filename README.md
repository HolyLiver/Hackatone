# Hackatone
ADMET prediction 


## App
The folder that contains the code submitted .py files.

- Mordred_descript_create.py - the file with the calculation of mordred descriptors.
- RDK_descript.py - the file with the calculation of morgan and rdkit descriptors.
- df_model_selection.py - the file with the selection of machine learning models.
- feature_selection.py - the file with the selection of descriptors.
- optuna_opt.py - A file with the selection of hyperparameters for the selected model.
- main.py - A file that runs all the code except for the calculation of descriptors.
- prediction_output.py - A file with a forecast of the target property for the target file. In our case, the target file is a test file with selected descriptors.

## Datasets
The folder containing the databases. Initial databases and databases that include descriptors.

## Final_model
A folder containing an impression of the final model and a table of values predicted on the test.

- FH_descr_RFE_32_CV - dataset with the selection of descriptors
- answer.csv - dataset with answers for the final/test dataset
- final_model - impression of the final model

## Requirements
The folder contains a file with versions of libraries and packages used in this project

## Tests
The folder containing code testing some functions.

- Test_of_descript.ipynb - the file containing the tests for functions: get_morgan and get_rdkit.

## .gitignore 
The file that allows you to ignore the environment when uploading a project to a local git.
