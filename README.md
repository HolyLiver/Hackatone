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

- test_data.csv - contains information about smiles in the "Drug" column, some additional property - "Property"
- train_admet.csv - contains information about smiles in the "Drug" column, the predicted property - "Y", some additional property - "Property"
- test_rdkit.csv, test_morgan.csv, test_rdkit_morgan.csv - test datasets containing descriptors rdkit, morgan, rdkit and morgan respectively
- test_mordred_desc795_date20_09_time12.csv - test datasets containing mordred descriptors (795 of them, in .int64 and .float64 format)
- train_rdkit_smote.csv, train_morgan_smote.csv - train datasets containing descriptors rdkit, morgan respectively, datasets are normalized using smote

Two datasets cannot be placed here because of their size, so links to a Google drive with them are given.

- train_rdkit_morgan_smote.csv - train datasets containing descriptors rdkit and morgan datasets are normalized using smote https://drive.google.com/file/d/1-I107JsnyBNHhkHgPNNZ4LYuZkBefV0u/view?usp=sharing
- test_mordred_desc795_date20_09_time12.csv - mordred (795 of them, in .int64 and .float64 format) datasets are normalized using smote https://drive.google.com/file/d/10c63j1J6HCbTy652OGw3YBlPdVCvrUBw/view?usp=sharing

## Final_model
A folder containing an impression of the final model and a table of values predicted on the test.

- FH_descr_RFE_CV - dataset with the selection of descriptors
- answer.csv - dataset with answers for the final/test dataset
- final_model - impression of the final model

## Requirements
The folder contains a file with versions of libraries and packages used in this project

## Tests
The folder containing code testing some functions.

- Test_of_descript.ipynb - the file containing the tests for functions: get_morgan and get_rdkit.

## .gitignore 
The file that allows you to ignore the environment when uploading a project to a local git.
