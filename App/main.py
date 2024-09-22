import os
from feature_selection import FeatureSelection
from df_model_selection import ModelCorrelation
from optuna_opt import Optuna_optimization
from prediction_output import Form_result
import pandas as pd

dir_name = os.path.split(os.getcwd())[0]
df_paths = [f"{dir_name}\\Hackatone\\Datasets\\train_rdkit_smote.csv",
            f"{dir_name}\\Hackatone\\Datasets\\train_morgan_smote.csv"]

res = ModelCorrelation().get_best_conditions(df_paths)
print(res["Dataset"])

features_list = FeatureSelection().get_opt_features_number(res["Dataset"])


