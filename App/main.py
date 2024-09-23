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
best_file = res["Dataset"]

features_list = FeatureSelection().get_opt_features_number(best_file)

model = Optuna_optimization().get_best_model(file_path=best_file, features=features_list, trials=1)

df_answer = Form_result(test_path_file=f"{dir_name}\\Hackatone\\Datasets\\test_rdkit_morgan.csv", 
                        train_path_file=best_file, 
                        model_path=f"{dir_name}\\Hackatone\\Final_model\\final_model",
                        features_list=features_list).form_df_x()