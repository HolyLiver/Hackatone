import numpy as np
import pandas as pd

#Service modules
import timeit
import inspect, os

#Import tools to work with data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing metrics 
from sklearn.metrics import roc_auc_score

# Importing ML models 
from catboost import CatBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier


class Dataset:

    def __init__(self, file_name) -> None:
        self.file_name = file_name
        self.download_path = get_dir_path(self.file_name)
        self.database = pd.read_csv(self.download_path)

    def x_data(self):
        return self.database.drop(["Y"], axis=1).copy()
    
    def y_data(self):
        return self.database["Y"].copy()

def get_dir_path(file_name):
    file_path = os.getcwd()
    dir_name = os.path.split(file_path)[0]
    return f"{dir_name}\\database\\{file_name}"

FCh_MoFP_path = get_dir_path("train_rdkit_morgan_smote.csv")
FCh_path = get_dir_path("train_rdkit_smote.csv")

df_train_FCh_descr = pd.read_csv(FCh_path)
df_train_FCh_MoFP_descr = pd.read_csv(FCh_MoFP_path, index_col=0)

x_FCh_MoFP = df_train_FCh_MoFP_descr.drop(["Y"], axis=1).copy()
y_FCh_MoFP = df_train_FCh_MoFP_descr["Y"].copy()

x_FCh = df_train_FCh_descr.drop(["Y"], axis=1).copy()
y_FCh = df_train_FCh_descr["Y"].copy()

FCh = Dataset("train_rdkit_smote.csv")
FCh_MoFP = Dataset()

datas = {"Phys-chem and Morgan desc": {"X": x_FCh_MoFP, "y": y_FCh_MoFP, "file": FCh_MoFP_path}, 
         "Phys-chem desc": {"X": x_FCh, "y": y_FCh, "file": FCh_path}}

models = {"XGBoost":XGBClassifier(random_state=42), 
          "GradientBoosting": GradientBoostingClassifier(random_state=42), 
          "RandomForest": RandomForestClassifier(random_state=42), 
          "ExtraTrees": ExtraTreesClassifier(), 
          "LightGBM": LGBMClassifier(verbosity=-1, random_state=42), 
          "CatBoost": CatBoostClassifier(silent=True, random_state=42)
        }

# results = list()
# count = 0
# data_num = len(datas.keys())
# model_num = len(models.keys())
# for data_name, X_y in datas.items():
#     scaler = StandardScaler()
#     X_y["X"] = pd.DataFrame(scaler.fit_transform(X_y["X"]), columns=X_y["X"].columns)
#     X_train, X_test, y_train, y_test = train_test_split(X_y["X"], X_y["y"], test_size=0.2, random_state=42)
#     model_res = list()
#     for name, model in models.items():
#         mod = model
#         start = timeit.default_timer()
#         mod.fit(X_train, y_train)
#         y_pred = model.predict_proba(X_test)[:, 1]
#         auc_roc = round(roc_auc_score(y_test, y_pred), 4)
#         stop = timeit.default_timer()
#         time = round(stop-start, 3)
#         model_res.extend([auc_roc, time])
#         count += 1
#         print(f" {count}/{data_num*model_num}. Finished model {name} with dataset {data_name}, AUC_ROC: {round(auc_roc, 5)}, spended time: {round(stop-start, 2)} sec")
#     results.append(model_res)