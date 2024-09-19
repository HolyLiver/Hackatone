import numpy as np
import pandas as pd

#Service modules
import timeit
import inspect, os
import ntpath

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

class Pipeline:
    
    used_models= {"XGBoost":XGBClassifier(random_state=42), 
          "GradientBoosting": GradientBoostingClassifier(random_state=42), 
          "RandomForest": RandomForestClassifier(random_state=42), 
          "ExtraTrees": ExtraTreesClassifier(), 
          "LightGBM": LGBMClassifier(verbosity=-1, random_state=42),
          "CatBoost": CatBoostClassifier(silent=True, random_state=42)
          }
    
    def __init__(self, logging=True) -> None:
        self.logging = logging
        self.used_db = dict()
    
    def get_best_conditions(self, paths, label_column="Y"):
        self.label_column = label_column
        self.used_models = self.used_models
        self.results = list()
        for path in paths:
            if self.__check_csv_from_path(path, label_column) == False:
                break
            file_name = ntpath.basename(path)
            df = self.__read_csv(path)
            self.used_db[file_name] = {"X": db_x(df, self.label_column), "Y": db_y(df, self.label_column), "file path": path}
            db_res = self.__db_model_corr(df_x=self.used_db[file_name]["X"], df_y=self.used_db[file_name]["Y"], file_name=file_name)
            self.results.append(db_res)
            
        self.df_results = self.__form_dataframe()
        self.best_params = self.__detect_best_params()
        if self.logging:
            print(f"Лучший результат {self.best_result}, на датасете: {self.best_params["Dataset"]}, на модели {self.best_params["Model"]}")
        return self.best_params
    
    def __read_csv(self, path):
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)
            if self.logging:
                print(f"Dropped 'Unnamed: 0' column")
        return df

    def __detect_best_params(self):
        file = self.df_results.stack().idxmax()[0]
        model = self.df_results.stack().idxmax()[1]
        self.best_result = self.df_results.stack().max(),
        return {"Model": model, "Dataset": self.used_db[file]["file path"]}

    def __form_dataframe(self):
        df = pd.DataFrame(self.results, columns=self.used_models.keys())
        df["File"] = self.used_db.keys()
        df.set_index("File", inplace=True)
        return df
    
    def show_table(self):
        pass

    def __db_model_corr(self, df_x, df_y, file_name):
        if self.logging:
            print(f"Started analysing models with file: {file_name}")
        scaler = StandardScaler()
        df_x_sc = pd.DataFrame(scaler.fit_transform(df_x), columns=df_x.columns)
        X_train, X_test, y_train, y_test = train_test_split(df_x_sc, df_y, test_size=0.2, random_state=42)
        models_res = list()
        model_num = len(self.used_models)
        count = 0
        for name, model in self.used_models.items():
            mod = model
            start = timeit.default_timer()
            mod.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            auc_roc = round(roc_auc_score(y_test, y_pred), 4)
            stop = timeit.default_timer()
            time = round(stop-start, 3)
            models_res.extend([auc_roc])
            count += 1
            if self.logging:
                print(f" {count}/{model_num}. Finished model {name} with dataset {file_name}, AUC_ROC: {round(auc_roc, 5)}, spended time: {round(stop-start, 2)} sec")
        return models_res


    def __check_csv_from_path(self, path, label_column):
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"{path} didn't found. File won't used further.")
            return False
        except Exception:
            print("Errors occuried. File won't used further.")
            return False
        
        try:
            df[label_column]
        except KeyError:
            print(f"Label column {label_column} in file {path} isn't detected. File won't used further.")
            return False
        
        return True


def db_x(df, lable_column):
    return df.drop([lable_column], axis=1).copy() 

def db_y(df, lable_column):
    return df[lable_column].copy()


if __name__ == "__main__":
    pipe = Pipeline()
    print(pipe.get_best_conditions(['c:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\database\\train_rdkit_morgan_smote.csv',
                                    'c:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\database\\train_rdkit_smote.csv']))