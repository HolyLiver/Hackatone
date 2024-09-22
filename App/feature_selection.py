import numpy as np
import pandas as pd

#Import tools to work with data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Importing metrics
from sklearn.metrics import roc_auc_score

# Importing ML models
import lightgbm as lgb

#Import tools for graphs
import matplotlib.pyplot as plt


class FeatureSelection:
    def __init__(self) -> None:
        pass
        
    def get_opt_features_number(self,file_path, search_step_percent=5, label_column="Y", logging=True):
        self.file_path = file_path
        self.label_column = label_column
        self.__download_df()
        self.search_step_percent = search_step_percent
        self.importance_rating = self.get_feature_importance()
        x_length  = len(self.df_x.columns)
        step = int(x_length * self.search_step_percent /100)
        result_1 = self.features_drop_cycle(0, len(self.df_x.columns), step)
        plato_point_1 =self.opt_number_determination(result_1)
        result_2 = self.features_drop_cycle(plato_point_1-step, plato_point_1+step+1, 1)
        plato_point_2 =self.opt_number_determination(result_2)
        print(plato_point_2)
        return self.importance_df["feature_name"].tolist()[:plato_point_2]
    
    def __download_df(self):
        self.df = pd.read_csv(self.file_path)
        sc = StandardScaler()
        self.df_x = self.df.drop([self.label_column], axis=1).copy()
        self.df_y = self.df[self.label_column].copy()
        col = self.df_x.columns
        self.df_x = pd.DataFrame(sc.fit_transform(self.df_x), columns=col)
    
    def opt_number_determination(self, result, window_size=5):
        y = result["AUC-ROC"]
        x = result["n_feature"]
        y_rolling_mean = np.convolve(y, np.ones(window_size)/window_size, mode='valid')

        # Определение точки, где график выходит на плато
        plateau_threshold = 0.99* np.max(y_rolling_mean)
        plateau_idx = np.where(y_rolling_mean >= plateau_threshold)[0][0] + window_size//2
        plateau_x = x[plateau_idx]
        print(plateau_idx, plateau_x)
        return plateau_x

    def features_drop_cycle(self, begin, end, step):
        result = {"n_feature":[],"AUC-ROC":[], "features":[]}
        for n in range(begin,end, step):
            try:
                model = lgb.LGBMClassifier(importance_type='gain', verbosity=-1)
                columns = self.importance_df["feature_name"].tolist()[:n+1]
                # print(columns)
                X = self.df_x[columns]
                X_train, X_test, y_train, y_test = train_test_split(X, self.df_y, test_size=0.2, random_state=10)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                result["n_feature"].append(n)
                result["features"].append(X.columns)
                result["AUC-ROC"].append(roc_auc)
                # print(f"Calcualted {n}")
            except (ValueError, IndexError) as e:
                pass
        if end//step != 0:
            model = lgb.LGBMClassifier(importance_type='gain', verbosity=-1)
            columns = self.importance_df["feature_name"].tolist()[:end+2]
            X = self.df_x[columns]
            X_train, X_test, y_train, y_test = train_test_split(X, self.df_y, test_size=0.2, random_state=10)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            result["n_feature"].append(end)
            result["features"].append(self.importance_df["feature_name"].tolist())
            result["AUC-ROC"].append(roc_auc)
        print(result["AUC-ROC"])
        return result
        


    def get_feature_importance(self):
        X_train, X_test, y_train, y_test = train_test_split(self.df_x, self.df_y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier(importance_type='gain', verbosity=-1)
        model.fit(X_train, y_train)
        importance = model.feature_importances_
        self.importance_df = (
            pd.DataFrame({
                'feature_name': self.df_x.columns,
                'importance_gain': importance
            })
            .sort_values('importance_gain', ascending=False)
            .reset_index(drop=True)
        )

# if __name__ == "main":
#     path_for_dataset = "C:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\Datasets\\train_rdkit_smote.csv"
#     f_selection = FeatureSelection(path_for_dataset)
#     aaa = f_selection.get_opt_features_number()
#     print(aaa)