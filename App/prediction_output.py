from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import inspect, os


class Form_result():
    def __init__(self, test_path_file, train_path_file, features_list, model_path, label_column="Y") -> None:
        self.test_path_file = test_path_file
        self.train_path_file = train_path_file
        self.features_list = features_list
        self.model_path = model_path
        self.label_column = label_column
        

    def form_df_x(self):
        self.df_x = pd.read_csv(self.test_path_file)[self.features_list]
        self.train_df_x = pd.read_csv(self.train_path_file)[self.features_list]
        scaler = StandardScaler()
        scaler.fit(self.train_df_x)
        self.X_sc = scaler.transform(self.df_x)
        self.X_sc = pd.DataFrame(self.X_sc, columns=self.features_list)
        self.get_answer_df()
        return self.df_answer

    def get_answer_df(self):
        model = CatBoostClassifier()
        model.load_model(self.model_path)
        print("Model has been loaded")
        y_proba = model.predict_proba(self.X_sc)[:,1]
        self.df_answer = pd.DataFrame(y_proba, columns=[self.label_column])
        self.df_answer.to_csv("proba.csv")

# test_path = "C:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\database\\test_rdkit.csv"
# features = ['property', 'SlogP_VSA8', 'fr_nitro', 'TPSA', 'SlogP_VSA10', 'fr_nitroso', 'VSA_EState3', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA1', 'SlogP_VSA3', 'VSA_EState4', 'BCUT2D_MWHI', 'SMR_VSA5', 'SMR_VSA10', 'SMR_VSA4', 'VSA_EState5', 'EState_VSA1', 'BCUT2D_LOGPLOW', 'fr_epoxide', 'EState_VSA2', 'MaxPartialCharge']
# model = "C:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\Models\\FH_descr_RFE_20_CV"
# train_path_file = "C:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\database\\train_rdkit_smote.csv"
# df = Form_result(test_path_file=test_path, train_path_file=train_path_file, features_list=features, model_path=model).form_df_x()
