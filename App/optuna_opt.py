import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class Optuna_optimization:
    def __init__(self, label_column="Y", logging = True) -> None:
        self.label_column = label_column

    def get_best_model(self, features, file_path):
        self.features = features
        self.file_path = file_path
        df = pd.read_csv(file_path)
        self.df_y = df[self.label_column]
        self.df_x = df.drop(self.label_column, axis=1)[self.features]
        scaler = StandardScaler()
        self.df_x_sc = pd.DataFrame(scaler.fit_transform(self.df_x), columns=self.features)
        self.set_trials()
        return self.save_model()

    
    def CB_objective(self, trial):
        param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        'used_ram_limit': '8gb',
        'eval_metric': 'AUC',
        'logging_level': 'Silent'
    }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        clf = CatBoostClassifier(**param, early_stopping_rounds=20)
        scores = cross_val_score(clf, self.df_x_sc, self.df_y, cv=kf, scoring="roc_auc")
        print("aa")
        return np.mean(scores)

    def set_trials(self):
        study_1 = optuna.create_study(direction="maximize", study_name="CB regressor")
        func = lambda trial: self.CB_objective(trial)
        study_1.optimize(func, n_trials=3)
        self.best_params = study_1.best_params
    
    def save_model(self):
        model = CatBoostClassifier(**self.best_params, silent=True)
        X_train, X_test, y_train, y_test = train_test_split(self.df_x_sc, self.df_y, test_size=0.2,random_state=0)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)
        roc = roc_auc_score(y_test, y_pred_proba[:,1])
        print(roc)
        model.save_model(fname = "FH_descr_RFE_20_CV", format="cbm")
        return model
    

# if __name__ == "__main__":
#     file_path = "C:\\Users\\User\\OneDrive\\Документы\\Python\\Hackatone\\database\\train_rdkit_smote.csv"
#     features = ['property', 'SlogP_VSA8', 'fr_nitro', 'TPSA', 'SlogP_VSA10', 'fr_nitroso', 'VSA_EState3', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA1', 'SlogP_VSA3', 'VSA_EState4', 'BCUT2D_MWHI', 'SMR_VSA5', 'SMR_VSA10', 'SMR_VSA4', 'VSA_EState5', 'EState_VSA1', 'BCUT2D_LOGPLOW', 'fr_epoxide', 'EState_VSA2', 'MaxPartialCharge']
#     best_model = Optuna_optimization().get_best_model(file_path=file_path, features=features)
