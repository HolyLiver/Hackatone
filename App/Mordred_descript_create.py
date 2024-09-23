
! pip install rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles
import matplotlib.pyplot as plt
import seaborn as sns
! pip install mordred
from mordred import Calculator, descriptors
import imblearn
from imblearn.over_sampling import SMOTE
import os

"""Расчет дескрипторов Mordred"""

def calc_mordred(df):
    calc = Calculator(descriptors, ignore_3D=False)

    mols = df['Drug'].apply(lambda mol: Chem.MolFromSmiles(mol))
    mordred = calc.pandas(mols)
    mordred_desc = mordred.columns.values.tolist()

    numeric_columns = []
    for col in mordred.columns:
        if mordred[col].dtype in ('int64', 'float64'):
            numeric_columns.append(col)

    for descriptor in numeric_columns:
        df[descriptor] = mordred[descriptor]

    df_new = df.drop(['Drug'], axis = 1)

    return df_new

def get_mordred(df, other_df = None):

    if other_df is None:
        df_mordred = calc_mordred(df)

        return df_mordred

    else:
        df_ = pd.concat([df.drop(['Y'], axis=1), other_df], axis=0)
        df_ = df_.reset_index()
        df_ = df_.drop(['index'], axis=1)

        df_mor = calc_mordred(df_)

        df_mor_train = pd.concat([df['Y'], df_mor[:len(df)]], axis = 1)

        df_mor_test = df_mor[len(df):]
        df_mor_test = df_mor_test.reset_index()
        df_mor_test = df_mor_test.drop(['index'], axis=1)

        return df_mor_train, df_mor_test



train = pd.read_csv("/content/drive/MyDrive/ADMET/train_admet.csv", sep =",")
test = pd.read_csv("/content/drive/MyDrive/ADMET/test_admet.csv", sep =",")
train = train.drop(["Unnamed: 0", "Drug_ID"], axis=1)
test = test.drop(["Unnamed: 0", "Drug_ID"], axis=1)

dir_name = os.path.split(os.getcwd())[0]
df_paths = [f"{dir_name}\\Hackatone\\Datasets\\train_rdkit_smote.csv",
            f"{dir_name}\\Hackatone\\Datasets\\train_morgan_smote.csv"]
train_mordred, test_mordred = get_mordred(train, test)

oversample = SMOTE()
train_mordred_X, train_mordred_y = oversample.fit_resample(train_mordred.drop(['Y'], axis=1), train_mordred['Y'])
train_mordred_smote = pd.concat([train_mordred_y, train_mordred_X], axis=1)

train_mordred_smote.to_csv("/content/drive/MyDrive/ADMET/train_mordred_desc795_smote_date22_09.csv", index = False)
test_mordred.to_csv("/content/drive/MyDrive/ADMET/test_mordred_desc795_date22_09.csv", index = False)