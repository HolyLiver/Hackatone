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


def get_rdkit(df):
    computed_descriptors = Chem.Descriptors.descList
    for descriptor in computed_descriptors:
        name = descriptor[0]
        df[name] = df["Drug"].apply(lambda x: descriptor[1](MolFromSmiles(x)))


def get_morgan(df, radius=2, nBits=1024):
    df['Morgan'] = df['Drug'].apply(
        lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), radius=radius, nBits=nBits))
    for i in range(nBits):
        df[f'Bit_{i}'] = df['Morgan'][0][i]

    df_new = df.drop(['Morgan'], axis=1)
    df_new = df_new.drop(['Drug'], axis=1)
    return df_new

dir_name = os.path.split(os.getcwd())[0]
train = pd.read_csv(f"{dir_name}\\Hackatone\\Datasets\\train_admet.csv", sep =",")
test = pd.read_csv(f"{dir_name}\\Hackatone\\Datasets\\test_data.csv", sep =",")
train = train.drop(["Unnamed: 0", "Drug_ID"], axis=1)
test = test.drop(["Unnamed: 0", "Drug_ID"], axis=1)

get_rdkit(train)
get_rdkit(test)

train['Y'].plot(kind='hist', bins=20, title='Y')
plt.gca().spines[['top', 'right',]].set_visible(False)

train=train.dropna(axis=0)
train=train.reset_index()
train=train.drop(['index'], axis=1)

train_rdkit_morgan = get_morgan(train)
test_rdkit_morgan = get_morgan(test)

oversample = SMOTE()
train_rdkit_morgan_X, train_rdkit_morgan_y = oversample.fit_resample(train_rdkit_morgan.drop(['Y'], axis=1), train_rdkit_morgan['Y'])
train_rdkit_morgan = pd.concat([train_rdkit_morgan_y, train_rdkit_morgan_X], axis=1)

train_rdkit_morgan['Y'].plot(kind='hist', bins=20, title='Y')
plt.gca().spines[['top', 'right',]].set_visible(False)

train_rdkit_morgan.to_csv(f"{dir_name}\\Hackatone\\Datasets\\train_rdkit_morgan_smote.csv", index = False)
test_rdkit_morgan.to_csv(f"{dir_name}\\Hackatone\\Datasets\\test_rdkit_morgan.csv", index = False)

train_rdkit=train_rdkit_morgan.loc[:,"Y": "fr_urea"]
test_rdkit=test_rdkit_morgan.loc[:,"property": "fr_urea"]
train_morgan=pd.concat([train_rdkit_morgan.loc[:,"Y": "property"], train_rdkit_morgan.loc[:,"Bit_0": "Bit_1023"]], axis=1)
test_morgan=pd.concat([test_rdkit_morgan['property'], test_rdkit_morgan.loc[:,"Bit_0": "Bit_1023"]], axis=1)

train_morgan.to_csv(f"{dir_name}\\Hackatone\\Datasets\\train_morgan_smote.csv", index = False)
test_morgan.to_csv(f"{dir_name}\\Hackatone\\Datasets\\test_morgan.csv", index = False)
train_rdkit.to_csv(f"{dir_name}\\Hackatone\\Datasets\\train_rdkit_smote.csv", index = False)
test_rdkit.to_csv(f"{dir_name}\\Hackatone\\Datasets\\test_rdkit.csv", index = False)
