# Prétraitement des données
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    # Imputation des valeurs manquantes
    df.fillna(method='ffill', inplace=True)

    # Encodage des variables catégorielles
    label_cols = df.select_dtypes(include='object').columns
    for col in label_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Normalisation des variables numériques
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    return df
