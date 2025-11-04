"""
Loads the UCI Breast Cancer Wisconsin dataset using sklearn's built-in dataset loader.
Handles exceptions if data retrieval fails and converts it to a pandas DataFrame.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    try:
        dataset = load_breast_cancer()
        df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        df['target'] = dataset.target
        return df, dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    df, dataset = load_data()
    print("Data shape:", df.shape)
    print(df.head())
