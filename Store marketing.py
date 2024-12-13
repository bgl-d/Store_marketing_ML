import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder


def data_observation(df):
    print(df.describe())
    print(df.dtypes)
    print(df.shape)
    df.hist(dataset.columns, bins=50)


if __name__ == '__main__':
    # settings of the terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 400)
    np.set_printoptions(linewidth=400)

    # read data
    dataset = pd.read_csv('../Store marketing/superstore_data.csv')

    # data observation
    data_observation(dataset)

    # missing values
    cols_with_missing = [col for col in dataset.columns if dataset[col].isnull().any()]
    for col in cols_with_missing:
        if dataset[col].dtypes == 'float64' or dataset[col].dtypes == 'int64':
            dataset[col] = dataset[col].fillna(dataset[col].median())

    # duplicates
    dataset.drop_duplicates()

    # date column to datetime
    dataset['Dt_Customer'] = pd.to_datetime(dataset['Dt_Customer'])

    # ordinal encoding
    object_cols = dataset.select_dtypes(include=['object']).columns
    dataset.loc[dataset['Marital_Status'] == 'YOLO', 'Marital_Status'] = 'NaN'
    dataset.loc[dataset['Marital_Status'] == 'Alone', 'Marital_Status'] = 'NaN'
    dataset.loc[dataset['Marital_Status'] == 'Absurd', 'Marital_Status'] = 'NaN'
    ordinal_encoder = OrdinalEncoder()
    dataset[object_cols] = ordinal_encoder.fit_transform(dataset[object_cols])

    # correlation matrix
    corr_m = dataset.corr()
    print(corr_m["Response"])
