import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def read_datasets(fileprefix: str = './assets/input/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the Train and Test data and returns it
    :param fileprefix: the file name prefix
    :return: The train and the test dataset
    """
    train_data = pd.read_csv(fileprefix+'train.csv')
    test_data = pd.read_csv(fileprefix+'test.csv')
    return train_data, test_data


def write_prediction(df: pd.DataFrame) -> None:
    df.Id = df['Id'].astype('int32')
    print(df.dtypes)
    os.makedirs(f'./assets/output', exist_ok=True)
    df.to_csv(f'./assets/output/{time.time()}-submission.csv', index=False)


def plot_columns(df: pd.DataFrame, transform_function=None):
    transform_function(df)
    df.plot.bar()
    plt.show()


def get_ordinal_columns(df: pd.DataFrame):
    vals = {'Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'}
    df = df.fillna('None')
    cols = [col for col in df.select_dtypes('object').columns if set(df[col].unique()).issubset(vals)]
    return cols
