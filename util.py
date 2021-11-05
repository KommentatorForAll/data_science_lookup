from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd


def read_file(fileprefix: str = '') -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    df.to_csv('submission.csv', index=False)


def plot_columns(df: pd.DataFrame, transform_function=None):
    transform_function(df)
    df.plot.bar()
    plt.show()
