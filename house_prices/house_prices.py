from typing import List, Tuple, Set, Union

import pandas as pd
from pandas import Index
from sklearn.impute import SimpleImputer, KNNImputer
# This is just for Typing. I don't acutally use this Class anywhere
from sklearn.impute._base import _BaseImputer

from util import read_file, write_prediction, plot_columns

imputers_to_look_at = [
    SimpleImputer,
    KNNImputer,
]


def get_na_to_drop(X: pd.DataFrame, threshold: float = 20) -> List[str]:
    """
    Selects and returns all Columns whose missing percentages is greater than the threshold
    :param X: The Dataframe to get the columns from
    :param threshold: at what point a column is counted as unusable
    :return: A list of columns which are unusable due to missing data

    """
    percent_missing = X.isnull().mean() * 100
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    return missing_value_df.loc[missing_value_df['percent_missing'] > threshold].index


def split_obj_cols_by_cardinality(obj_cols: pd.DataFrame, threshold: int = 10) -> Tuple[Set[str], Set[str]]:
    """
    Splits the given object columns into low and high cardinality lists.
    :param obj_cols: all columns to look at
    :param threshold: the point at which the cardinality is counted as high
    :return: A tuple of low and high cardinality columns
    """
    full_set = set(obj_cols.columns)
    low_cardinality_columns = set([col for col in obj_cols.columns if obj_cols[col].nunique() < threshold])
    high_cardinality_columns = full_set - low_cardinality_columns
    return low_cardinality_columns, high_cardinality_columns


def fill_na(X: pd.DataFrame, strategies: List[Tuple[Union[List[str], Index, Set[str]], _BaseImputer]]) -> pd.DataFrame:
    """
    Fills up all NaN values of the dataframe
    :param X: The dataframe to fill the na of
    :param strategies: what strategy to use for which columns
    :return:
    """
    df = X

    for columns, imputer in strategies:
        impuned = pd.DataFrame(imputer.fit_transform(X[columns]))
        impuned.columns = columns
        df[impuned.columns] = impuned
    return df


def get_cols_to_drop_for_ordinal_encoder(
        high_cardinality_columns: Set[str],
        train_X: pd.DataFrame,
        val_X: pd.DataFrame
) -> Set[str]:
    """
    Returns a set of columns which have to be dropped in order for the ordinal encoder to work.
    :param high_cardinality_columns: A Set of columns which can potentially used for ordinal encoding
    :param train_X: the train set
    :param val_X: the validation set
    :return: The set of columns which have to be dropped
    """
    return high_cardinality_columns - set([col for col in high_cardinality_columns if set(val_X[col]).issubset(train_X[col])])


def transform_dataset(X: pd.DataFrame, test_X: pd.DataFrame) -> pd.DataFrame:
    """
    transforms the given dataset
    :param X: the dataset to transform
    :param test_X: The data to predict (used to check for ordinal usage)
    :return: the transformed dataset
    """
    cols_to_drop: List[str] = get_na_to_drop(X)
    X = X.drop(cols_to_drop, axis=1)

    obj_cols: pd.DataFrame = X.select_dtypes('object')
    num_cols: pd.DataFrame = X.select_dtypes(['float64', 'int64', 'float32', 'int32'])
    low_cardinality_columns, high_cardinality_columns = split_obj_cols_by_cardinality(obj_cols)

    X = fill_na(X, [
        (num_cols.columns, SimpleImputer(strategy='mean')),
        (low_cardinality_columns, SimpleImputer(strategy='most_frequent')),  # KNN might be quite useful here
        (high_cardinality_columns, SimpleImputer(strategy='most_frequent')),  # Same over here
    ])

    get_cols_to_drop_for_ordinal_encoder()

    return X


def generate_model(X: pd.DataFrame, y: pd.Series):
    return None


def get_top_10_missing_percent(x: pd.DataFrame) -> pd.DataFrame:
    """
    Transformation function to get the percentage of NaN values for each column. This selects the top 10 of them
    :param x: The dataframe to transform
    :return: The top 10 columns by their NaN percentage
    """
    percent_missing = x.isnull().mean() * 100
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})

    missing_value_df.sort_values('percent_missing', inplace=True, ascending=False)
    return missing_value_df.head(10)


def main():
    """
    Loads the dataset analyses it and writes the predictions into a file
    :return:
    """

    train_data, test_data = read_file()
    X: pd.DataFrame = train_data
    # Setting y to the SalePrice as that is the value we are trying to predict
    y = X.pop('SalePrice')

    X = transform_dataset(X)

    test_X = transform_dataset(test_data)

    # plot_columns(X, get_top_10_missing_percent)

    model = generate_model(X, y)
    predictions = None
    # predictions = model.predict(test_X)
    write_prediction(predictions)

    # print(f"obj_cols: {obj_cols}")


if __name__ == '__main__':
    main()
