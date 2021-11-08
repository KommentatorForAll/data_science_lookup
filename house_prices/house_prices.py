import time
from typing import List, Tuple, Set, Union, Iterable, Any

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from pandas import Index
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.impute import SimpleImputer, KNNImputer
# This is just for Typing. I don't acutally use this Class anywhere
from sklearn.impute._base import _BaseImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing._encoders import _BaseEncoder, OneHotEncoder, OrdinalEncoder

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
        try:
            # this throws an error, if either the transform function is not implemented (it is for SimpleImputer)
            # or if the the transform wasn't fit yet.
            # That way the transform function is called for the test dataset,
            # because the imputer was fit with the train set.
            impuned = pd.DataFrame(imputer.transform(X[columns]))
        except:
            impuned = pd.DataFrame(imputer.fit_transform(X[columns]))
        impuned.columns = columns
        df[impuned.columns] = impuned
    return df


def encode_columns(
        X: pd.DataFrame,
        num_cols: Iterable[str],
        strategies: List[Tuple[Union[List[str], Index, Set[str]], _BaseEncoder]]
) -> pd.DataFrame:
    """
    Encodes all given columns using the defined strategy
    :param X: The dataframe to encode
    :param strategies: which strategy should be used for which columns
    :return: The encoded dataframe
    """
    df = X[num_cols]
    for columns, encoder in strategies:
        try:
            encoded = pd.DataFrame(encoder.transform(X[columns]))
        except:
            encoded = pd.DataFrame(encoder.fit_transform(X[columns]))
        encoded.index = X.index
        if len(columns) == len(encoded.columns):
            encoded.columns = columns
        else:
            encoded.columns = [str(type(encoder)).split('.')[-1].split('\'')[0] + str(col) for col in encoded.columns]
        df = pd.concat([df, encoded], axis=1)
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
    return high_cardinality_columns - set(
        [col for col in high_cardinality_columns if set(val_X[col]).issubset(train_X[col])])


def cluster_data(X: pd.DataFrame,
                 test_x: pd.DataFrame,
                 clusters: List[Tuple[List[str], ClusterMixin]]
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clusters the given DataFrame for each cluster using KMeans
    :param X: The dataframe to fit_transform
    :param test_x: The dataframe to transform
    :param clusters: the columns to cluster after and the algorithm to use
    :return: The transformed dataframes
    """
    for i, cluster in enumerate(clusters):
        col_name = f'cluster' + str(i)
        X[col_name] = cluster[1].fit_predict(X[cluster[0]])
        # X[col_name] = X[col_name].astype("category")
        try:
            test_x[col_name] = cluster[1].predict(test_x[cluster[0]])
        except:
            test_x[col_name] = cluster[1].fit_predict(test_x[cluster[0]])
        # test_x[col_name] = test_x[col_name].astype('category')
    return X, test_x


def transform_dataset(X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    transforms the given dataset
    :param X: the dataset to transform
    :param test_X: The data to predict (used to check for ordinal usage)
    :return: the transformed dataset
    """
    cols_to_drop: Iterable[str] = get_na_to_drop(X)
    X = X.drop(cols_to_drop, axis=1)

    obj_cols: pd.DataFrame = X.select_dtypes('object')
    num_cols: pd.DataFrame = X.select_dtypes(['float64', 'int64', 'float32', 'int32'])
    low_cardinality_columns, high_cardinality_columns = split_obj_cols_by_cardinality(obj_cols, 1000)

    strategies = [
        (num_cols.columns, SimpleImputer(strategy='mean')),
        (low_cardinality_columns, SimpleImputer(strategy='most_frequent')),  # KNN might be quite useful here
        # (high_cardinality_columns, SimpleImputer(strategy='most_frequent')),  # Same over here
    ]

    X = fill_na(X, strategies)
    test_X = fill_na(test_X, strategies)

    cols_to_drop = get_cols_to_drop_for_ordinal_encoder(high_cardinality_columns, X, test_X)
    # removes all dropped columns from the set, to not get KeyErrors while transforming
    high_cardinality_columns -= cols_to_drop
    X = X.drop(cols_to_drop, axis=1)
    test_X = test_X.drop(cols_to_drop, axis=1)

    strategies = [
        (low_cardinality_columns, OneHotEncoder(handle_unknown='ignore', sparse=False)),
        (high_cardinality_columns, OrdinalEncoder())
    ]

    X = encode_columns(X, num_cols.columns, strategies)
    test_X = encode_columns(test_X, num_cols.columns, strategies)

    clusters = [
        (['MSSubClass', 'Neighborhood'], KMeans()),
        (['YearBuilt'], KMeans()),
        (['GarageYrBlt', 'GarageArea', 'GarageCars'], KMeans()),
    ]

    # X, test_X = cluster_data(X, test_X, clusters)

    X.head().to_csv('validation_test.csv')

    return X, test_X


# Sadly, the RandomForestRegressor and XGBoost don't have a Common super class,
# which could be used to define a return type here
def generate_model(X: pd.DataFrame, y: pd.Series, model: {"fit", "predict"}) -> Tuple[Any, float]:
    """
    prepares and scores the given model
    :param X: The train dataset
    :param y: The values to predict
    :param model: The model to prepare
    :return: The Trained model and its MAPE score
    """

    def get_score(x_train, x_val, y_train, y_val, model) -> float:
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        return mean_absolute_percentage_error(y_val, predictions)

    score = get_score(*train_test_split(X, y), model)
    model.fit(X, y)
    return model, score


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

    X, test_X = transform_dataset(X, test_data)

    # plot_columns(X, get_top_10_missing_percent)

    models = [
        *[[i, XGBRegressor(n_estimators=i, random_state=0)] for i in range(10, 151, 10)]
    ]

    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'n_estimators': list(range(10, 101, 10)),
        'max_depth': [3, 4, 5],
        'learn_rate': [0.1, 0.01, 0.05, 0.03],
    }

    xgb = XGBClassifier(objective='binary:logistic',
                        silent=True, nthread=1)

    captures: List[Tuple[int, Any, float]] = []

    folds = 5
    param_comb = 50

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='neg_mean_absolute_error', n_jobs=4,
                                       cv=skf.split(X, y), verbose=3, random_state=1001)

    start = time.time()
    random_search.fit(X, y)
    stop = time.time()
    print(f'took {stop-start/60} seconds')

    print('\n All results:')
    print(random_search.cv_results_)
    print('\n Best estimator:')
    print(random_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * 2 - 1)
    print('\n Best hyperparameters:')
    print(random_search.best_params_)
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

    # for i, model in models:
    #     captures.append((i, *generate_model(X, y, model)))
    # print(captures)
    # best_capture = None
    # for i, model, score in captures:
    #     if best_capture is None or best_capture[1] > score:
    #         best_capture = model, score
    # plt.plot([cap[0] for cap in captures], [cap[2] for cap in captures])
    # plt.show()
    # best_model = best_capture[0]
    # predictions = best_model.predict(test_X)
    # write_prediction(pd.DataFrame({'Id': test_X.Id, 'SalePrice': predictions}))


if __name__ == '__main__':
    main()
