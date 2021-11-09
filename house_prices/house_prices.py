import os
import time
import warnings
from typing import List, Tuple, Set, Union, Iterable

import pandas as pd
from pandas import Index
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer, KNNImputer
# This is just for Typing. I don't acutally use this Class anywhere
from sklearn.impute._base import _BaseImputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing._encoders import _BaseEncoder, OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

import util
from util import read_datasets

warnings.filterwarnings('ignore')

imputers_to_look_at = [
    SimpleImputer,
    KNNImputer,
]


ORDINAL_ORDER = ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']



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


def split_onehot_ordinal_columns(obj_cols: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """
    Splits the given object columns into Columns suitable for orinal encoding and onehot encoding
    :param obj_cols: all columns to look at
    :param threshold: the point at which the cardinality is counted as high
    :return: A tuple of onehot and ordinal encodeable columns
    """
    full_set = set(obj_cols.columns)
    ordinal_cols = set(util.get_ordinal_columns(obj_cols))
    print(ordinal_cols)
    onehot_cols = full_set - ordinal_cols
    return onehot_cols, ordinal_cols


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
        if type(encoder) == OrdinalEncoder:
            # for col in columns:
            #     categories = np.array([x for x in ORDINAL_ORDER if x in X[col].unique()])
            #     print(categories.shape)
            #     print(X[col].unique().shape)
            #     encoder = OrdinalEncoder(categories=list(categories))
            #     encoded = pd.DataFrame(encoder.fit_transform(pd.DataFrame(X[col])))
            #     encoded.index = X.index
            #     encoded.columns = [col]
            #     df = pd.concat([df, encoded], axis=1)
            # continue
            X[list(columns)] = X[list(columns)].fillna('None')
            existing_values = []
            [existing_values.append([x for x in ORDINAL_ORDER if x in X[c].unique()]) for c in columns]
            print(existing_values)
            encoder = OrdinalEncoder(categories=existing_values)

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
    low_cardinality_columns, high_cardinality_columns = split_onehot_ordinal_columns(obj_cols)

    strategies = [
        (num_cols.columns, SimpleImputer(strategy='mean')),
        # (low_cardinality_columns, SimpleImputer(strategy='most_frequent')),  # KNN might be quite useful here
        # (high_cardinality_columns, SimpleImputer(fill_value='None')),  # Same over here
    ]
    obj_cols[list(high_cardinality_columns)] = obj_cols[list(high_cardinality_columns)].fillna('None')

    X = fill_na(X, strategies)
    test_X = fill_na(test_X, strategies)

    cols_to_drop = get_cols_to_drop_for_ordinal_encoder(high_cardinality_columns, X, test_X)
    # removes all dropped columns from the set, to not get KeyErrors while transforming
    high_cardinality_columns -= cols_to_drop
    X = X.drop(cols_to_drop, axis=1)
    test_X = test_X.drop(cols_to_drop, axis=1)

    strategies = [
        (low_cardinality_columns, OneHotEncoder(handle_unknown='ignore', sparse=False)),
        (high_cardinality_columns, OrdinalEncoder(categories={'Ex', 'Gd', 'TA', 'Fa', 'Po', 'None'}))
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
# def generate_model(X: pd.DataFrame, y: pd.Series, model: {"fit", "predict"}) -> Tuple[Any, float]:
#     """
#     prepares and scores the given model
#     :param X: The train dataset
#     :param y: The values to predict
#     :param model: The model to prepare
#     :return: The Trained model and its MAPE score
#     """
#
#     def get_score(x_train, x_val, y_train, y_val, model) -> float:
#         model.fit(x_train, y_train)
#         predictions = model.predict(x_val)
#         return mean_absolute_percentage_error(y_val, predictions)
#
#     score = get_score(*train_test_split(X, y), model)
#     model.fit(X, y)
#     return model, score


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


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(df[(df['BsmtQual'] == 'Gd') & (df['SalePrice'] > 500000)].index)
    df = df.drop(df[(df['BsmtQual'] == 'TA') & (df['SalePrice'] > 350000)].index)

    df = df.drop(df[(df['BsmtCond'] == 'Gd') & (df['SalePrice'] > 400000)].index)

    df = df.drop(df[(df['ExterQual'] == 'Fa') & (df['SalePrice'] > 200000)].index)
    df = df.drop(df[(df['ExterQual'] == 'Gd') & (df['SalePrice'] > 450000)].index)

    df = df.drop(df[(df['FireplaceQu'] == 'TA') & (df['SalePrice'] > 400000)].index)
    df = df.drop(df[(df['FireplaceQu'] == 'Ex') & (df['SalePrice'] > 500000)].index)

    df = df.drop(df[(df['GarageFinish'] == 'RFn') & (df['SalePrice'] > 500000)].index)
    df = df.drop(df[(df['GarageFinish'] == 'UInf') & (df['SalePrice'] > 300000)].index)
    df = df.drop(df[(df['GarageFinish'] == 'Fin') & (df['SalePrice'] > 550000)].index)

    df = df.drop(df[(df['KitchenQual'] == 'Gd') & (df['SalePrice'] > 500000)].index)
    df = df.drop(df[(df['KitchenQual'] == 'TA') & (df['SalePrice'] > 300000)].index)

    df = df.drop(df[(df['1stFlrSF'] > 4000)].index)

    df = df.drop(df[(df['FullBath'] == 0) & (df['SalePrice'] > 300000)].index)

    df = df.drop(df[(df['GarageCars'] == 4)].index)

    df = df.drop(df[(df['GrLivArea'] > 4000)].index)

    df = df.drop(df[(df['OverallCond'] == 2) & (df['SalePrice'] > 300000)].index)
    df = df.drop(df[(df['OverallCond'] == 5) & (df['SalePrice'] > 700000)].index)
    df = df.drop(df[(df['OverallCond'] == 6) & (df['SalePrice'] > 700000)].index)

    df = df.drop(df[(df['OverallQual'] == 3) & (df['SalePrice'] > 200000)].index)
    df = df.drop(df[(df['OverallQual'] == 8) & (df['SalePrice'] > 300000)].index)
    df = df.drop(df[(df['OverallQual'] == 10) & (df['SalePrice'] < 200000)].index)

    df = df.drop(df[(df['TotalBsmtSF'] > 5000)].index)

    df = df.drop(df[(df['TotRmsAbvGrd'] == 10) & (df['SalePrice'] > 500000)].index)
    df = df.drop(df[(df['TotRmsAbvGrd'] > 14)].index)

    return df


def main():
    """
    Loads the dataset analyses it and writes the predictions into a file
    :return:
    """

    train_data, test_data = read_datasets()
    # features: List[str] = [
    #     'OverallQual',
    #     'YearBuilt', 'TotalBsmtSF', '1stFlrSF',
    #     'GrLivArea', 'FullBath', 'GarageCars',
    #     'GarageArea',
    #     'Id',
    #     *util.get_ordinal_columns(train_data),
    # ]
    features: List[str] = [
        'BsmtQual', 'BsmtCond', 'CentralAir', 'ExterQual', 'FireplaceQu', 'GarageFinish',
        'HeatingQC', 'KitchenQual', '1stFlrSF', 'FullBath', 'GarageCars', 'GrLivArea', 'OverallCond', 'OverallQual',
        'TotalBsmtSF', 'TotRmsAbvGrd',
        'Id'
    ]
    # features = [*train_data.select_dtypes(exclude=['object']).columns, *util.get_ordinal_columns(train_data)]
    # features.remove('SalePrice')
    print(features)

    train_data = drop_outliers(train_data)
    X: pd.DataFrame = train_data[features].copy()
    X.drop('Id', axis=1)
    test_data = test_data[features].copy()
    # Setting y to the SalePrice as that is the value we are trying to predict
    y = train_data['SalePrice']

    X, test_X = transform_dataset(X, test_data)

    models = [
        *[[i, XGBRegressor(n_estimators=i, random_state=0)] for i in range(10, 151, 10)]
    ]

    params = {
        'min_child_weight': [0],  # [1, 5, 10, 20] -> 1; [0.5, 0.75, 1, 2, 3, 4] -> 0.5; [0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5] -> .05
        'gamma': [0],  # [0.5, 1, 1.5, 2, 5] -> 0.5; [0.1, 0.25, 0.5, 0.75, 1] -> 0.1; [0.01, 0.025, 0.05, 0.075, 0.1] -> 0.01
        'n_estimators': range(20,31,2),  # [22],  # range(10, 201,10) -> 20; range(10,30,2) -> 22
        'max_depth': range(3,6),  # [4],  # range(3,6) -> 4
        'eta': [0.25],  # [0.1, 0.01, 0.05, 0.03] -> 0.1; [0.1, 0.05, 0.075, 0.25, 0.5, 0.75, 1] -> 0.25; [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5] -> 0.2
        'tree_method': ['approx'],  # ['auto', 'exact', 'approx', 'hist', 'gpu_hist'] -> 'approx'
        'sketch_eps': [0.03],  # [0.01, 0.02, 0.03, 0.04, 0.05] -> 0.03;
    }

    xgb: XGBRegressor = XGBRegressor(nthread=1)

    folds = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    grid_search = GridSearchCV(xgb, param_grid=params, scoring='neg_mean_absolute_error', n_jobs=4,
                               cv=skf.split(X, y), verbose=3, refit=True)

    start = time.time()
    grid_search.fit(X, y)
    stop = time.time()

    print('\n All results:')
    print(grid_search.cv_results_)
    print('\n Best estimator:')
    print(grid_search.best_estimator_)
    print('\n Best normalized gini score for %d-fold search' % folds)
    print(grid_search.best_score_)
    print('\n Best hyperparameters:')
    print(grid_search.best_params_)
    results = pd.DataFrame(grid_search.cv_results_)
    print('\n Feature importance:')
    feature_importance = grid_search.best_estimator_.get_booster().get_score(importance_type='weight')
    print(feature_importance)
    os.makedirs(f'./assets/results/paramsearch/', exist_ok=True)
    results.to_csv(f'./assets/results/paramsearch/{start}-xgb-grid-search.csv', index=False)
    pd.DataFrame({key: [value] for key, value in feature_importance.items()}).to_csv(f'./assets/results/paramsearch/{start}-xgb-features.csv', index=False)
    m, sec = divmod(stop-start, 60)
    print(f'took {int(m)} min {int(sec)} sec')

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

    predictions = grid_search.best_estimator_.predict(test_X)
    util.write_prediction(pd.DataFrame({'Id': test_X['Id'], 'SalePrice': predictions}))


if __name__ == '__main__':
    main()
