# Bases on pipeline
# Import helpful libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt


def main():
    # Load the data, and separate the target
    iowa_file_path = '../train.csv'
    home_data = pd.read_csv(iowa_file_path)
    home_data.dropna(subset=['SalePrice'], inplace=True)
    y = home_data.SalePrice
    home_data.drop(['SalePrice'], axis=1, inplace=True)

    num_transformer = SimpleImputer()
    l_car_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    h_car_transformer = OrdinalEncoder()

    num_cols = home_data.select_dtypes(exclude=['object']).columns
    obj_cols = home_data.select_dtypes(include=['object']).columns
    l_car_cols = [col for col in obj_cols if home_data[col].nunique() < 10] + ['Cluster']
    h_car_cols = set(obj_cols) - set(l_car_cols)
    my_cols = set(num_cols) | set(obj_cols)
    X = home_data[my_cols].copy()


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('lcar', l_car_transformer, l_car_cols),
            # ('hcar', h_car_transformer, h_car_cols)
        ],
    )
    # print(X.dtypes)

    features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea']

    # X['TotalBsmtSF'] = 0

    # Standardize
    X_scaled = X.select_dtypes(exclude=['object']).fillna(0).to_numpy() #.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

    def get_score(val, preprocessor):
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(n_estimators=200, random_state=1))
            ]
        )
        # Fit the KMeans model to X_scaled and create the cluster labels
        kmeans = KMeans(n_clusters=val, n_init=10, random_state=1)
        X["Cluster"] = kmeans.fit_predict(X_scaled)
        X["Cluster"] = X["Cluster"].astype("category")
        # print(X['Cluster'])
        scores = (-1*cross_val_score(pipeline, X, y, cv=5, verbose=1, n_jobs=4, scoring='neg_mean_absolute_error'))
        print(scores)
        print(X.groupby('Cluster').size())
        return scores.mean()

    results = {
        i: get_score(i, preprocessor)
        for i in range(4, 17, 2)
    }
    plt.plot(list(results.keys()), list(results.values()))
    plt.show()


if __name__ == '__main__':
    main()
