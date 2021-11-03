# Bases on all_numval
# Import helpful libraries
import pandas as pd
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
    l_car_cols = [col for col in obj_cols if home_data[col].nunique() < 10]
    h_car_cols = set(obj_cols) - set(l_car_cols)
    my_cols = set(num_cols) | set(obj_cols)
    X = home_data[my_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('lcar', l_car_transformer, l_car_cols),
            # ('hcar', h_car_transformer, h_car_cols)
        ],
        remainder='drop'
    )

    def get_score(model, preprocessor):
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        scores = (-1*cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error'))
        print(scores)
        return scores.mean()

    results = {
        i: get_score(RandomForestRegressor(n_estimators=i, random_state=1), preprocessor)
        for i in range(50, 401, 50)
    }
    plt.plot(list(results.keys()), list(results.values()))
    plt.show()


if __name__ == '__main__':
    main()
