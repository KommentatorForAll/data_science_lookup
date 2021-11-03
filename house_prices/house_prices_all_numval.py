# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main():
    # Load the data, and separate the target
    iowa_file_path = '../train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice

    feature_list = [
        'MSSubClass',
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        '1stFlrSF',
        '2ndFlrSF',
        'LowQualFinSF',
        'GrLivArea',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'TotRmsAbvGrd',
        'Fireplaces',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        '3SsnPorch',
        'ScreenPorch',
        'PoolArea',
        'MiscVal',
        'MoSold',
        'YrSold'
    ]
    X = home_data[feature_list]
    # Split the given Set into train and validation.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    # Write prediction and real into test1.csv to be able to visualize it with Excel
    df = pd.DataFrame({'pred': rf_val_predictions, 'real': val_y})
    df.to_csv('test1.csv')

    # Select columns corresponding to features, and preview the data
    X = home_data[feature_list]
    X.head()

    # Run the analysis on the whole Set
    fr_full_data = RandomForestRegressor(random_state=1)
    fr_full_data.fit(X, y)
    test_data = pd.read_csv('../test.csv')
    test_X = test_data[feature_list]
    # Predict for the test data
    predictions = fr_full_data.predict(test_X)
    df = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
    # Write predicted data into the submission.csv
    df.to_csv('submission.csv', index=False)

    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


if __name__ == '__main__':
    main()
    