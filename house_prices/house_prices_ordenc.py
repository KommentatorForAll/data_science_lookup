# Bases on nummiss
# THIS IS SLIGHTLY WORSE
# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder


def main():
    # Load the data, and separate the target
    iowa_file_path = '../train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice

    imp = SimpleImputer()
    enc = OrdinalEncoder()


    # Exclude everything which is not numeric
    X = home_data.drop(['SalePrice'], axis=1)


    # Split the given Set into train and validation.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Categorical columns in the training data
    object_cols = [col for col in train_X.columns if train_X[col].dtype == "object"]

    # Columns that can be safely ordinal encoded
    good_label_cols = [col for col in object_cols if
                       set(val_X[col]).issubset(set(train_X[col]))]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    train_X = train_X.drop(bad_label_cols, axis=1)
    val_X = val_X.drop(bad_label_cols, axis=1)

    train_names = train_X.columns
    val_names = val_X.columns
    # transforming the data using simple imputer
    train_X = pd.DataFrame(imp.fit_transform(train_X))
    val_X = pd.DataFrame(imp.transform(val_X))
    train_X.columns = train_names
    val_X.columns = val_names

    train_X[good_label_cols] = enc.fit_transform(train_X[good_label_cols])
    val_X[good_label_cols] = enc.transform(val_X[good_label_cols])

    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    # Write prediction and real into test1.csv to be able to visualize it with Excel
    df = pd.DataFrame({'pred': rf_val_predictions, 'real': val_y})
    df.to_csv('test1.csv')

    names = X.columns
    X = pd.DataFrame(imp.fit_transform(X))
    X.columns = names

    # Run the analysis on the whole Set
    fr_full_data = RandomForestRegressor(random_state=1)
    fr_full_data.fit(X, y)
    test_data = pd.read_csv('../test.csv')
    test_X = test_data.select_dtypes(exclude=['object'])
    names = test_X.columns
    test_X = pd.DataFrame(imp.fit_transform(test_X))
    test_X.columns = names
    # Predict for the test data
    predictions = fr_full_data.predict(test_X)
    df = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
    # Write predicted data into the submission.csv
    df.to_csv('submission.csv', index=False)

    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


if __name__ == '__main__':
    main()
