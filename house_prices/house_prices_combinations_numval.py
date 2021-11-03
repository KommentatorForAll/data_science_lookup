# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import itertools

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

# Generates a list of combinations to test.
# It will only select up to ~5 elements per combination.
feature_combinations = []
for L in range(1, len(feature_list) // 4):
    for subset in itertools.combinations(feature_list, L):
        feature_combinations.append(list(subset))

min_mae = None
# enumerating the runs to have some kind of progress bar
for i, combination in enumerate(feature_combinations):
    # Fits and tests a RandomForestRegressor for each of the combinations
    X = home_data[combination]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    # Checks if the current combination performed better than the last best.
    if min_mae is None or rf_val_mae < min_mae[1]:
        min_mae = [combination, rf_val_mae]
    # Prints a small progress message after 100 runs
    if i % 100 == 0:
        print(f'finished {i} runs')
        print(f'best so far: {min_mae}')

# Select columns corresponding to features, and preview the data
X = home_data[min_mae[0]]
X.head()

fr_full_data = RandomForestRegressor(random_state=1)
fr_full_data.fit(X, y)

# shows best combination
print("Validation MAE for Random Forest Model: {:,.0f} with columns: {}".format(min_mae[1], min_mae[0]))

# Performs prediction for the test dataset
test_data = pd.read_csv('../test.csv')
test_X = test_data[min_mae[0]]
test_preds = fr_full_data.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
# Writes predictions to file
output.to_csv('submission.csv', index=False)
