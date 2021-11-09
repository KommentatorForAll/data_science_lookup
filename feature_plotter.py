import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import util


def plot_object_columns(X: pd.DataFrame):
    for i, name in enumerate(X.select_dtypes('object').columns):
        # if name not in ['LandContour', 'MiscVal', 'SaleType', 'SaleCondition']:
        #     continue
        # g.map(sns.catplot, name, 'SalePrice', data=X)
        X[name] = X[name].fillna('None')
        sns.catplot(name, 'SalePrice', data=X)
        print(f'finished plot {i}')
        os.makedirs('./house_prices/assets/plots/by_raw/', exist_ok=True)
        plt.savefig(f'./house_prices/assets/plots/by_raw/{name}.png')
    # plt.show()


def plot_numeric_columns(X: pd.DataFrame):
    for i, name in enumerate(X.select_dtypes(exclude='object').columns):
        plt.scatter(X[name], X['SalePrice'])
        plt.xlabel(name)
        print(f'finished plot {i}')
        os.makedirs('./house_prices/assets/plots/by_raw/', exist_ok=True)
        plt.savefig(f'./house_prices/assets/plots/by_raw/{name}.png')
        # plt.show()


def main():
    train, _ = util.read_datasets()
    X: pd.DataFrame = train
    cols = len(X.columns)
    print(f'amount of columns: {cols}')
    plot_numeric_columns(X)
    plot_object_columns(X)


if __name__ == '__main__':
    main()
