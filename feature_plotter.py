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
        sns.boxplot(name, 'SalePrice', data=X)
        print(f'finished plot {i}')
        os.makedirs('./house_prices/assets/plots/by_raw/', exist_ok=True)
        plt.savefig(f'./house_prices/assets/plots/by_raw/{name}_box.png')
        plt.clf()
    # plt.show()


def plot_numeric_columns(X: pd.DataFrame):
    for i, name in enumerate(X.select_dtypes(exclude='object').columns):
        sns.regplot(X[name], X['SalePrice'])
        plt.xlabel(name)
        print(f'finished plot {i}')
        os.makedirs('./house_prices/assets/plots/by_raw/', exist_ok=True)
        plt.savefig(f'./house_prices/assets/plots/by_raw/{name}_reg.png')
        plt.clf()
        # plt.show()


def main():
    train, _ = util.read_datasets('./house_prices/assets/input/')
    X: pd.DataFrame = train
    cols = len(X.columns)
    print(f'amount of columns: {cols}')
    plot_numeric_columns(X)
    plot_object_columns(X)


if __name__ == '__main__':
    main()
