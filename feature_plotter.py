import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import util


def main():
    train, _ = util.read_datasets()
    X: pd.DataFrame = train
    # y: pd.Series = X.pop('SalePrice')
    cols = len(X.columns)
    print(f'amount of columns: {cols}')
    # g = sns.FacetGrid(X)
    for i, name in enumerate(X.select_dtypes('object').columns):
        if name not in ['LandContour', 'MiscVal', 'SaleType', 'SaleCondition']:
            continue
        # g.map(sns.catplot, name, 'SalePrice', data=X)
        X[name] = X[name].fillna('None')
        sns.catplot(name, 'SalePrice', data=X)
        print(f'finished plot {i}')
    plt.show()
    # for i, name in enumerate(X.select_dtypes(exclude='object').columns):
    #     plt.scatter(X[name], X['SalePrice'])
    #     plt.xlabel(name)
    #     print(f'finished plot {i}')
    #     plt.show()
    print(util.get_ordinal_columns(X))


if __name__ == '__main__':
    main()
