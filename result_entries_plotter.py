import matplotlib.pyplot as plt
import pandas as pd


def main():
    df: pd.DataFrame = pd.read_csv('home-data-for-ml-course-publicleaderboard.csv')
    print(df.sort_values('Score').tail(10))
    # df['Score'].hist()
    plt.hist(df['Score'], bins=300, range=(10000, 25000))
    plt.show()


def prices():
    df: pd.DataFrame = pd.read_csv('train.csv')
    df['SalePrice'].hist()
    plt.show()


if __name__ == '__main__':
    # prices()
    main()
