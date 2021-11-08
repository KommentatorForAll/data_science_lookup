import matplotlib.pyplot as plt
import pandas as pd


def main():
    df: pd.DataFrame = pd.read_csv('home-data-for-ml-course-publicleaderboard.csv')
    print(df.sort_values('Score').tail(10))
    plt.hist(df['Score'], bins=100, range=(10000, 25000))
    plt.show()


if __name__ == '__main__':
    main()
