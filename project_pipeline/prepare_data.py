import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data():
    df = pd.read_csv('../Data/train.csv')
    return df


def prepare_data(df):
    df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    add_mean_age(df)
    df.dropna(inplace=True)
    cast_to_numeric(df, ['Sex', 'Embarked'])


def cast_to_numeric(df, categorical):
    le = LabelEncoder()
    for _column in categorical:
        df[_column] = le.fit_transform(df[_column])


def add_mean_age(df):
    mean_age = df['Age'].mean()
    print(mean_age)
    df['Age'] = df['Age'].fillna(mean_age)


def main():
    df = load_data()
    prepare_data(df=df)
    df.to_csv(prepared_data, index=False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: python3 prepare_data.py train prepared_data')
        sys.exit(1)
    train = sys.argv[1]
    prepared_data = sys.argv[2]
    main()
