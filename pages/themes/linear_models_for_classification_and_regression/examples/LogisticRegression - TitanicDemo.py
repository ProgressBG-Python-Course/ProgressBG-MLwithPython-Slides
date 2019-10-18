#!/usr/bin/env python

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


def prepare_data(df):

    def fill_nan_values(df):
        # Put port = Southampton for 'Embarked' null values:
        df["Embarked"] = df["Embarked"].fillna("S")

        # put the mean passengers age for 'Age' null values
        df["Age"] = df["Age"].fillna(df["Age"].median())

        # put 0 for cabin number for all 'Cabin' null values
        df["Cabin"] = df["Cabin"].fillna(0)

        # put the mean Fare for 'fare' null values:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

        return df

    def categories_to_numbers(df):
        if df['Sex'].dtype == "object":
            df["Sex"] = np.where(df["Sex"] == "male", 1,0)

        if df['Embarked'].dtype == "object":
            # is this more readable?
            # df.loc[df["Embarked"] == "S", "Embarked"] = 0
            # df.loc[df["Embarked"] == "C", "Embarked"] = 1
            # df.loc[df["Embarked"] == "Q", "Embarked"] = 2

            # Get the unique values of Embarked
            embarks = sorted(df['Embarked'].unique())

            # Generate a mapping of Embarked string to a numbers (0,1,...)
            embarks_map = dict(zip(embarks, range(0, len(embarks) + 1)))

            # Transform Embarked from a string to a number representation
            df['Embarked'] = df['Embarked'].map(embarks_map).astype(int)

        # print("df['Sex'].dtype", df['Sex'].dtype)
        # print("df['Embarked'].dtype", df['Embarked'].dtype)

        return df


    df = fill_nan_values(df)
    df = categories_to_numbers(df)

    return df

def select_features(df, features, drop,y_col):
    if drop==1:
        for col in features:
            if col in df:
                df = df.drop(labels=col, axis=1)
    else:
        y_col_data = df[y_col]
        df = df[features].copy()
        df[y_col] = y_col_data

    return df

def split_train_test(df, y_col):
    return train_test_split(
        df.drop(y_col,axis=1),
        df[y_col],
        random_state=1)


def train_and_repdict():
    pass

def print_start(sym, count):
    print("\n{}".format(sym * count))

def print_end(sym, count):
    print("{}\n".format(sym * count))


if __name__ == "__main__":
    ### Load the data
    df_train = pd.read_csv("../datasets/Titanic/train.csv", index_col='PassengerId')
    df_test = pd.read_csv("../datasets/Titanic/test.csv", index_col='PassengerId')

    # print("Columns: ", df_train.columns)

    ## prepare data:
    df_train = prepare_data(df_train)

    ## Select features
    # All:['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket','Fare', 'Cabin', 'Embarked'],
    drop_features = ['Name', 'SibSp', 'Parch', 'Ticket','Fare', 'Cabin', 'Embarked']
    use_features = ['Pclass']

    df_train = select_features(df_train, use_features,drop=0,y_col="Survived")

    # print("df_train.head(3):",df_train.head(3))



    X_train, X_test, y_train, y_test = split_train_test(df_train, 'Survived');
    print_start('*',50)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')
    print_end('*',50)



    print_start('*',50)
    print("Features used: {}".format(list(X_train.columns)))
    print_end('*',50)

    # instantiate and fit the model
    model = LogisticRegression()
    fitted = model.fit(X_train,y_train)


    ### Make predictions
    predictions = model.predict(X_test)

    # let's check the "learned" co-efficients:
    print_start('*',50)
    print("Intercept",fitted.intercept_)
    print("Coef",fitted.coef_)
    print_end('*',50)


    ### Predict (classify unknown input sample)
    y_pred = fitted.predict(X_test)


    ## Evaluate the model
    print_start('*',50)
    print("classification_report:\n",classification_report(y_test,predictions))
    print_end('*',50)

    print_start('*',50)
    print("MSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("accuracy_score:",metrics.accuracy_score(y_test, y_pred))
    print_end('*',50)



