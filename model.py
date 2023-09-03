import pandas as pd
import numpy as np
import streamlit as st
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pickle import load

def split_data(df: pd.DataFrame):

    y = df['satisfaction']
    X = df[['Customer Type', 'On-board service', 'Checkin service', 'Seat comfort', 'Cleanliness', 'Class']]

    return X, y


def open_data(path="./data/clients.csv"):

    df = pd.read_csv(path)
    df = df[['Customer Type', 'On-board service', 'Checkin service', 'Seat comfort', 'Cleanliness', 'Class', 'satisfaction']]

    return df


def preprocess_data(df: pd.DataFrame, test=True):

    if test:
        X_df, y_df = split_data(df)
        y_df = y_df.loc[y_df['satisfaction'] != '-']
        y_df['satisfaction'].replace(['satisfied', 'neutral or dissatisfied'], [1, 0], inplace=True)
    else:
        X_df = df


    for col in ['On-board service', 'Checkin service', 'Seat comfort', 'Cleanliness']:
        X_df = X_df.loc[X_df[col] <= 5]
        X_df[col].replace(0, 1, inplace=True)

    X_df['Customer Type'].replace([np.nan, 'Loyal Customer', 'disloyal Customer'], [1, 1, 0], inplace=True)
    X_df['Class'] = df['Class'].replace(['Eco', 'Eco Plus', 'Business'], [1, 2, 3])

    if test:
        return X_df, y_df
    else:
        return X_df


def fit_and_save_model(X_df, y_df, path='./data/logit_regression.pkl'):
    model = LogisticRegression(C = 86, penalty = 'l2', solver = 'sag', random_state=42)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        pickle.dump(model, file)

    print(f"Model was saved to {path}")


def load_model_and_predict(df, path='./data/logit_regression.pkl'):

    with open(path, "rb") as file:
        model = pickle.load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        1: "Пассажир доволен перелетом с вероятностью",
        0: "Пассажир не доволен перелетом с вероятностью"
    }

    encode_prediction = {
        0: "Пассажир не доволен перелетом",
        1: "Пассажир доволен перелетом"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)


