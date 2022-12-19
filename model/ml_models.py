from conf.conf import logging, settings
from util.util import save_model, load_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import os.path
from connector.pg_connector import get_data



def split(df: pd.DataFrame) -> list:
    # Split variables into train and test
    # Filter out target column
    logging.info("Defining X and y")
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info("Splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, # independent variables
                                                        y # dependent variable
                                                        )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, path_to_model: str):
    # Initialize the model
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    logging.info("Training the Random Forest model")
    # Train the model
    rfc.fit(X_train, y_train)
    logging.info(f'Accuracy for Random Forest model is {rfc.score(X_test, y_test)}')
    save_model(dir=path_to_model, model=rfc)
    logging.info("Random Forest model is saved")
    return rfc


def train_naive_bayes(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, path_to_model: str):
    # Initialize the model
    lr = GaussianNB()
    logging.info("Training the Naive Bayes model")
    # Train the model
    lr.fit(X_train, y_train)
    logging.info(f'Accuracy for Naive Bayes model is {lr.score(X_test, y_test)}')
    save_model(dir=path_to_model, model=lr)
    logging.info("Naive Bayes model is saved")
    return lr


def prediction(values, path_to_model: str) -> None:
    clf = load_model(dir=path_to_model)
    logging.info(f'Prediction for model is {clf.predict(values)}')
