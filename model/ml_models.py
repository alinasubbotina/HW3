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


def hyper_param(model, param_grid: dict, X_train: pd.DataFrame, y_train: pd.DataFrame):
    logging.info("Hyperparameter tuning for the model")
    CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    CV.fit(X_train, y_train)
    return CV.best_params_

def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, path_to_model: str):
    # Initialize the model
    rfc = RandomForestClassifier(max_depth=2, random_state=0)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }
    best_params = hyper_param(rfc, param_grid, X_train, y_train)
    rfc_tuned = RandomForestClassifier(**best_params, random_state=0)
    logging.info("Training the Random Forest model")
    # Train the model
    rfc_tuned.fit(X_train, y_train)
    logging.info(f'Accuracy for Random Forest model is {rfc_tuned.score(X_test, y_test)}')
    save_model(dir=path_to_model, model=rfc_tuned)
    logging.info("Random Forest model is saved")
    return rfc_tuned


def train_naive_bayes(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame, path_to_model: str):
    # Initialize the model
    lr = GaussianNB()
    param_grid = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }
    best_params = hyper_param(lr, param_grid, X_train, y_train)
    lr_tuned = GaussianNB(**best_params)
    logging.info("Training the Naive Bayes model")
    # Train the model
    lr_tuned.fit(X_train, y_train)
    logging.info(f'Accuracy for Naive Bayes model is {lr_tuned.score(X_test, y_test)}')
    save_model(dir=path_to_model, model=lr_tuned)
    logging.info("Naive Bayes model is saved")
    return lr_tuned


def prediction(path_to_data: str, path_to_model: str) -> None:
    df = get_data(path_to_data)
    X_train, X_test, y_train, y_test = split(df)
    if not os.path.exists(path_to_model) and path_to_model == settings.rf_conf:
        train_random_forest(X_train, y_train, X_test, y_test, path_to_model)
    elif not os.path.exists(path_to_model) and path_to_model == settings.lr_conf:
        train_naive_bayes(X_train, y_train, X_test, y_test, path_to_model)
    else:
        logging.info("The model is found")
    clf = load_model(dir=path_to_model)
    logging.info(f'Prediction for model is {clf.predict(X_test)}')
#    return clf.predict(X_test)
