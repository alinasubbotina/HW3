from model.ml_models import split, train_random_forest, train_naive_bayes, logging, predict
from conf.conf import settings
from connector.pg_connector import get_data

df = get_data(settings.data_set)
X_train, X_test, y_train, y_test = split(df)

# Random Forest
clf = train_random_forest(X_train, y_train, settings.rf_conf)

logging.info(f'Accuracy for Random Forest model is {clf.score(X_test, y_test)}')

response = predict(X_test, settings.rf_conf)
logging.info(f'Prediction for Random Forest model is {clf.predict(X_test)}')

# Naive Bayes
clf = train_naive_bayes(X_train, y_train, settings.lr_conf)

logging.info(f'Accuracy for Naive Bayes model is {clf.score(X_test, y_test)}')

response = predict(X_test, settings.lr_conf)
logging.info(f'Prediction for Naive Bayes model is {clf.predict(X_test)}')
