import argparse
import pickle as pickle

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

font = {'size': 14}
matplotlib.rc('font', **font)

def calc_diff(date, dc):
    return (dc-date).days

def get_data(filename):
    """Load raw data from a file and return training data and responses.
    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.
    Returns
    -------
    data: A dataframe containing features to be preprocessed and used for training.
    y: A dataframe containing labels, used for model response.
    """
    data = pd.read_csv(filename, sep=sep, header=0)
    return data


def preprocessing(data):
    """Take cleaned dataframe and preprocess features to prepare for training
    Parameters
    ----------
    data: a dataframe containing features to be preprocessed and used for training.
    Returns
    -------
    X: A dataframe of preprocessed data ready for training
    """
    data['last_trip_date'] = data['last_trip_date'].astype('datetime64')
    data['signup_date'] = data['signup_date'].astype('datetime64')

    data.loc[:,['avg_rating_by_driver']] = data[['avg_rating_by_driver']].fillna(data[['avg_rating_by_driver']].mean())
    data.loc[:,['avg_rating_of_driver']] = data[['avg_rating_of_driver']].fillna(data[['avg_rating_of_driver']].mean())
    data.loc[:,['phone']] = data[['phone']].replace(np.nan, 'Other')

    data = pd.get_dummies(data, columns=['city','phone'])
    data = data.drop(columns=['city_Astapor', 'phone_Other'])

    col = 'last_trip_date'
    data[col]= pd.to_datetime(data[col])

    dc = pd.Timestamp("2014-07-01")
    data['days_since_last_ride'] = data.apply(lambda row: calc_diff(row['last_trip_date'], dc),axis=1)
    data['churn'] = data['days_since_last_ride']>30

    data_model = data[['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_surge'
                   , 'surge_pct', 'trips_in_first_30_days',
       'luxury_car_user', 'weekday_pct', "city_King's Landing",
       'city_Winterfell', 'phone_Android', 'phone_iPhone','churn']]

    X = data_model.copy()
    y = X.pop('churn')
    return X, y

def train(filename, use_tree):
    """Load raw data from a file and train either using linear regression or random forest

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.
    use_tree: if True, use Random Forest Regressor

    Returns
    -------
    model: a model fit using either linear regression or random forest
    """
    data = get_data(filename)
    X, y = preprocessing(data)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-2)
    logit = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    if use_tree:
        clf = rf
    else:
        clf = logit
    model = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    summary(X_test, y_test)
    return model


def summary(X_test, y_test):
    """return descriptive statistics of predicted and true responses

    Parameters
    ----------
    y_true: the true value of the response
    y_predicted: the predicted value of the response

    Returns
    -------
    summary_output: descriptive statistical metrics
    """
    rs = model.score(X_test,y_test)
    summary_output = f'''
        R squared = {rs : .2f}
    '''
    print(summary_output)


def predict(filename, model_input_path):
    """Predict responses based on feature inputs

    Parameters
    ----------
    filename: The path to a csv file containing the raw text data and response.
    model_path: the location of the model
    Returns
    -------
    y_preds: predictions of the responses based on the input model
    """
    X = get_data(filename)

    model = joblib.load(model_input_path)
    y_preds = model.predict(X)
    return np.exp(y_preds)


def parse_arguments():
    """command line arguments

    Parameters
    ----------
    Returns
    -------
    args: arguments used for executing script on the command line
    """
    parser = argparse.ArgumentParser(
        description='Fit a Classifier model and save the results.')
    parser.add_argument('--data', help='A tab delimited csv file with input data.')
    parser.add_argument('--model_output_path',
                        help='A file to save the serialized model object to.', default='model.joblib')
    parser.add_argument('mode', help='train or predict model', default='predict')
    parser.add_argument('--output_file', help='where to save the model predictions',
                        default='predictions.txt')
    parser.add_argument('--tree_model', action='store_true',
                        help='if True, use Random Forest Model')
    parser.add_argument('--model_input_path', help='model to load', default='model.joblib')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    if args.mode == 'train':
        model = train(args.data, args.tree_model)
        joblib.dump(model, args.model_output_path)
    if args.mode == 'predict':
        preds = predict(args.data, args.model_input_path)
        np.savetxt(args.output_file, preds, delimiter="\t")
