import string

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso


def train_lasso_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_args: dict = {},
) -> tuple[pd.DataFrame, pd.Series, pd.Series, BaseEstimator]:
    """
    Fits a lasso model on the train data and returns a pandas dataframe with feature names and
    feature coefficients, train predictions, test predictions, and the trained model object.

    Args:
        X_train: A pandas DataFrame containing the feature values for the training data.
        X_test: A pandas DataFrame containing the feature values for the test data.
        y_train: A pandas Series containing the target values for the training data.
        y_test: A pandas Series containing the target values for the test data.

    Returns:
        A tuple containing the following:
        - The trained linear regression model object.
        - A pandas dataframe with two columns containing feature names and feature coefficients.
        - A pandas Series with the train predictions.
        - A pandas Series with the test predictions.
    """
    # Fit a linear regression model on the training data
    model = Lasso(*model_args)
    model.fit(X_train, y_train)

    # Get the feature names and coefficients from the model
    feature_names = X_train.columns
    feature_coeffs = model.coef_
    feature_df = pd.DataFrame(
        {"feature names": feature_names, "feature coefficients": feature_coeffs}
    )

    # Make predictions on the training and test data
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    return model, feature_df, train_preds, test_preds


def _get_bambi_predictions(pred_object):
    final_preds = (
        az.summary(pred_object).reset_index().query("index.str.contains('_mean')")["mean"].values
    )

    return final_preds


def train_bayesian(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, priors=None
) -> tuple[pd.DataFrame, pd.Series, pd.Series, BaseEstimator]:
    """
    Fits a Bayesian linear regression model on the train data and returns a pandas dataframe with feature names
    and feature coefficients, train predictions, test predictions, and the trained model object.

    Args:
        X_train: A pandas DataFrame containing the feature values for the training data.
        X_test: A pandas DataFrame containing the feature values for the test data.
        y_train: A pandas Series containing the target values for the training data.
        y_test: A pandas Series containing the target values for the test data.
        priors: A dictionary with prior distributions for the model parameters. Defaults to None.

    Returns:
        A tuple containing the following:
        - The trained Bayesian linear regression model object.
        - A pandas dataframe with two columns containing feature names and feature coefficients.
        - A pandas Series with the train predictions.
        - A pandas Series with the test predictions.
    """
    # Remove all punctuation and spaces from the column names
    translator = str.maketrans("", "", string.punctuation + " ")
    X_train.columns = [col.translate(translator) for col in X_train.columns]
    X_test.columns = [col.translate(translator) for col in X_test.columns]

    # Add the target variable to the X_train and X_test dataframes
    X_train = X_train.copy()
    X_train["target"] = y_train
    X_test = X_test.copy()
    X_test["target"] = y_test

    # Define the model using Bambi
    model_equation = "target ~ " + " + ".join(X_train.columns[:-1])
    model = bmb.Model(model_equation, data=X_train)
    if priors is not None:
        model.set_priors(priors)

    # Fit the model using MCMC sampling
    trace = model.fit(draws=5, cores=1, chains=2, tune=5)

    # Get the posterior distribution of the model coefficients
    feature_df = az.summary(trace)["mean"].reset_index().query("~index.str.contains('y_mean')")
    feature_df = feature_df.rename(
        columns={"index": "feature names", "mean": "feature coefficients"}
    )

    # Make predictions on the training and test data
    train_preds_object = model.predict(idata=trace, data=X_train, inplace=False)
    test_preds_object = model.predict(idata=trace, data=X_test, inplace=False)

    train_preds = _get_bambi_predictions(pred_object=train_preds_object)
    test_preds = _get_bambi_predictions(pred_object=test_preds_object)

    return model, feature_df, train_preds, test_preds
