import string

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, Ridge


def train_linear_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "lasso",
    model_args: dict = {},

) -> tuple[pd.DataFrame, pd.Series, pd.Series, BaseEstimator]:
    """
    Fits a lasso model on the train data and returns a pandas dataframe with feature names and
    feature coefficients, train predictions, test predictions, and the trained model object.

    Args:
        X_train: A pandas DataFrame containing the feature values for the training data.
        X_test: A pandas DataFrame containing the feature values for the test data.
        y_train: A pandas Series containing the target values for the training data.
        model_type: string value having "lasso" or "ridge"
        model_args: model arguments to linear model (refer sklearn documentation for possible argume)

    Returns:
        A tuple containing the following:
        - The trained linear regression model object.
        - A pandas dataframe with two columns containing feature names and feature coefficients.
        - A pandas Series with the train predictions.
        - A pandas Series with the test predictions.
    """
    # Fit a linear regression model on the training data

    # constrain the coefficients to be +ve
    model_args["positive"] = True

    if model_type == "lasso":
        model = Lasso(**model_args)
    elif model_type == "ridge":
        model = Ridge(**model_args)
    else:
        raise ValueError("Please enter either 'ridge' or 'lasso' as the model_type")

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


def process_input_priors(pr_cfg_df: pd.DataFrame, data: pd.DataFrame) -> dict:
    """function to map user input priors in excel file to bambi prior format

    Args:
        pr_cfg_df (pd.DataFrame): user input prior config in excel read as a pandas dataframe
        data (pd.DataFrame): training data (to check if the idv entered by the user are present in the data)
    Returns:
        processed_priors (dict): dictionary containing proper prior format as required by bambi
        
    """
    priors_mapping = {
        "normal": "Normal",
        "halfnormal": "HalfNormal",
    }

    processed_priors = {}
    
    for _, row in pr_cfg_df.iterrows():
        idv = row["idv"]

        # check if idv in prior config file is present in data
        if idv not in data.columns.tolist():
            raise ValueError(f"{idv} is not present in the data, Please the check the column names in the prior config excel file")

        # if we have a prior input from the user
        if not pd.isnull(row["prior_est"]):
            print(row["prior_est"])
            input_prior_est = row["prior_est"].lower()

            # check if prior distribution is there in the mapping dictionary
            if input_prior_est not in priors_mapping:
                raise ValueError(f"Please use priors from this list, {priors_mapping.keys()}. To use other priors, directly use priors in the format that bambi requires")

            match input_prior_est:
                case "normal":
                    try:
                        mu = row["mu"]
                        sigma = row["sigma"]
                        processed_priors[row["idv"]] = bmb.Prior("Normal", mu=mu, sigma=sigma)
                    except:
                        raise Exception(f"Please enter valid mu and sigma for {input_prior_est}")
                case "halfnormal":
                    try:
                        sigma = row["sigma"]
                        processed_priors[row["idv"]] = bmb.Prior("HalfNormal", sigma=sigma)
                    except:
                        raise Exception(f"Please enter valid sigma for {input_prior_est}")

    return processed_priors


def create_model_equation(pr_cfg_df, group_var="", random_intercept=True) -> str:
    """function to create mixed model equation from the user

    Args:
        pr_cfg_df (pd.DataFrame): Pandas dataframe with user results
        group_var (str, optional): group variable for random effects. Defaults to "".
        random_intercept (bool, optional): do we want intercept for random effects. Defaults to False.

    Returns:
        model_equation (str): returns the mixed model equation.
    """
    
    dv = pr_cfg_df["dv"].unique()[0]
    
    fixed_effects = pr_cfg_df.query("is_fixed == 1")["idv"].values.tolist()
    fixed_equation = "+".join(fixed_effects)
         
    random_idvs = pr_cfg_df.query("is_random == 1")["idv"].values.tolist()
    
    if random_intercept:
        random_effects = [f"(1 + {var} | {group_var})" for var in random_idvs]
    else:
        random_effects = [f"(-1 + {var} | {group_var})" for var in random_idvs]
        
    ranef_equation = "+".join(random_effects)
    
    if len(random_effects) > 0:
        rhs_eq = fixed_equation + "+ " + ranef_equation
    else:
        rhs_eq = fixed_equation

    model_equation = dv + " ~ " + rhs_eq
    return model_equation


def _get_bambi_predictions(pred_object):
    final_preds = (
        az.summary(pred_object).reset_index().query("index.str.contains('_mean')")["mean"].values
    )

    return final_preds


def train_bayesian(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series,
    model_equation: str = "",
    priors_config: dict = None,
    model_args: dict = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, bmb.Model]:
    """
    Fits a Bayesian linear regression model on the train data and returns a pandas dataframe with feature names 
    and feature coefficients, train predictions, test predictions, and the trained model object.
    
    Args:
        X_train: A pandas DataFrame containing the feature values for the training data.
        X_test: A pandas DataFrame containing the feature values for the test data.
        y_train: A pandas Series containing the target values for the training data.
        y_test: A pandas Series containing the target values for the test data.
        priors_config: A dictionary with prior distributions for the model parameters. Defaults to None.
    
    Returns:
        A tuple containing the following:
        - A pandas dataframe with two columns containing feature names and feature coefficients.
        - A pandas Series with the train predictions.
        - A pandas Series with the test predictions.
        - The trained Bayesian linear regression model object.
    """
    dv_name = model_equation.split('~')[0].strip()

    # Add the target variable to the X_train and X_test dataframes
    X_train[dv_name] = y_train
    X_test[dv_name] = y_test
    
    # Define the model using Bambi
    model = bmb.Model(model_equation, data=X_train)
    if priors_config is not None:
        model.set_priors(priors_config)
    
    model_args["include_mean"] = True
    # Fit the model using MCMC sampling
    trace = model.fit(**model_args)
    
    # Get the posterior distribution of the model coefficients
    feature_df = az.summary(trace)['mean'].reset_index().query("~index.str.contains('y_mean')")
    feature_df = feature_df.rename(columns={ "index": "feature names", "mean": "feature coefficients" })
    
    # Make predictions on the training and test data
    train_preds_object = model.predict(idata=trace, data=X_train, inplace=False)
    test_preds_object = model.predict(idata=trace, data=X_test, inplace=False)
    
    train_preds = _get_bambi_predictions(pred_object=train_preds_object)
    test_preds = _get_bambi_predictions(pred_object=test_preds_object)
    
    return model, feature_df, train_preds, test_preds
