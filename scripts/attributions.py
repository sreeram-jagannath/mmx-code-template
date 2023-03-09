import numpy as np
import pandas as pd


def additive_attribution(X : pd.DataFrame, y : pd.Series ,  y_preds : pd.Series , coeffs : pd.DataFrame ,intercept : float = 0) -> pd.DataFrame :
    """Calulates attribution values for the model

    Args:
        X (pd.DataFrame): Dependent vars of the model
        y (pd.Series): Independent vars of the model
        y_preds (pd.Series): Predictions of independent vars
        coeffs (pd.DataFrame): Coefficents of the model
        intercept (float): intercept of the model .Defaults to 0 .

    Returns:
        pd.DataFrame: Dataframe with Residual and attribution values for each dv
    """

    res = pd.DataFrame()
    res["Residual"] = y.reset_index(drop=True) - y_preds.reset_index(drop=True)  + intercept
    coeff_mat = coeffs.set_index("feature names").T.reset_index(drop=True)
    res = pd.concat([res,pd.DataFrame(X[coeff_mat.columns].values * coeff_mat.values,columns=coeff_mat.columns)],axis=1)
    return res