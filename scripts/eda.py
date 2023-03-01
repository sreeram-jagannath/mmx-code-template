from typing import List

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calculate_vif(X : pd.DataFrame , idv : List  , dv : str) -> pd.DataFrame:
    temp = pd.DataFrame()
    temp["variable name"] = idv
    vif_filter = X[idv] 
    vif_filter = vif_filter.assign(const=1)  
    variables = list(range(vif_filter.shape[1]))

    vif = [variance_inflation_factor(vif_filter.iloc[:, variables].values, ix)
            for ix in range(vif_filter.iloc[:, variables].shape[1])]
    vif=vif[:-1]
    temp["VIF"] = vif

    corr_mat = X[[*idv,dv]].corr()
    corr_mat = corr_mat[dv]
    corr_mat = corr_mat.drop(index=dv).reset_index()

    return temp.merge(corr_mat,how="inner",left_on="variable name",right_on="index").drop("index",axis=1).round(2)
