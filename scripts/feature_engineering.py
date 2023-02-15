from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import add_trend


def add_holidays(data: pd.DataFrame, holidays: pd.DataFrame, date_col: List, granularity: str) -> pd.DataFrame:
    """Adds Holidays Column to the dataframe .If granularity at week level holidays will have 0 or 1 values 
    .For other granularity holidays will have the sum of holidays for the week or month .

    Args:
        data (pd.DataFrame):  Input dataframe.
        holidays (pd.DataFrame): Dataframe with holidays in date format.
        date_col (List): List with Date column names
        granularity(str):Level at which the data is at.Accepatable inputs are {"Day","Week","Month"}

    Returns:
        pd.DataFrame: Input dataframe with "Is_Holidays" column
    """
    if granularity == "Week" :
        holidays = move_nearest_day(holidays, date_col=date_col[1])
    elif granularity == "Month" :
        holidays = move_nearest_month(holidays,date_col=date_col[1])
    
    holidays["Is_Holidays"] = 1
    temp = holidays.groupby(date_col[1]).sum().reset_index()

    result = pd.merge(data, temp, how="left", left_on=date_col[0], right_on=date_col[1])
    result["Is_Holidays"] = result["Is_Holidays"].fillna(0)
    return result


def move_nearest_day(df: pd.DataFrame, date_col: str, day: int = 6) -> pd.DataFrame:
    """Converts the dates in dataframe to the required upcoming day

    Args:
        df (pd.DataFrame): Input dataframe.
        date_col (str): Date column on which transformation to be applied.
        day (int, optional): Weekday to which the dates should be moved. Defaults to 6. Takes input 0-6 with 0 being Monday

    Returns:
        pd.DataFrame: Dataframe with dates moved to upcoming day.
    """
    df[date_col] = df[date_col] + pd.offsets.Week(n=0, weekday=day)
    return df

def move_nearest_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Moves dates in dataframe to end of month

    Args:
        df (pd.DataFrame): Input Dataframe
        date_col (str): Column name on which transformation to be applied.
    
    Returns:
        pd.DataFrame: Dataframe with dates moved to upcoming day.
    """

    df[date_col] = df[date_col] +pd.offsets.MonthEnd(0)
    return df


    
def indentify_outliers(df : pd.DataFrame , columns_to_indentify :List , outlier_direction :int = 0) ->pd.DataFrame :
    """Returns Dataframe with boolean values of whether the value is oulier or not 

    Args:
        df (pd.DataFrame): Input Dataframe
        columns_to_indentify (List):  Columns on which oulier detection to be performed
        outlier_direction (int, optional): Direction in which outlier treatment to be performed.Acceptable inputs are {1,0,-1} .if 0 performes for both upper and lower bound.
                                            If 1 performes for upper bound . if -1 performs for lower bound . Defaults to 0.

    Returns:
        pd.DataFrame: dataframe with boolean values of outlier columns .
    """


    bounds = df[columns_to_indentify].quantile([0.25,0.75])
    IQR = bounds.diff().reset_index().drop("index",axis=1).drop(0,axis=0).rename(index = {1:"IQR"})
    bounds = pd.concat([bounds,IQR]).T
    bounds["lower_bound"] = bounds[0.25] - (1.5 * bounds["IQR"])
    bounds["upper_bound"] = bounds[0.75] + (1.5 * bounds["IQR"])

    bounds = bounds.T

    if outlier_direction == 0 :
        outliers = df[columns_to_indentify].apply(lambda x :  (x < bounds.loc["lower_bound",x.name]) | 
                                                (x > bounds.loc["upper_bound",x.name]))
    elif outlier_direction == 1 :
        outliers = df[columns_to_indentify].apply(lambda x :  (x > bounds.loc["upper_bound",x.name]))

    elif outlier_direction == -1 :
        outliers = df[columns_to_indentify].apply(lambda x :  (x < bounds.loc["lower_bound",x.name]))
    else :
        raise ValueError("Invalid outlier detection direction . Available options are 1,0,-1 .")


    outliers = pd.concat([df.loc[:,~df.columns.isin(columns_to_indentify)], outliers], axis=1)

    return outliers

def treat_outliers(df : pd.DataFrame , columns_to_indentify :List ,method : str = "remove" ,outlier_direction :int = 0) -> pd.DataFrame :
    """Returns dataframe without outliers . Can perform two types of outlier treatment . Either to remove the ouliers or to cap them

    Args:
        df (pd.DataFrame): Input Dataframe
        columns_to_indentify (List): Columns on which outlier treatment to be performed
        method (str, optional): Two methods availabe {"cap","remove"}. Defaults to "remove".
        outlier_direction (int, optional): Direction in which outlier treatment to be performed.if 0 performes for both upper and lower .
                                            If 1 performes for upper bound . if -1 performs for lower bound .Defaults to 0.


    Returns:
        pd.DataFrame: Dataframe without outliers
    """
    bounds = df[columns_to_indentify].quantile([0.25,0.75])
    IQR = bounds.diff().reset_index().drop("index",axis=1).drop(0,axis=0).rename(index = {1:"IQR"})
    bounds = pd.concat([bounds,IQR]).T
    bounds["lower_bound"] = bounds[0.25] - (1.5 * bounds["IQR"])
    bounds["upper_bound"] = bounds[0.75] + (1.5 * bounds["IQR"])

    bounds = bounds.T

    if method == "remove" :
        if outlier_direction == 0 :
            outliers = df[columns_to_indentify].apply(lambda x :  (x < bounds.loc["lower_bound",x.name]) | 
                                                (x > bounds.loc["upper_bound",x.name]))
        elif outlier_direction == 1 :
            outliers = df[columns_to_indentify].apply(lambda x :  (x > bounds.loc["upper_bound",x.name]))

        elif outlier_direction == -1 :
            outliers = df[columns_to_indentify].apply(lambda x :  (x < bounds.loc["lower_bound",x.name]))
        else :
            raise ValueError("Invalid outlier detection direction . Available options are 1,0,-1 .")
        
        df_treat = pd.concat([df.loc[:,~df.columns.isin(columns_to_indentify)], df_treat], axis=1).dropna()
        
    elif method == "cap" :
        df_treat = df.copy()
        for col in columns_to_indentify :
            if outlier_direction == 0 :
                df_treat[col] = np.where(df_treat[col]> bounds.loc["upper_bound",col], bounds.loc["upper_bound",col],
                        np.where(df_treat[col]< bounds.loc["lower_bound",col], bounds.loc["lower_bound",col],df_treat[col]))
            elif outlier_direction == 1 :
                df_treat[col] = np.where(df_treat[col]> bounds.loc["upper_bound",col], bounds.loc["upper_bound",col],df_treat[col])
            elif outlier_direction == -1 :
                df_treat[col] = np.where(df_treat[col]< bounds.loc["lower_bound",col], bounds.loc["lower_bound",col],df_treat[col])
            else :
                raise ValueError("Invalid outlier detection direction . Available options are 1,0,-1 .")

    else :
        raise ValueError("Invalid outlier treatment strategy. Available options are 'remove' and 'cap'.")
    
    return df_treat



def add_linear_trend(df : pd.DataFrame , date_col : str) -> pd.DataFrame :
    """Adds weekly linear trend column "trend" based on date column i.e as the weekly date increases the trend value increases

    Args:
        df (pd.DataFrame): Input Dataframe
        date_col (str): Date indentifier column

    Returns:
        pd.DataFrame: Input Dataframe with "trend" column
    """

    temp = pd.date_range(start=df[date_col].min(),end=df[date_col].max(),freq="W")

    temp = add_trend(pd.DataFrame({date_col:temp}),trend = "t")

    return df.merge(temp,how="left",on=date_col)


def add_monthly_trend(df : pd.DataFrame ,date_col :str ) -> pd.DataFrame :
    """Adds monthly trend column "trend" based on date column i.e as the month changes the the trend value increases

    Args:
        df (pd.DataFrame): Input dataframe
        date_col (str): Date column

    Returns:
        pd.DataFrame: Input dataframe with "trend" column
    """

    temp = pd.date_range(start=df[date_col].min(),end=df[date_col].max(),freq="W")
    temp = pd.DataFrame({date_col:temp})

    temp["X_YEAR"] = temp[date_col].dt.strftime("%Y")
    temp["X_MONTH"] = temp[date_col].dt.strftime("%m")

    min_year = temp["X_YEAR"].min()

    temp["trend"] = ( temp["X_YEAR"] -  min_year ) + temp["X_MONTH"]

    temp.drop(["X_YEAR","X_MONTH"],axis=1,inplace=True)

    return df.merge(temp,how="left",on=date_col)

def get_seasonality_column(data: pd.DataFrame, dv: str, date_col: str ,  decompose_method : str , granularity: str = "week" , seasonal_method : str = "week_avg" , threshold : List = None ) -> pd.DataFrame :
    """Add seasonality column.

    Args:
        data (pd.DataFrame): Input Dataset
        dv (str): Dependent variable name 
        date_col (str): Date variable name
        decompose_method (str) : Model to use when seasonal_method is "decompose" . Acceptable inputs are {“additive”, “multiplicative”}
        granularity (str, optional): Granularity of data in date column . Accepatable inputs {"week","month" }.Defaults to "week".
        seasonal_method (str, optional): Method by which seasonal column is created . Acceptable inputs are {"week_avg" ,"decompose" ,"custom"} .
                                        Defaults to "week_avg".
        threshold (List, optional): List with 2 elements containing the lower and upper threshold respectively . 
                                    Input required when seasonal_method set to "custom" .Defaults to None.
        

    Returns:
        pd.DataFrame: Dataset with the seasonality column.
    """
    
    if  seasonal_method == "week_avg" or seasonal_method == "custom":
        if granularity == "week" :
            data[granularity] = data[date_col].dt.week
        elif granularity == "month" :
            data[granularity] = data[date_col].dt.month
        else :
            raise ValueError("Invalid granularity . Accepatable values are week or month .")
        
        data_w = data.groupby(granularity, as_index=False).agg({dv: "mean"})
        avg = data[dv].mean()
        data_w[dv] = data_w[dv] / avg

        s_col_name = "s_index" + "_weekly" if granularity == "week" else "_monthly"
        data_w.rename(columns={dv: s_col_name}, inplace=True)

        data = data.merge(data_w, on=granularity, how="left")
        data.drop(granularity, axis=1, inplace=True)

        if seasonal_method == "custom" :
            if len(threshold) != 2 :
                raise ValueError("Wrong number of arguments for threshold . Required elements are 2 for lower and upper threshold respectively . ")
            data[s_col_name] = np.where(data[s_col_name] > threshold[1] , 1 , np.where(data[s_col_name] < threshold[0] ,-1,0))
        
    elif seasonal_method == "decompose" :
        if decompose_method != "additive" or "multiplicative" :
            raise ValueError("Wrong decompose method . Acceptable inputs are {“additive”, “multiplicative”}.")

        temp = seasonal_decompose(data[[dv,date_col]].set_index(date_col),model=decompose_method)

        temp = pd.DataFrame(temp.seasonal).reset_index()

        data = data.merge(temp , on=date_col , how="left") 
    
    else : 
        raise ValueError("Wrong seasonal method . Acceptable inputs are {“week_avg” ,“decompose” ,“custom”}.")
    
    return data


def s_curve_values (alpha : float , beta : float , x : float ) -> float :
    """Compute s-curve value for x given alpha and beta

    Args:
        alpha (float): s curve parameter
        beta (float): s curve parameter
        x (float): input value

    Returns:
        float: s-curve transformed x
    """

    return alpha * (1- np.exp(-1*beta*x))

def get_scurve_transform(df: pd.DataFrame , feature_config : pd.DataFrame ) -> pd.DataFrame:
    """Generate S-Curve column based on feature config dataframe

    Args:
        df (pd.DataFrame): Input Dataframe
        feature_config (pd.DataFrame): Dataframe with 3 columns {"Column Name","Alpha","Beta"} containing the column name on which
                                        the transform to be applied with its corresponding alpha and beta values respectively.

    Returns:
        pd.DataFrame: Input dataframe with S-Curve columns
    """

    for index,row in feature_config.iterrows() :
        new_col_name = row["Column Name"] + "_S_" + str(row["Alpha"]) + "_" + str(row["Beta"])
        df[new_col_name] = df[row["Column Name"]].apply(lambda val : s_curve_values(alpha=row["Alpha"],beta=row["Beta"],x=val))

    return df











    

    
