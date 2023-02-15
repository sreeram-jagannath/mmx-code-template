import logging
import traceback
from typing import List

import pandas as pd


def remove_columns_with_all_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """function to remove columns which have all their values as zeros

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: modified dataframe
    """

    try:
        all_zero_col_filter = (df == 0).all(axis=0)
        df = df.loc[:, ~all_zero_col_filter]
        return df
    except:
        print(traceback.print_exc())


def add_index_indentifiers(
    df: pd.DataFrame, date_col: List, calender: pd.DataFrame = None
) -> pd.DataFrame:
    """Adds week,quarter,month,year indentifiers to the dataframe
    .If calender is provided appends it to the dataframe else uses the date column to extract idenifiers.

    Args:
        df (pd.DataFrame): Input Dataframe.
        date_col (List) :  List with Date column names
        calender (pd.DataFrame, optional): Custom calender dataframe. Defaults to None.
    Returns:
        pd.DataFrame: Modified Dataframe
    """

    if calender != None:
        df = pd.merge(df, calender, how="left", left_on=date_col[0], right_on=date_col[1])
        return df

    df["Year"] = df[date_col[0]].dt.strftime("%Y")
    df["Month"] = df[date_col[0]].dt.strftime("%B")
    df["Week"] = df[date_col[0]].dt.strftime("%W")
    df["Quarter"] = pd.PeriodIndex(df[date_col[0]], freq="Q")

    return df
