import logging
import traceback
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(level=logging.INFO)


def log_transformation(columns_to_transform: List, data: pd.DataFrame) -> pd.DataFrame:
    """Get the list of columns from the user to transform and replace those
    columns with log(x + 1)

    Args:
        columns_to_transform (List): list of the columns to apply log(1 + x) transformation
        data (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: transformed data which has log values in the columns.
    """

    for col in columns_to_transform:
        try:
            data[col] = np.log1p(data[col])

        except KeyError as e:
            logging.error(f"Column {e} not present in the data.")

        except Exception as e:
            logging.error(traceback.print_exc())
            continue

    return data


def scale_columns(df: pd.DataFrame, columns: list, strategy: str) -> pd.DataFrame:
    """
    Scales the specified columns in a Pandas dataframe using the specified strategy.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    columns (list): The list of column names to be scaled.
    strategy (str): The scaling strategy to be used. Available options are 'custom', 'MinMaxScaler', and 'StandardScaler'.

    Returns:
    pd.DataFrame: The input dataframe with the specified columns scaled according to the specified strategy.
    """
    if strategy == "custom":
        for col in columns:
            df[col] = df[col] / df[col].max()

    elif strategy == "MinMaxScaler":
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])

    elif strategy == "StandardScaler":
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        raise ValueError(
            "Invalid scaling strategy. Available options are 'custom', 'MinMaxScaler', and 'StandardScaler'."
        )

    return df


def generate_cpm(
    df: pd.DataFrame, data_dict: pd.DataFrame, granularity: str = "quarter"
) -> pd.DataFrame:
    """Generate CPM for marketing channels

    Args:
        df (pd.DataFrame): Input Dataframe
        data_dict (pd.DataFrame): Data Dictionary with spend and activity variables
        granularity (str, optional): Level at which CPM to be generated either quarter or year. Defaults to "quarter".

    Returns:
        pd.DataFrame: Dataframe with CPM for marketing channels.
    """

    if granularity == "year":
        temp = df.groupby("Year").sum()
    elif granularity == "quarter":
        temp = df.groupby(["Year", "Quarter"]).sum()
    elif granularity == "month":
        temp = df.groupby(["Year", "Month"]).sum()
    else:
        temp = df.groupby(["Year", "Week"]).sum()

    for index, row in data_dict.iterrows():
        new_col = row["Variable Description"] + "_CPM"
        temp[new_col] = (temp[row["Spend"]] / temp[row["Activity"]]) * 1000

    return temp.filter(regex="CPM_$")


def find_percentile(
    df: pd.DataFrame, lower_percentile: int = 0, upper_percentile: int = 95
) -> pd.DataFrame:
    """Computes the percentile value for each column in the dataframe .

    Args:
        df (pd.DataFrame): Input Dataframe .
        lower_percentile (int, optional): Lower cutoff percentile. Defaults to 0.
        upper_percentile (int, optional): Upper cutoff percentile. Defaults to 95.

    Returns:
        pd.DataFrame: Dataframe with cutoff values for lower and upper percentile
    """
    return df.quantile([lower_percentile / 100, upper_percentile / 100])


def treat_outliers(
    df: pd.DataFrame, id: List, lower_percentile: int = 0, upper_percentile: int = 95
) -> pd.DataFrame:
    """Detects outliers and returns dataframe without outliers based on the percentile range .

    Args:
        df (pd.DataFrame): Input Dataframe
        id (List): List of index variables
        lower_percentile (int, optional): Lower cutoff percentile. Defaults to 0.
        upper_percentile (int, optional): Upper cutoff percentile. Defaults to 95.

    Returns:
        pd.DataFrame:  Dataframe with outlier removed .
    """
    # filter out id variables to find quantile values of each columns
    filt_df = df.loc[:, ~df.columns.isin(id)]

    quant_df = find_percentile(filt_df, lower_percentile, upper_percentile)

    # Filter out outlier values
    filt_df = filt_df.apply(
        lambda x: x[
            (x > quant_df.loc[lower_percentile / 100, x.name])
            & (x < quant_df.loc[upper_percentile / 100, x.name])
        ],
        axis=0,
    )

    # combine dataframe  and outlier values

    filt_df = pd.concat([df.loc[:, id], filt_df], axis=1).dropna()

    return filt_df
