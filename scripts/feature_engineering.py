import pandas as pd
from typing import List


def add_holidays(data: pd.DataFrame, holidays: pd.DataFrame, date_col: List) -> pd.DataFrame:
    """
    Adds holidays Column to the dataframe
    :param data: Input dataframe.
    :param holidays: Dataframe with holidays in date format.
    :param date_col: List with Date column names
    :return: Input dataframe with "is_holiday" column
    """
    holidays["is_holidays"] = 1
    holidays = move_nearest_day(holidays, date_col="Week")
    result = pd.merge(data, holidays, how="left", left_on=date_col[0], right_on=date_col[1])
    result["is_holidays"] = result["is_holidays"].fillna(0)
    return result


def move_nearest_day(df: pd.DataFrame, date_col: str, day: int = 6) -> pd.DataFrame:
    """
    Converts the dates in dataframe to the upcoming day

    :param df: input dataframe.
    :param date_col: date column on which transformation to be applied.
    :param day: weekday to which the dates should be moved. Default is Sunday. Takes input 0-6 with 0 being Monday
    :return: dataframe with dates moved to upcoming day.
    """
    df[date_col] = df[date_col] + pd.offsets.Week(n=0, weekday=day)
    return df
