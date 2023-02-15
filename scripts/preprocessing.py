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
        df = df.loc[(df != 0).any(axis=1)]
        return df
    except:
        print(traceback.print_exc())




