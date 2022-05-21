import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from sklearn.impute import SimpleImputer


class DataframeAnalysis:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def columns_stats(self) -> pd.DataFrame:
        return self.df.describe().T

    @property
    def columns_numerical(self) -> list:
        """
        Return numerical columns, i.e. dtype == 'number'

        :rtype: list
        """
        return list(self.df.select_dtypes(include='number').columns)

    @property
    def columns_categorical(self) -> list:
        """
        Return categorical columns, i.e. dtype == 'object'

        :rtype: list
        """
        return list(self.df.select_dtypes(include='object').columns)

    @property
    def columns_with_missing(self) -> list:
        """
        Return columns with missing values

        :rtype: list
        """
        return [col for col in self.df.columns if self.df[col].isnull().any()]

    def columns_missing_count(self, sort=True) -> pd.Series:
        """
        Return Pandas Series with missing values count for columns with
        missing values

        :param sort: if True sort values in descending order

        :rtype: pd.Series
        """
        columns_missing_count = (
            self.df[self.columns_with_missing].isnull().sum())
        if sort:
            columns_missing_count = columns_missing_count.sort_values(
                ascending=False)
        return columns_missing_count


class PreProcessing:

    @staticmethod
    def fillna_type(df: pd.DataFrame, number_val=0, category_val='None'):
        df_copy = df.copy(deep=True)
        for name in df_copy.select_dtypes("number"):
            df_copy[name] = df_copy[name].fillna(number_val)
        for name in df.select_dtypes("category"):
            df_copy[name] = df_copy[name].fillna(category_val)
        return df_copy

    @staticmethod
    def encode_type(df: pd.DataFrame, columns_nominal,
                    columns_values_ordinal):
        df_copy = df.copy(deep=True)

        # Nominal categories
        for name in columns_nominal:
            df_copy[name] = df_copy[name].astype("category")
            # Add a None category for missing values
            if "None" not in df[name].cat.categories:
                df_copy[name].cat.add_categories("None", inplace=True)

        # Ordinal categories
        for name, levels in columns_values_ordinal.items():
            df_copy[name] = df_copy[name].astype(
                CategoricalDtype(levels, ordered=True))
        return df_copy
