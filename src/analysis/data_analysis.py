import pandas as pd


class DataAnalysis:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    @property
    def columns_stats(self, transposed=True) -> pd.DataFrame:
        """
        Return Pandas DataFrame with columns statistics

        :param transposed: if True applies transposition to stats
        """
        return self.df.describe().T if transposed else self.df.describe()

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

    @property
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

    @property
    def unique_values(self, columns: list = []) -> pd.Series:
        """
        Return Pandas Series with unique values for each column of the
        dataframe

        :param columns: columns subset
        """
        columns_subset = columns if columns else self.df.columns
        return self.df[columns_subset].apply(lambda col: col.unique())

    @property
    def unique_values_count(self, columns: list = []) -> pd.Series:
        """
        Return Pandas Series with count of unique values for each column of the
        dataframe

        :param columns: columns subset
        """
        columns_subset = columns if columns else self.df.columns
        return self.df[columns_subset].apply(lambda col: col.nunique())

    def columns_values_subset(self, df: pd.DataFrame, columns: list) -> list:
        """
        Return columns of the df parameter which are subset of the columns
        of the object dataframe.

        Useful for applying ordinal encoding safely
        """
        return [
            col for col in columns if set(df[col]).issubset(set(self.df[col]))]
