import numpy as np
import pandas as pd
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from analysis.data_analysis import DataAnalysis


class PreProcessing:

    def __init__(self) -> None:
        self.simple_imputer = SimpleImputer()
        self.ordinal_encoder = OrdinalEncoder()
        self.one_hot_encoder = OneHotEncoder(
            handle_unknown='ignore', sparse=False)

    @staticmethod
    def fillna_type(df: pd.DataFrame, number_val=0, category_val='None'):
        df_copy = df.copy(deep=True)
        for name in df_copy.select_dtypes("number"):
            df_copy[name] = df_copy[name].fillna(number_val)
        for name in df.select_dtypes("category"):
            df_copy[name] = df_copy[name].fillna(category_val)
        return df_copy

    @staticmethod
    def label_encode(df: pd.DataFrame):
        df_copy = df.copy()
        for colname in df_copy.select_dtypes(["category"]):
            df_copy[colname] = df_copy[colname].cat.codes
            # df_copy[colname] = df_copy[colname].factorize()
        return df_copy

    @staticmethod
    def encode_type(df: pd.DataFrame, columns_nominal,
                    columns_values_ordinal):
        df_copy = df.copy()

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

    def simple_impute(self, X_train, X_valid, track=False):
        """
        Simple imputation

        :param track: if True keeps track of which values were imputed
        """
        # Imputation
        X_train_copy, X_valid_copy = X_train.copy(), X_valid.copy()

        if track:
            da_train = DataAnalysis(X_train)
            # Make new columns indicating what will be imputed
            for col in da_train.columns_with_missing:
                X_train_copy[col + '_was_missing'] = X_train_copy[col].isnull()
                X_valid_copy[col + '_was_missing'] = X_valid_copy[col].isnull()

        imputed_X_train = pd.DataFrame(
            self.simple_imputer.fit_transform(X_train_copy))
        imputed_X_valid = pd.DataFrame(
            self.simple_imputer.transform(X_valid_copy))

        # Imputation removed index and column names; put them back
        imputed_X_train.index = X_train.columns
        imputed_X_train.columns = X_train.index
        imputed_X_valid.index = X_valid.columns
        imputed_X_valid.columns = X_valid.index

        return imputed_X_train, imputed_X_valid

    def encode_columns_ordinal(self,
                               X_train: pd.DataFrame,
                               X_valid: pd.DataFrame,
                               columns_ordinal):
        """
        Encode ordinal columns
        """
        X_train_copy, X_valid_copy = X_train.copy(), X_valid.copy()
        X_train_copy[columns_ordinal] = self.ordinal_encoder.fit_transform(
            X_train[columns_ordinal])
        X_valid_copy[columns_ordinal] = self.ordinal_encoder.transform(
            X_valid[columns_ordinal])

        return X_train_copy, X_valid_copy

    def one_hot_encode_columns_nominal(self,
                                       X_train: pd.DataFrame,
                                       X_valid: pd.DataFrame,
                                       columns_nominal):
        """
        One hot encoding nominal columns
        """
        OH_cols_train = pd.DataFrame(
            self.one_hot_encoder.fit_transform(X_train[columns_nominal]))
        OH_cols_valid = pd.DataFrame(
            self.one_hot_encoder.transform(X_valid[columns_nominal]))

        # One-hot encoding removed index; put it back
        OH_cols_train.index = X_train.index
        OH_cols_valid.index = X_valid.index

        # Remove categorical columns (will replace with one-hot encoding)
        num_X_train = X_train.drop(columns_nominal, axis=1)
        num_X_valid = X_valid.drop(columns_nominal, axis=1)

        # Add one-hot encoded columns to numerical features
        OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
        OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

        return OH_X_train, OH_X_valid

    @staticmethod
    def pipeline_preproc(columns_numerical,
                         columns_categorical,
                         strat_num='constant',
                         strat_cat='most_frequent'):
        """
        Preprocessor with Pipeline
        """
        # TODO: use DataAnalaysis class to select columns
        # Preprocessing for numerical data
        transformer_numerical = SimpleImputer(strategy=strat_num)

        # Preprocessing for categorical data
        transformer_categorical = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=strat_cat)),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', transformer_numerical, columns_numerical),
                ('cat', transformer_categorical, columns_categorical)
            ])

        return preprocessor

    @staticmethod
    def cluster_labels(df, features, n_clusters=20):
        """
        Apply Kmeans
        """
        X = df.copy()
        X_scaled = X.loc[:, features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
        X_new = pd.DataFrame()
        X_new["Cluster"] = kmeans.fit_predict(X_scaled)
        return X_new

    @staticmethod
    def cluster_distance(df, features, n_clusters=20):
        """
        Apply Kmeans
        """
        X = df.copy()
        X_scaled = X.loc[:, features]
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
        kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
        X_cd = kmeans.fit_transform(X_scaled)
        # Label features and join to dataset
        X_cd = pd.DataFrame(
            X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
        )
        return X_cd

    @staticmethod
    def apply_pca(X, standardize=True):
        # Standardize
        if standardize:
            X = (X - X.mean(axis=0)) / X.std(axis=0)
        # Create principal components
        pca = PCA()
        X_pca = pca.fit_transform(X)
        # Convert to dataframe
        component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=component_names)
        # Create loadings
        loadings = pd.DataFrame(
            pca.components_.T,  # transpose the matrix of loadings
            columns=component_names,  # so the columns are the principal components
            index=X.columns,  # and the rows are the original features
        )
        return pca, X_pca, loadings


class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        """
        """
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        """
        """
        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded
