import pandas as pd
from sklearn.feature_selection import (
    mutual_info_regression,
    mutual_info_classif
)


class FeatureEvaluation:

    @staticmethod
    def make_mi_scores(X, y, target_type, sort=True):
        """
        Compute mi score between features according to target type
        """
        X = X.copy()
        mutual_info = (
            mutual_info_regression if target_type.upper() == 'continuous'
            else mutual_info_classif
        )
        for colname in X.select_dtypes(["object", "category"]):
            X[colname], _ = X[colname].factorize()
        # All discrete features should now have integer dtypes
        discrete_features = [
            pd.api.types.is_integer_dtype(t) for t in X.dtypes]
        mi_scores = mutual_info(
            X, y, discrete_features=discrete_features, random_state=0)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        if sort:
            mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores
