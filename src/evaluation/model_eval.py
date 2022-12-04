import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class ModelEvaluation:
    """

    Hyperparameters Tuning:
    Training loss should steadily decrease, steeply at first, and then more
    slowly until the slope of the curve reaches or approaches zero.
    If the training loss does not converge, train for more epochs.
    If the training loss decreases too slowly, increase the learning rate.
    Note that setting the learning rate too high may also prevent training
    loss from converging.
    If the training loss varies wildly (that is, the training loss jumps
    around), decrease the learning rate.
    Lowering the learning rate while increasing the number of epochs or the
    batch size is often a good combination.
    Setting the batch size to a very small batch number can also cause
    instability. First, try large batch size values. Then, decrease the batch
    size until you see degradation.
    For real-world datasets consisting of a very large number of examples, the
    entire dataset might not fit into memory. In such cases, you'll need to
    reduce the batch size to enable a batch to fit into memory.
    """

    # Function for comparing different approaches
    def score_dataset_1(X_train,
                        X_valid,
                        y_train,
                        y_valid,
                        model=XGBRegressor()):
        """
        Evaluate without cross validation
        """
        # model = RandomForestRegressor(n_estimators=10, random_state=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    @staticmethod
    def score_dataset_2(X, y, model=XGBRegressor(),
                        scoring='neg_mean_squared_error'):
        """
        Evaluate with cross validation
        """
        # Label encoding for categoricals
        #
        # Label encoding is good for XGBoost and RandomForest, but one-hot
        # would be better for models like Lasso or Ridge. The `cat.codes`
        # attribute holds the category levels.
        for colname in X.select_dtypes(["category"]):
            X[colname] = X[colname].cat.codes
        # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
        log_y = np.log(y)
        score = cross_val_score(
            model, X, log_y, cv=5, scoring=scoring,
        )
        score = -1 * score.mean()
        score = np.sqrt(score)
        return score
