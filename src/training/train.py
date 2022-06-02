from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


class Training:

    @staticmethod
    def train_pipeline(preprocessor, model, X_train, y_train):
        """
        Train using pipeline
        """
        my_pipeline = Pipeline(
            steps=[('preprocessor', preprocessor),
                   ('model', model)])

        # Preprocessing of training data, fit model
        my_pipeline.fit(X_train, y_train)

    def cross_validation_score(
     pipeline, X, y, cv=5, scoring='neg_mean_absolute_error'):
        """
        Cross validation score
        """
        return cross_val_score(
            pipeline, X, y, cv=cv, scoring=scoring)
