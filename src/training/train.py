import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


class Training:

    @staticmethod
    def train_model(model, feature, label, epochs, batch_size):
        """
            WARNING: TENSORFLOW
            Train the model by feeding it data.
        """

        # Feed the feature values and the label values to the
        # model. The model will train for the specified number
        # of epochs, gradually learning how the feature values
        # relate to the label values.
        history = model.fit(x=feature,
                            y=label,
                            batch_size=batch_size,
                            epochs=epochs)

        # Gather the trained model's weight and bias.
        trained_weight = model.get_weights()[0]
        trained_bias = model.get_weights()[1]

        # The list of epochs is stored separately from the
        # rest of history.
        epochs = history.epoch

        # Gather the history (a snapshot) of each epoch.
        hist = pd.DataFrame(history.history)

        # Specifically gather the model's root mean
        # squared error at each epoch.
        rmse = hist["root_mean_squared_error"]

        return trained_weight, trained_bias, epochs, rmse

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
