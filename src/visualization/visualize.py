import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Visualization:

    @staticmethod
    def set_figure_size(width, height):
        plt.figure(figsize=(width, height))

    @staticmethod
    def set_title(title: str):
        plt.title(title)

    @staticmethod
    def set_xlabel(label: str):
        plt.xlabel(label)

    @staticmethod
    def set_ylabel(label: str):
        plt.ylabel(label)

    @staticmethod
    def plot_line_chart(df: pd.DataFrame):
        sns.lineplot(data=df)

    @staticmethod
    def plot_bar_chart(xcolumn: pd.Series, ycolumn: pd.Series):
        sns.barplot(x=xcolumn, y=ycolumn)

    @staticmethod
    def plot_heatmap(df: pd.DataFrame, show_cell_values: bool = True):
        sns.heatmap(data=df, annot=show_cell_values)

    @staticmethod
    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    @staticmethod
    def plot_variance(pca, width=8, dpi=100):
        # Create figure
        fig, axs = plt.subplots(1, 2)
        n = pca.n_components_
        grid = np.arange(1, n + 1)
        # Explained variance
        evr = pca.explained_variance_ratio_
        axs[0].bar(grid, evr)
        axs[0].set(
            xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
        )
        # Cumulative Variance
        cv = np.cumsum(evr)
        axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
        axs[1].set(
            xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
        )
        # Set up figure
        fig.set(figwidth=width, dpi=dpi)
        return axs

    @staticmethod
    def corrplot(df, method="pearson", annot=True, **kwargs):
        sns.clustermap(
            df.corr(method),
            vmin=-1.0,
            vmax=1.0,
            cmap="icefire",
            method="complete",
            annot=annot,
            **kwargs,
        )

    def plot_regression_model(trained_weight, trained_bias, feature, label):
        """
            WARNING: TENSORFLOW
            Plot the trained model against the training feature and label.
        """

        # Label the axes.
        plt.xlabel("feature")
        plt.ylabel("label")

        # Plot the feature values vs. label values.
        plt.scatter(feature, label)

        # Create a red line representing the model. The red line starts
        # at coordinates (x0, y0) and ends at coordinates (x1, y1).
        x0 = 0
        y0 = trained_bias
        x1 = feature[-1]
        y1 = trained_bias + (trained_weight * x1)
        plt.plot([x0, x1], [y0, y1], c='r')

        # Render the scatter plot and the red line.
        plt.show()

    def plot_regression_loss_curve(epochs, rmse):
        """
            WARNING: TENSORFLOW
            Plot the loss curve, which shows loss vs. epoch.
        """

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Root Mean Squared Error")

        plt.plot(epochs, rmse, label="Loss")
        plt.legend()
        plt.ylim([rmse.min()*0.97, rmse.max()])
        plt.show()
