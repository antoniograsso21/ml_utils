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
