import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import pandas as pd
import numpy as np


class ChartMaker:
    default_path = pathlib.PurePath("files", "charts")

    def __init__(self, data, title, cm="Blues_d", palette="Blues_d", style="whitegrid"):
        self.data = data
        self.chart_style = style
        self.palette = palette
        self.cm = sns.color_palette(cm)
        self.title = title
        self.plot = plt
        self.figure = plt.figure()

    def get_correlation_matrix(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.corr().abs()

    def set_sns_theme_style(self):
        sns.set_theme(style=self.chart_style)

    # https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap
    def create_sns_heatmap(self, matrix):
        sns.heatmap(matrix, cmap=self.cm)

    # https://seaborn.pydata.org/generated/seaborn.barplot.html?highlight=barplot#seaborn.barplot
    def create_sns_barplot(self, y_title: str):
        sns.barplot(x=self.data.index, y=y_title, data=self.data, palette=self.palette)

    def set_figure_title(self, title: str, fontsize=9):
        self.figure.suptitle(title)

    def set_plot_title(self, title: str):
        self.plot.title(title)

    def save_plot_figure(self, filename: str):
        save_final = pathlib.PurePath(self.default_path, filename)
        self.plot.savefig(save_final)

    def set_plot_figure_fize(self, x: int, y: int):
        self.figure.figsize(x, y)

    def plot_show_figures(self):
        self.plot.show()

    def set_plot_ylim(self, y_range_start: int, y_range_stop: int):
        # range allowed on y-axis
        self.plot.ylim(y_range_start, y_range_stop)

    def set_plot_ylabel(self, y_label: str):
        self.plot.ylabel(y_label)

    def set_yticks(self, start: int, stop: int, step: float):
        self.plot.yticks(np.arange(start, stop, step=step))
