import os

from vizpool.static import EDA, Evaluation

from .common import get_unique_filename


class Plots(object):
    def __init__(self, data, target, parent_dir, plots_dir):
        self.eda = EDA(data)
        self.eda_dir = os.path.join(
            parent_dir, plots_dir, "EDA", get_unique_filename("EDA"))
        os.makedirs(self.eda_dir, exist_ok=True)
        self.target = target

    def EDA(self):
        plt = self.eda.histogram()
        plt_path = os.path.join(self.eda_dir, "histogram")
        plt.savefig(plt_path)

        plt = self.eda.boxplot()
        plt_path = os.path.join(self.eda_dir, "boxplot")
        plt.savefig(plt_path)

        plt = self.eda.corr_heatmap(width=14, height=10)
        plt_path = os.path.join(self.eda_dir, "corr_heatmap")
        plt.savefig(plt_path)

        plt = self.eda.countplot()
        plt_path = os.path.join(self.eda_dir, "countplot")
        plt.savefig(plt_path)

        plt = self.eda.barplot(y=self.target)
        plt_path = os.path.join(self.eda_dir, "barplot")
        plt.savefig(plt_path)

        print("*****" * 13)
        print("EDA plots have been saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {self.eda_dir}\n")


class Eval:
    def __init__(self, y_val, parent_dir, plots_dir):
        self.y_val = y_val
        self.eval_dir = os.path.join(
            parent_dir, plots_dir, "Evaluation", get_unique_filename("Evaluation"))
        os.makedirs(self.eval_dir, exist_ok=True)

    def feature_importance(self, model, X_val):
        self.vizpool_eval = Evaluation(self.y_val)
        self.model_name = model['regressor'].__class__.__name__
        plt = self.vizpool_eval.feature_importance(model, X_val, pipeline=True)
        plt_path = os.path.join(
            self.eval_dir, f"{self.model_name}_feature_importance.png")
        try:
            plt.savefig(plt_path)
        except AttributeError:
            pass

    def residplot(self, y_predicted):
        plt = self.vizpool_eval.residplot(
            y_predicted=y_predicted, height=12, width=16)
        plt_path = os.path.join(
            self.eval_dir, f"{self.model_name}_residplot.png")
        try:
            plt.savefig(plt_path)
        except AttributeError:
            pass
