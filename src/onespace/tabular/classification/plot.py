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
        plt = self.eda.pie_bar(self.target)
        plt_path = os.path.join(self.eda_dir, "pie_bar")
        plt.savefig(plt_path)

        plt = self.eda.histogram(hue=self.target)
        plt_path = os.path.join(self.eda_dir, "histogram")
        plt.savefig(plt_path)

        plt = self.eda.boxplot(hue=self.target)
        plt_path = os.path.join(self.eda_dir, "boxplot")
        plt.savefig(plt_path)

        plt = self.eda.violinplot(hue=self.target)
        plt_path = os.path.join(self.eda_dir, "violinplot")
        plt.savefig(plt_path)

        plt = self.eda.corr_heatmap(width=14, height=10)
        plt_path = os.path.join(self.eda_dir, "corr_heatmap")
        plt.savefig(plt_path)

        plt = self.eda.pairplot(hue=self.target)
        plt_path = os.path.join(self.eda_dir, "pairplot")
        plt.savefig(plt_path)

        plt = self.eda.countplot()
        plt_path = os.path.join(self.eda_dir, "countplot")
        plt.savefig(plt_path)

        print("*****" * 13)
        print("EDA plots have been saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {self.eda_dir}\n")


class Eval:
    def __init__(self, y_val, target_names, parent_dir, plots_dir):
        self.y_val = y_val
        self.eval_dir = os.path.join(
            parent_dir, plots_dir, "Evaluation", get_unique_filename("Evaluation"))
        os.makedirs(self.eval_dir, exist_ok=True)
        self.target_names = target_names

    def confusion_matrix(self, y_predicted, model):
        self.model = model
        self.vizpool_eval = Evaluation(self.y_val)
        plt = self.vizpool_eval.confusion_matrix(y_predicted, self.target_names, normalize=True,
                                                 width=14, height=10, title=f"Confusion Matrix for {model}")
        plt_path = os.path.join(
            self.eval_dir, f"{self.model}_confusion_matrix_pct")
        try:
            plt.savefig(plt_path)
        except AttributeError:
            pass
        plt = self.vizpool_eval.confusion_matrix(y_predicted, self.target_names, normalize=False,
                                                 width=14, height=10, title=f"Confusion Matrix for {self.model}")
        plt_path = os.path.join(
            self.eval_dir, f"{self.model}_confusion_matrix_count")
        try:
            plt.savefig(plt_path)
        except AttributeError:
            pass

    def feature_importance(self, model, X_val):
        self.model_name = model['classifier'].__class__.__name__
        plt = self.vizpool_eval.feature_importance(model, X_val, pipeline=True)
        plt_path = os.path.join(
            self.eval_dir, f"{self.model_name}_feature_importance.png")
        try:
            plt.savefig(plt_path)
        except AttributeError:
            pass

    def auc_roc(self, X_val, models: list, model_names: list):
        plt = self.vizpool_eval.auc_roc_plot(X_val, models, model_names)
        plt_path = os.path.join(self.eval_dir, "auc_roc_plot.png")
        try:
            plt.savefig(plt_path)
        except AttributeError:
            pass

        print("\n")
        print("*****" * 13)
        print("Plots for model evaluation have been saved at the following location")
        print("*****" * 13)
        print(f"\n ==> {self.eval_dir}\n")
