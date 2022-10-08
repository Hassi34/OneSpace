#params
data_dir = "tabular_data"
csv_file_name = "titanic.csv"
target_column ="Survived"
autopilot = False
eda = True           # Exploratory Data Analysis (EDA)
metrics = 'f1_score' # available_selections = ['accuracy', 'f1_score', 'recall', 'precision']
validation_split = 0.20
scaler = "StandardScaler" # available_selections = ['MinMaxScaler', 'StandardScaler', 'MaxAbsScaler', 'RobustScaler']
imputer = "SimpleImputer" # available_selections = ['KNNImputer', ''SimpleImputer']
PloynomialFeatures = False
remove_outliers = True
handle_imbalance = False
pca = False               # Principal Component Analysis (PCA)
feature_selection = False

# Artifacts (Directory names to store the artifacts, can be customized according to the user requirements)
artifacts_dir = "Artifacts"
pipelines_dir = "Pipelines"
plots_dir = "Plots"
model_name = "my_test_model"
experiment_name = "light model testing"
plot_name = "results_plot"

#logs (Directory names to record logs, can be customized according to the user requirements)
logs_dir = "Logs"
csv_logs_dir = "CSV Logs"
csv_logs_file = "csv_logs_file"
comments = "making comparision for optimizers"
executed_by = 'hasnain'