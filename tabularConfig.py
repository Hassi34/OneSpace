#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "tabular_data"
csv_file_name = "insurance.csv"
target_column ="charges"
autopilot = False    # True if you want to automatically configure everything for the training job and run the job without user interaction else, False.  
eda = False          # Exploratory Data Analysis (EDA)
metrics = 'r2_score' # selection_for_classificaton = ['accuracy', 'f1_score', 'recall', 'precision']
                     # selection_for_regression = ['r2_score', 'mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error',
                     # 'median_absolute_error', 'explained_variance_score']
validation_split = 0.20
scaler = "RobustScaler" # available_selections = ['MinMaxScaler', 'StandardScaler', 'MaxAbsScaler', 'RobustScaler']
imputer = "SimpleImputer" # available_selections = ['KNNImputer', ''SimpleImputer']
PloynomialFeatures = False
remove_outliers = True
handle_imbalance = True # Only applicable to the classification problems.
pca = True               # Principal Component Analysis (PCA).
feature_selection = True  # This will use recursive feature elimination (RFE)
#--------------------------------------------------------------------------------------------------------------------------------
# Artifacts (Directory names to store the results & resources, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
project_name = 'Titanic Project1.0'
artifacts_dir = "Artifacts"
pipelines_dir = "Pipelines"
plots_dir = "Plots"
model_name = "my_test_model"
experiment_name = "light model testing"

#--------------------------------------------------------------------------------------------------------------------------------
# Logs (Directory names to record logs, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
logs_dir = "Logs"
csv_logs_dir = "CSV Logs"
csv_logs_file = "csv_logs_file"
comments = "making comparision for optimizers"
executed_by = 'hasanain'

#--------------------------------------------------------------------------------------------------------------------------------
# Database Integration
#--------------------------------------------------------------------------------------------------------------------------------
# Please Note that before making any change in this section, create a .env file and store the mongo db connection string or MySQL credentials in the environment variables 
# Guideline for creating .env is available on project description main page
from dotenv import load_dotenv
load_dotenv()
db_integration_mysql = False
db_integration_mongodb = False 