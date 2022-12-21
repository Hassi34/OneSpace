<p align="center">
    <b>
        <h1 align="center">
            <em>♾️ OneSpace ♾️</em>
        </h1>
    </b>
</p>
<p align="center">
    <em>A high-level Python framework to automate the project lifecycle of Machine and Deep Learning Projects</em>
</p>

<p align="center">
    <a href="https://pypi.org/project/onespace">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/onespace?color=g">
    </a>
    <a href="https://pypi.org/project/onespace">
        <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/onespace?color=g">
    </a>
    <a href="https://pepy.tech/project/onespace">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/onespace?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=Downloads">
    </a>
    <a href="https://github.com/hassi34/onespace">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/hassi34/onespace/main?color=g">
    </a>
    <a href="https://github.com/Hassi34/OneSpace/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/hassi34/onespace?color=g">
    </a>
    <a href="https://github.com/hassi34/onespace/issues">
        <img alt="Issue Tracking" src="https://img.shields.io/badge/issue_tracking-github-brightgreen.svg">
    </a>
    <a href="https://github.com/hassi34/onespace/issues">
        <img alt="Open Issues" src="https://img.shields.io/github/issues/hassi34/onespace">
    </a>
    <a href="https://pypi.org/project/onespace">
        <img alt="Github Actions Status" src="https://img.shields.io/github/actions/workflow/status/hassi34/onespace/cicd.yml?branch=main">
    </a>
    <a href="https://pypi.org/project/onespace">
        <img alt="Github Actions Status" src="https://img.shields.io/github/checks-status/hassi34/onespace/main?color=gree">
    </a>
    <a href="https://pypi.org/project/onespace">
        <img alt="Code Size" src="https://img.shields.io/github/languages/code-size/hassi34/onespace?color=g">
    </a>
    <a href="https://pypi.org/project/onespace">
        <img alt="Repo Size" src="https://img.shields.io/github/repo-size/hassi34/onespace?color=g">
    </a>
</p>

## Overview
``OneSpace`` enables you to train high performace production ready Machine Learning And Deep Learning Models effortlessly with less than five lines of code. ``OneSpace`` provides you a unified workspace so you can work on all kinds of Machine Learning and Deep Learning problem statements without having to leave your workspace environment. It don't just run everything as a black box to present you results, instead it makes the complete model training process easily explainable using artifacts, logs and plots. ``OneSpace`` also provides you the optional parameters to pass database credentials which will create a table with the project name in the database of your choice and will log all the training activities and the parameters in the database.<br>
Following are the major contents to follow, you can jump to any section:

>   1. [Installation](#install-)
>   2. [Usage](#use-)
>   3. [Getting Started with OneSpace (Tutorials)](#tutorials-)<br>
>      - [Tabular](#tabular-)<br>
>        - [Training a Regression Model With Tabular Data](#reg-)<br>
>        - [Training a Classification Model With Tabular Data](#clf-)<br>
>      - [Computer Vision](#cv-)<br>
>        - [Training an Image Classification Model with OneSpace(Tensorflow)](#tf-imgclf)<br>
>        - [Training an Image Classification Model with OneSpace(PyTorch)](#pytorch-imgcls)<br>
>   4. [Contributing](#contributing-)
>   5. [Conclusion](#conclusion-)
### 🔗 Project Link
**``OneSpace``** is being distributed through PyPI. Check out the PyPI Package [here](https://pypi.org/project/onespace/)


### 1. **Installation**<a id='install-'></a>
To avoid any dependency conflict, make sure to create a new Python virtual environment and then install via Pip!
```bash
pip install onespace
```
### 2. **Usage**<a id='use-'></a>
Get the **[config.py](https://github.com/Hassi34/onespace/blob/main/tabularConfig.py)** and **[training.py](https://github.com/Hassi34/onespace/blob/main/training.py)** files ready. You can get it from this repo or from the following tutorials section. 
- **Prepare ``training.py``**
```bash
import config # In case, you renamed config.py to something else, make sure to use the same name here
from onespace.tabular.regression import Experiment # Importing Experiment class to train a regression model

def training(config):
    exp = Experiment(config)
    exp.run_experiment()

if __name__ == "__main__":
    training(config)
```
* Now run the following command in your terminal to start the training job:
```bash
python training.py
```
Please following along with these ``quick tutorials``👇 to understand the complete setup and training process.
### 3. **Getting Started with OneSpace**<a id='tutorials-'></a>

* Ensure you have [Python 3.7+](https://www.python.org/downloads/) installed.

* Create a new Python conda environment for the OneSpace:

```
$ conda create -n venv  # create venv
$ conda activate venv  # activate venv
$ pip install onespace # install onespace
```

* Create a new Python virtual environment with pip for the OneSpace:
```
$ python3 -m venv venv  # create venv
$ . venv/bin/activate   # activate venv
$ pip install onespace # install onespace
```
#### **Tabular**<a id='tabular-'></a>
In this section, you will learn to train a Machine Learning model with ``OneSpace``:
#### **Training a Regression Model With Tabular Data:**<a id='reg-'></a>
First step is to setup the initial directory. This should present a following tree structure at the beginning of the training:
```bash
│   .env
│   tabularConfig.py
│   training.py
│
└───data_folder
        insurance.csv
``` 
Now let's discuss these files one by one 
* ``.env``<br>
This file should only be used when the database integration is required. You will be required to hold the database credentials in ``.env`` as shown below
```bash
MONGO_CONN_STR="Connection String for MongoDB"
MYSQL_HOST="database_host(could be an ip or domain)"
MYSQL_USER="database_user_name"
MYSQL_PASS="database_password"
MYSQL_DB="database_name"
```
* ``tabularConfig.py``<br>
This file is used to setup the configuration for Experiment. It has standard format as shown below. Parameters could be updated according to the training job requirements
```python

#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "data_folder"
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
remove_outliers = False
handle_imbalance = False # Only applicable to the classification problems.
pca = False               # Principal Component Analysis (PCA).
feature_selection = False  # This will use recursive feature elimination (RFE)
#--------------------------------------------------------------------------------------------------------------------------------
# Artifacts (Directory names to store the results & resources, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
project_name = 'Insurance Project'
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
``` 
* ``training.py``<br>
```python
import tabularConfig
from onespace.tabular.regression import Experiment


def training(config):
    exp = Experiment(config)
    exp.run_experiment()


if __name__ == "__main__":
    training(tabularConfig)
```
Run the following command in the terminal to start the training job
```bash
python training.py
```
**Now let the ``OneSpace`` take care of your end-to-end model training and evaluation process.**<br>
After the training job is completed, the directories in the workspace should look like as follows:
```bash
│   .env
│   tabularConfig.py
│   training.py
│
├───cachedir
│   └───joblib
│       └───sklearn
│           └───pipeline
│               └───_fit_transform_one
├───data_folder
│       insurance.csv
│
├───Tabular
│   └───Regression
│       └───Insurance Project
│           ├───Artifacts
│           │   └───Pipelines_on_20221012_at_004113
│           │           GradientBoostingRegressor_on_20221012_at_004124.pkl
│           │           LGBMRegressor_on_20221012_at_004126.pkl
│           │           StackingRegressor_on_20221012_at_004128.pkl
│           │           VotingRegressor_on_20221012_at_004128.pkl
│           │
│           ├───Logs
│           │   ├───Final
│           │   │       csv_logs_file.csv
│           │   │
│           │   ├───Model Comparision
│           │   │       modelsComparisionAfterTuning_on_20221012_at_004128..csv
│           │   │       modelsComparisionBeforTuning_on_20221012_at_004116..csv
│           │   │
│           │   └───Preprocessing
│           │           preprocessed_on_20221012_at_004113..csv
│           │
│           └───Plots
│               ├───EDA
│               │   └───EDA_on_20221012_at_004111
│               │           barplot.png
│               │           boxplot.png
│               │           corr_heatmap.png
│               │           countplot.png
│               │           histogram.png
│               │
│               └───Evaluation
│                   └───Evaluation_on_20221012_at_004120
│                           GradientBoostingRegressor_feature_importance.png
│                           GradientBoostingRegressor_residplot.png
│                           LGBMRegressor_feature_importance.png
│                           LGBMRegressor_residplot.png
│
└───__pycache__
        tabularConfig.cpython-310.pyc

```

#### **Training a Classification Model With Tabular Data** :<a id='clf-'></a>
First step is to setup the initial directory. This should present a following tree structure at the beginning of the training:
```bash
│   .env
│   tabularConfig.py
│   training.py
│
└───data_folder
        titanic.csv
``` 
Now let's discuss these files one by one 
* ``.env``<br>
This file should only be used when the database integration is required. You will be required to hold the database credentials in ``.env`` as shown below
```bash
MONGO_CONN_STR="Connection String for MongoDB"
MYSQL_HOST="database_host(could be an ip or domain)"
MYSQL_USER="database_user_name"
MYSQL_PASS="database_password"
MYSQL_DB="database_name"
```
* ``tabularConfig.py``<br>
This file is used to setup the configuration for Experiment. It has standard format as shown below. Parameters could be updated according to the training job requirements
```python

#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "data_folder"
csv_file_name = "titanic.csv"
target_column ="Survived"
autopilot = False    # True if you want to automatically configure everything for the training job and run the job without user interaction else, False.  
eda = True          # Exploratory Data Analysis (EDA)
metrics = 'f1_score' # selection_for_classificaton = ['accuracy', 'f1_score', 'recall', 'precision']
                     # selection_for_regression = ['r2_score', 'mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error',
                     # 'median_absolute_error', 'explained_variance_score']
validation_split = 0.20
scaler = "RobustScaler" # available_selections = ['MinMaxScaler', 'StandardScaler', 'MaxAbsScaler', 'RobustScaler']
imputer = "SimpleImputer" # available_selections = ['KNNImputer', ''SimpleImputer']
PloynomialFeatures = False
remove_outliers = False
handle_imbalance = False # Only applicable to the classification problems.
pca = False               # Principal Component Analysis (PCA).
feature_selection = False  # This will use recursive feature elimination (RFE)
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
``` 
* ``training.py``<br>
```python
import tabularConfig
from onespace.tabular.classification import Experiment


def training(config):
    exp = Experiment(config)
    exp.run_experiment()


if __name__ == "__main__":
    training(tabularConfig)
```
Run the following command in the terminal to start the training job
```bash
python training.py
```
**Now let the ``OneSpace`` take care of your end-to-end model training and evaluation process.**<br>
After the training job is completed, the directories in the workspace should look like as follows:
```bash
│   .env
│   tabularConfig.py
│   training.py
│
├───cachedir
│   └───joblib
│       └───sklearn
│           └───pipeline
│               └───_fit_transform_one
├───data_folder
│       titanic.csv
│
├───Tabular
│   └───Classification
│       └───Titanic Project1.0
│           ├───Artifacts
│           │   └───Pipelines_on_20221012_at_003107
│           │           ExtraTreesClassifier_on_20221012_at_003128.pkl
│           │           RandomForestClassifier_on_20221012_at_003130.pkl
│           │           StackingClf_on_20221012_at_003136.pkl
│           │           VotingClf_on_20221012_at_003136.pkl
│           │
│           ├───Logs
│           │   ├───Final
│           │   │       csv_logs_file.csv
│           │   │
│           │   └───Preprocessing
│           │           preprocessed_on_20221012_at_003107..csv
│           │
│           └───Plots
│               ├───EDA
│               │   └───EDA_on_20221012_at_003048
│               │           boxplot.png
│               │           corr_heatmap.png
│               │           countplot.png
│               │           histogram.png
│               │           pairplot.png
│               │           pie_bar.png
│               │           violinplot.png
│               │
│               └───Evaluation
│                   └───Evaluation_on_20221012_at_003124
│                           auc_roc_plot.png
│                           ExtraTreesClassifier_confusion_matrix_count.png
│                           ExtraTreesClassifier_confusion_matrix_pct.png
│                           ExtraTreesClassifier_feature_importance.png
│                           RandomForestClassifier_confusion_matrix_count.png
│                           RandomForestClassifier_confusion_matrix_pct.png
│                           RandomForestClassifier_feature_importance.png
│                           StackingClf_confusion_matrix_count.png
│                           StackingClf_confusion_matrix_pct.png
│                           VotingClf_confusion_matrix_count.png
│                           VotingClf_confusion_matrix_pct.png
│
└───__pycache__
        tabularConfig.cpython-310.pyc
```
#### **Computer Vision**<a id='cv-'></a>
This API enable users to create image classification models with Tensorflow or PyTorch.
More services for computer vision API are being developed.
#### **Training an Image Classification Model with OneSpace(Tensorflow)** :<a id='tf-imgclf'></a>
First step is to setup the initial directory. This should present a following tree structure at the beginning of the training:
```bash
│   .env
│   tensorflowConfig.py
│   training.py
│   
└───data_folder
    ├───train
    │   ├───ants
    │   │       0013035.jpg
    │   │       1030023514_aad5c608f9.jpg
    │   │       1095476100_3906d8afde.jpg
    │   │       1099452230_d1949d3250.jpg
    │   │       116570827_e9c126745d.jpg
    │   │
    │   └───bees
    │           1092977343_cb42b38d62.jpg
    │           1093831624_fb5fbe2308.jpg
    │           1097045929_1753d1c765.jpg
    │           1232245714_f862fbe385.jpg
    │           129236073_0985e91c7d.jpg
    │
    └───val
        ├───ants
        │       10308379_1b6c72e180.jpg
        │       1053149811_f62a3410d3.jpg
        │       1073564163_225a64f170.jpg
        │
        └───bees
                1032546534_06907fe3b3.jpg
                10870992_eebeeb3a12.jpg
                1181173278_23c36fac71.jpg



``` 
Now let's discuss these files one by one 
* ``.env``<br>
This file should only be used when the database integration is required. You will be required to hold the database credentials in ``.env`` as shown below
```bash
MONGO_CONN_STR="Connection String for MongoDB"
MYSQL_HOST="database_host(could be an ip or domain)"
MYSQL_USER="database_user_name"
MYSQL_PASS="database_password"
MYSQL_DB="database_name"
```
* ``tensorflowConfig.py``<br>
This file is used to setup the configuration for Experiment. It has standard format as shown below. Parameters could be updated according to the training job requirements
```python
#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "data_folder" # main directory for data
train_folder_name = "train" # name of the training folder. Pass "None" if there is no training and validation folder available
val_folder_name = "val"     # name of the validation folder pass "None" if there is no training and validation folder available
transfer_learning = False    # Pass "False" if you want to train the model from scratch
model_architecture = "MobileNetV3Small" # available_selections = ["VGG16", "MobileNetV3Large", "MobileNetV3Small", "DenseNet201", "EfficientNetV2L", "EfficientNetV2S", "ResNet50", "ResNet50V2", "ResNet152V2", "VGG19", "Xception"]
freeze_all = True           # Will freeze the weights for all the layers except for last one (Dense Layer)
freeze_till = None          # selection = [-1,-2,-3,-4,-5] => if you want to freeze weights until a specific layer, select any value from the list and observe the trainable prams
augmentation = False         # Pass True if the augmentation should be applied on the images. It helps with regularization.
epochs = 2                 # Number of Epochs
batch_size = 64
input_shape = (224, 224, 3) 
activation = "relu"         # available_selections = ["relu","selu", "swish", "elu", "gelu", "tanh"] => This param is used when there is no transferlearning involved and model is being trained from scratch 
activation_output = "softmax" # available_selections = ["softmax","sigmoid", "hard_sigmoid"]
loss_function = "binary_crossentropy" # available_selection = ["binary_crossentropy","categorical_crossentropy" , "sparse_categorical_crossentropy"]
metrics = ["AUC","Recall" ]   # available_selection = ["accuracy", "AUC", "Precision", "Recall" ]
lr_scheduler = 'InverseTimeDecay' #availabel_selection = ["InverseTimeDecay","CosineDecay", "ExponentialDecay", "CosineDecayRestarts", "PolynomialDecay"] # can be observed on tensorboard
optimizer = "Adam"          # available_selection = ["SGD", "Adam", "RMSprop", "Adadelta", "Adagrad", "Adamax", "Nadam", "Ftrl"]
validation_split = 0.20     # The proportion of data should be used for validation. This param won't be used if a seperate Validation Data Folder is being passed
es_patience = 5             # Early Stopping Patience

#--------------------------------------------------------------------------------------------------------------------------------
# Artifacts (Directory names to store the results & resources, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
project_name = 'AntsBeesProject_tf'
artifacts_dir = "Artifacts"
model_dir = "Models"
plots_dir = "Plots"
model_name = "my_test_model"
experiment_name = "Exp for Demo"
plot_name = "results_plot"
model_ckpt_dir = "Model Checkpoints"
callbacked_model_name = "model_ckpt"

#--------------------------------------------------------------------------------------------------------------------------------
# Logs (Directory names to record logs, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
logs_dir = "Logs"
tensorboard_root_log_dir = "Tensorboard Logs"
csv_logs_dir = "CSV Logs"
csv_logs_file = "cv_test_logs.csv"
comments = "Running a demo with transfer learning"
executed_by = 'Hasanain'

#--------------------------------------------------------------------------------------------------------------------------------
# Database Integration
#--------------------------------------------------------------------------------------------------------------------------------
# Please Note that before making any change in this section, create a .env file and store the mongo db connection string or MySQL credentials in the environment variables 
# Guideline for creating .env is available on project description main page
from dotenv import load_dotenv
load_dotenv()
db_integration_mysql = False
db_integration_mongodb = False 
``` 
* ``training.py``<br>
```python
import tensorflowConfig
from onespace.tensorflow.cv import Experiment


def training(config):
    exp = Experiment(config)
    exp.run_experiment()


if __name__ == "__main__":
    training(tensorflowConfig)
```
Run the following command in the terminal to start the training job
```bash
python training.py
```
**Now let the ``OneSpace`` take care of your end-to-end model training and evaluation process.**<br>
After the training job is completed, the directories in the workspace should look like as follows:
```bash
│   .env
│   tensorflowConfig.py
│   training.py
│   
├───ComputerVision
│   └───TensorFlow
│       └───AntsBeesProject_tf
│           ├───Artifacts
│           │   ├───Images
│           │   │   ├───Training Images
│           │   │   │       img_grid__on_20221012_at_090752.png
│           │   │   │       
│           │   │   └───Validation Images
│           │   │           img_grid__on_20221012_at_090752.png
│           │   │
│           │   ├───Models
│           │   │   ├───Model Checkpoints
│           │   │   │       model_ckpt__on_20221012_at_090755_.h5
│           │   │   │
│           │   │   └───TrainedModels
│           │   │           my_test_model__on_20221012_at_090800_.h5
│           │   │
│           │   └───Plots
│           │       ├───models
│           │       │       my_test_model__on_20221012_at_091731.png
│           │       │
│           │       └───results
│           │               results_plot__on_20221012_at_091731.png
│           │
│           └───Logs
│               ├───CSV Logs
│               │       cv_test_logs.csv
│               │
│               └───Tensorboard Logs
│                   └───logs__on_20221012_at_090755
│                       ├───train
│                       │       events.out.tfevents.1665540475.MY-LAPTOP.143340.0.v2
│                       │
│                       └───validation
│                               events.out.tfevents.1665540477.MY-LAPTOP.143430.1.v2
│
├───data_folder
│   ├───train
│   │   ├───ants
│   │   │       0013035.jpg
│   │   │       1030023514_aad5c608f9.jpg
│   │   │       1095476100_3906d8afde.jpg
│   │   │       1099452230_d1949d3250.jpg
│   │   │       116570827_e9c126745d.jpg
│   │   └───bees
│   │           1092977343_cb42b38d62.jpg
│   │           1093831624_fb5fbe2308.jpg
│   │           1097045929_1753d1c765.jpg
│   │           1232245714_f862fbe385.jpg
│   │           129236073_0985e91c7d.jpg
│   │
│   └───val
│       ├───ants
│       │       10308379_1b6c72e180.jpg
│       │       1053149811_f62a3410d3.jpg
│       │       1073564163_225a64f170.jpg
│       │
│       └───bees
│               1032546534_06907fe3b3.jpg
│               10870992_eebeeb3a12.jpg
│               1181173278_23c36fac71.jpg
│
└───__pycache__
        tensorflowConfig.cpython-310.pyc
```


#### **Training an Image Classification Model with OneSpace(PyTorch)** :<a id='pytorch-imgcls'></a>
First step is to setup the initial directory. This should present a following tree structure at the beginning of the training:
```bash
│   .env
│   pytorchConfig.py
│   training.py
│   
└───data_folder
    ├───train
    │   ├───ants
    │   │       0013035.jpg
    │   │       1030023514_aad5c608f9.jpg
    │   │       1095476100_3906d8afde.jpg
    │   │       1099452230_d1949d3250.jpg
    │   │       116570827_e9c126745d.jpg
    │   │
    │   └───bees
    │           1092977343_cb42b38d62.jpg
    │           1093831624_fb5fbe2308.jpg
    │           1097045929_1753d1c765.jpg
    │           1232245714_f862fbe385.jpg
    │           129236073_0985e91c7d.jpg
    │
    └───val
        ├───ants
        │       10308379_1b6c72e180.jpg
        │       1053149811_f62a3410d3.jpg
        │       1073564163_225a64f170.jpg
        │
        └───bees
                1032546534_06907fe3b3.jpg
                10870992_eebeeb3a12.jpg
                1181173278_23c36fac71.jpg
``` 
Now let's discuss these files one by one 
* ``.env``<br>
This file should only be used when the database integration is required. You will be required to hold the database credentials in ``.env`` as shown below
```bash
MONGO_CONN_STR="Connection String for MongoDB"
MYSQL_HOST="database_host(could be an ip or domain)"
MYSQL_USER="database_user_name"
MYSQL_PASS="database_password"
MYSQL_DB="database_name"
```
* ``pytorchConfig.py``<br>
This file is used to setup the configuration for Experiment. It has standard format as shown below. Parameters could be updated according to the training job requirements
```python
#--------------------------------------------------------------------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------------------------------------------------------------------
data_dir = "data_folder"
train_folder_name = "train"
val_folder_name = "val"
transfer_learning = True
model_architecture = "MobileNet_v3_small" # available_selections = ["AlexNet", "ConvnNeXt", "DenseNet121", "DenseNet201", "EfficientNet_b7", "EfficientNet_v2_s", "EfficientNet_v2_m", "EfficientNet_v2_l", "Wide_Resnet50_2",
                                         #"GoogleNet", "Inception_v3", "MnasNet0_5", "MnasNet1_3", "MobileNet_v2", "MobileNet_v3_large", "MobileNet_v3_small", "RegNet_y_32gf", "ResNet18",
                                         #"ResNet34", "ResNet50", "ResNet152", "ResNext101_32x8d", "ShuffleNet_v2_x1_5", "SqueezeNet1_0", "VGG11", "VGG13", "VGG16", "VGG19", "VisionTransformer"]
                                         
augmentation = False
epochs = 1
batch_size = 32
input_shape = (224, 224, 3)
lr_scheduler = "ExponentialLR"           # available_selections = ["OneCycleLR","StepLR", "LambdaLR", "ExponentialLR"]
optimizer = "RMSprop"                        # available_selections = ["SGD", "Adam", "Adadelta", "Adagrad", "RMSprop"]
validation_split = 0.20
grad_clip = 1                        # Should be a value ranging from 0.5 to 1.0 OR None
weight_decay = 0.001                 # Should be a value ranging from 0.5 to 1.0

#--------------------------------------------------------------------------------------------------------------------------------
# Artifacts (Directory names to store the results & resources, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
project_name = 'AntsBeensProject_pytorch'
artifacts_dir = "Artifacts"
model_dir = "Models"
plots_dir = "Plots"
model_name = "antbees_model"
experiment_name = "Demo Experiment"
plot_name = "results_plot"
model_ckpt_dir = "Model Checkpoints"
callbacked_model_name = "model_ckpt"

#--------------------------------------------------------------------------------------------------------------------------------
# Logs (Directory names to record logs, can be customized according to the user requirements)
#--------------------------------------------------------------------------------------------------------------------------------
logs_dir = "Logs"
tensorboard_root_log_dir = "Tensorboard Logs"
csv_logs_dir = "CSV Logs"
csv_logs_file = "cv_test_logs.csv"
comments = "This is a Demo"
executed_by = 'Hasanain'

#--------------------------------------------------------------------------------------------------------------------------------
# Database Integration
#--------------------------------------------------------------------------------------------------------------------------------
# Please Note that before making any change in this section, create a .env file and store the mongo db connection string or MySQL credentials in the environment variables 
# Guideline for creating .env is available on project description main page
from dotenv import load_dotenv
load_dotenv()
db_integration_mysql = False
db_integration_mongodb = False 
``` 
* ``training.py``<br>
```python
import pytorchConfig
from onespace.pytorch.cv import Experiment


def training(config):
    exp = Experiment(config)
    exp.run_experiment()


if __name__ == "__main__":
    training(pytorchConfig)
```
Run the following command in the terminal to start the training job
```bash
python training.py
```
**Now let the ``OneSpace`` take care of your end-to-end model training and evaluation process.**<br>
After the training job is completed, the directories in the workspace should look like as follows:
```bash
│   .env
│   pytorchConfig.py
│   training.py
│   
├───ComputerVision
│   └───PyTorch
│       └───AntsBeensProject_pytorch
│           ├───Artifacts
│           │   ├───Images
│           │   │   ├───Training Images
│           │   │   │       img_grid__on_20221012_at_093114.png
│           │   │   │       
│           │   │   └───Validation Images
│           │   │           img_grid__on_20221012_at_093114.png
│           │   │
│           │   ├───Models
│           │   │   ├───BaseModels
│           │   │   │       MobileNet_v3_small__on_20221012_at_093140_.pth
│           │   │   │
│           │   │   └───TrainedModels
│           │   │           antbees_model__on_20221012_at_093227_.pth
│           │   │
│           │   └───Plots
│           │       └───results
│           │               results_plot__on_20221012_at_093227.png
│           │
│           └───Logs
│               ├───CSV Logs
│               │       cv_test_logs.csv
│               │
│               └───Tensorboard Logs
│                   └───logs__on_20221012_at_093052
│                       │   events.out.tfevents.1665541852.My-Laptop.86003.0
│                       │   projector_config.pbtxt
│                       │
│                       ├───00000
│                       │   └───default
│                       │           metadata.tsv
│                       │           sprite.png
│                       │           tensors.tsv
│                       │
│                       └───1665541924.1492205
│                               events.out.tfevents.1665541924.My-Laptop.86003.1
│
├───data_folder
│   ├───train
│   │   ├───ants
│   │   │       0013035.jpg
│   │   │       1030023514_aad5c608f9.jpg
│   │   │       1095476100_3906d8afde.jpg
│   │   │       1099452230_d1949d3250.jpg
│   │   │       116570827_e9c126745d.jpg
│   │   │
│   │   └───bees
│   │           1092977343_cb42b38d62.jpg
│   │           1093831624_fb5fbe2308.jpg
│   │           1097045929_1753d1c765.jpg
│   │           1232245714_f862fbe385.jpg
│   │           129236073_0985e91c7d.jpg
│   │
│   └───val
│       ├───ants
│       │       10308379_1b6c72e180.jpg
│       │       1053149811_f62a3410d3.jpg
│       │       1073564163_225a64f170.jpg
│       │
│       └───bees
│               1032546534_06907fe3b3.jpg
│               10870992_eebeeb3a12.jpg
│               1181173278_23c36fac71.jpg
│
└───__pycache__
        pytorchConfig.cpython-310.pyc

```

### 4. **Contributing**<a id='contributing-'></a>
Yes, Please! We believe that there is alot of oportunity to make Machine Learning more interesting and less complicated for the comunity, so let's make it more efficient, let's go with low-code!!

### 5. **Conclusion**<a id='conclusion-'></a>
All the services which are being provided by ``OneSpace`` could be managed in a single directory without having to leave your workspace. A user only needs to take care of two things before running a training job:
* Setup the relavent configurations in ``config.py``.<br>
* Check if you are importing the right module in ``training.py``.<br>
* Now type ``python training.py`` on the terminal and hit enter.<br>
* Voila, you have a trained production ready model saved in the workspace, Great Work!
#### **Please give this repository a star if you find our work useful, Thank you! 🙏**<br><br>

**Copyright &copy; 2022 OneSpace** <br>
Let's connect on **[``LinkedIn``](https://www.linkedin.com/in/hasanain-mehmood-a37a4116b/)** <br>

