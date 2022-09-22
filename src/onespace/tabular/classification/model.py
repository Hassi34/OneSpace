"""
Author : Hasnain Mehmood
Contact : hasnainmehmood3435@gmail.com 
"""
import os
import timeit

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

from .common import (drop_zero_std, get_data_and_features, get_targets,
                     keep_or_drop_id, sort_by)
from .logs import save_preprocessed_data
from .model_training import TrainingFlow
from .param_selection import get_imputer, get_scaler


class Experiment:
    """ This class shall be used to train a model for a classification problem 
        and save the complete logs and data associated with it
        Written by : Hasnain
    """
    def __init__(self, config):
        self.config = config

        self.METRICS = config.metrics
        self.VALIDATION_SPLIT = config.validation_split
        self.TASK = config.task
        self.TARGET = config.target_column
        self.AUTO = config.autopilot
        self.SCALER = config.scaler 
        self.IMPUTER = config.imputer
        self.POLY = config.PloynomialFeatures
        self.HANDLE_IMBALANCE = config.handle_imbalance
        self.PCA = config.pca

        self.data_dir = os.path.join(".",config.data_dir, config.csv_file_name)
        self.artifacts_dir = config.artifacts_dir
        self.model_dir = config.model_dir
        self.model_name = config.model_name
        self.logs_dir = config.logs_dir
        self.plots_dir = config.plots_dir
        self.plot_name = config.plot_name
        self.csv_logs_dir_name = config.csv_logs_dir
        self.csv_logs_file = config.csv_logs_file
        self.experiment_name = config.experiment_name
        self.comments = config.comments
        self.executed_by = config.executed_by
    
    def run_experiment(self):
        """This method will start an experiment
           with variables provided at initialization
            Written by : Hasnain
        """
        # from sklearn import datasets
        # iris = datasets.load_iris()
        # data = pd.DataFrame(iris.data,columns=iris.feature_names)
        # data['flower'] = iris.target
        # data['flower'] = data['flower'].apply(lambda x: iris.target_names[x])

        self.parent_dir = os.path.join('Tabular', 'Classification')
        data = pd.read_csv(self.data_dir)
        targets = data.pop(self.TARGET)
        targets, target_names = get_targets(targets)
        data['Glucose'] = data['Glucose'].astype(str)
        data = drop_zero_std(data)
        data , self.cat_features, self.num_features = get_data_and_features(data, self.AUTO)
        #num_data = data[self.num_features]
        #cat_data = data[self.cat_features]
        if (self.cat_features is not None) and (len(self.cat_features) > 0):
            self.id_col, self.cat_features = keep_or_drop_id(data[self.cat_features], self.AUTO)
        else:
            self.id_col = []
        self.X_train, self.X_val, self.y_train,  self.y_val = train_test_split(data, targets, test_size=self.VALIDATION_SPLIT, stratify =targets, random_state = 42)
        if self.HANDLE_IMBALANCE:
            oversample = SMOTE()
            self.X_train, self.y_train = oversample.fit_resample(self.X_train, self.y_train)
        print(self.X_train.head())
        scaler = get_scaler(self.SCALER)
        imputer = get_imputer(self.IMPUTER)
        self.sort_by = sort_by(self.METRICS)
        training_flow = TrainingFlow(self.id_col, self.num_features, self.cat_features, target_names, self.METRICS,
                         imputer, scaler, self.X_train, self.X_val, self.y_train, self.y_val, self.AUTO, self.PCA, self.POLY, self.sort_by)
        self.prep_pipeline = training_flow.prep_pipeline()
        self.preprocessing()
        training_flow.compare_base_classifiers()
        training_flow.base_model_report()
        training_flow.get_best_model()
        training_flow.hyper_param_tuninig_run()

    def preprocessing(self):
        try:
            transformed_data = self.prep_pipeline.fit_transform(self.X_train)
            cat_fetures_enc = self.prep_pipeline.named_steps['col_transformer'].transformers_[1][1]\
                .named_steps['onehotenc'].get_feature_names_out(self.cat_features)
            all_features = self.num_features + cat_fetures_enc.tolist()
        except AttributeError:
            transformed_data = self.prep_pipeline.fit_transform(self.X_train)
            all_features = self.cat_features + self.num_features
        try:
            transformed_data = pd.DataFrame(transformed_data , columns=all_features)
        except ValueError:
            transformed_data = pd.DataFrame(transformed_data)
        save_preprocessed_data(transformed_data, self.parent_dir, self.logs_dir)
        

    
