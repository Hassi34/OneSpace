"""
Author : Hasanain Mehmood
Contact : hasanain@aicaliber.com
"""
import csv
import datetime
import os
import uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ...dbOps.mongo.mongoExe import save_logs_in_mongo
from ...dbOps.mysql.mysqlExe import save_logs_in_mysql
from .common import (drop_zero_std, get_data_and_features, get_targets,
                     get_unique_filename, keep_or_drop_id, remove_outliers_z)
from .logs import save_preprocessed_data
from .model_training import TrainingFlow
from .param_selection import get_imputers, get_scaler
from .plot import Plots


class Experiment:
    """ This class shall be used to train a model for a regression problem 
        Runs the end-to-end training job and records the logs of the experiment
        Written by : Hasnain
    """

    def __init__(self, config):
        self.config = config

        self.METRICS = config.metrics
        self.VALIDATION_SPLIT = config.validation_split
        self.TARGET = config.target_column
        self.AUTO = config.autopilot
        self.EDA = config.eda
        self.SCALER = config.scaler
        self.IMPUTER = config.imputer
        self.POLY = config.PloynomialFeatures
        self.PCA = config.pca
        self.RFE = config.feature_selection
        self.RM_OUTLIERS = config.remove_outliers

        self.data_dir = os.path.join(
            ".", config.data_dir, config.csv_file_name)
        self.project_name = config.project_name
        self.artifacts_dir = config.artifacts_dir
        self.pipelines_dir = config.pipelines_dir
        self.model_name = config.model_name
        self.logs_dir = config.logs_dir
        self.plots_dir = config.plots_dir
        self.csv_logs_dir_name = config.csv_logs_dir
        self.csv_logs_file = config.csv_logs_file
        self.experiment_name = config.experiment_name
        self.comments = config.comments
        self.executed_by = config.executed_by
        self.db_integration_mysql = config.db_integration_mysql
        self.db_integration_mongodb = config.db_integration_mongodb

    def run_experiment(self):
        """This method will start an experiment
           with variables provided at initialization
           Written by : Hasanain
        """
        self.parent_dir = os.path.join(
            'Tabular', 'Regression', self.project_name)
        data = pd.read_csv(self.data_dir)
        print("\n")
        print("*****" * 13)
        print('Data Description')
        print("*****" * 13)
        print(data.describe(include='all').T)
        data = drop_zero_std(data)
        data, self.cat_features, self.num_features = get_data_and_features(
            data, self.AUTO)

        if (self.cat_features is not None) and (len(self.cat_features) > 0):
            self.id_col, self.cat_features = keep_or_drop_id(
                data[self.cat_features], self.AUTO)
        else:
            self.id_col = []

        if self.TARGET in self.cat_features:
            self.cat_features.remove(self.TARGET)
        elif self.TARGET in self.num_features:
            self.num_features.remove(self.TARGET)

        if self.TARGET not in self.cat_features + self.num_features:
            data_eda = data[self.cat_features +
                            self.num_features + [self.TARGET]]
        else:
            data_eda = data[self.cat_features + self.num_features]
        if self.EDA:
            plots = Plots(data_eda, self.TARGET,
                          self.parent_dir, self.plots_dir)
            plots.EDA()

        if self.RM_OUTLIERS:
            data = self.remove_outliers(data)

        self.scaler = get_scaler(self.SCALER)
        self.num_imputer, self.cat_imputer = get_imputers(self.IMPUTER)
        self.sort_by = self.METRICS+"_val"
        targets = data.pop(self.TARGET)

        self.num_features = data.select_dtypes(
            include=[np.number]).columns.tolist()
        targets, target_names = get_targets(targets)

        self.X_train, self.X_val, self.y_train,  self.y_val = train_test_split(
            data, targets, test_size=self.VALIDATION_SPLIT, random_state=42)

        training_flow = TrainingFlow(self.id_col, self.num_features, self.cat_features, self.METRICS, self.num_imputer,
                                     self.cat_imputer, self.scaler, self.X_train, self.X_val, self.y_train, self.y_val,
                                     self.parent_dir, self.plots_dir, self.logs_dir, self.pipelines_dir, self.artifacts_dir,
                                     self.AUTO, self.PCA, self.POLY, self.RFE, self.sort_by)
        self.prep_pipeline = training_flow.prep_pipeline()
        self.preprocessing()
        training_flow.compare_regressors()
        training_flow.compare_before_tuning()
        training_flow.base_model_report()
        training_flow.get_best_model()
        self.best_model, self.best_score, self.training_time = training_flow.hyper_param_tuninig_run()
        training_flow.compare_after_tuning()
        self.record_logs()

    def preprocessing(self):
        transformed_data = self.prep_pipeline.fit_transform(self.X_train)
        all_features = self.prep_pipeline.get_feature_names_out()

        if all_features is None:
            all_features = self.prep_pipeline[:-1].get_feature_names_out()
        try:
            self.features_out = [feature.split(
                "__")[1] for feature in all_features]
            transformed_data = pd.DataFrame(
                transformed_data, columns=self.features_out)
            print(
                f"\n ==> Features being used for training : {self.features_out}")
        except IndexError:
            transformed_data = pd.DataFrame(
                transformed_data, columns=all_features)

        save_preprocessed_data(
            transformed_data, self.parent_dir, self.logs_dir)

    def record_logs(self):
        logs_header = ['Experiment ID', 'Exeriment Name', 'Executed By', 'Local Date Time', 'UTC Date Time', 'Target Column',
                       'Metrics', 'Validation Split', 'Auto Pilot', 'EDA', 'SCALER', 'IMPUTER', 'Remove Outliers', 'Polynomial Features',
                       'Recursive Feature Elimination', 'PCA', 'Best Model', 'Best Score', 'Training Time', 'Comments']

        logs_data = [str(uuid.uuid4()), self.experiment_name, self.executed_by, datetime.datetime.now(), datetime.datetime.utcnow(),
                     self.TARGET, self.METRICS, self.VALIDATION_SPLIT, self.AUTO, self.EDA,
                     self.SCALER, self.IMPUTER, self.RM_OUTLIERS, self.POLY, self.RFE,
                     self.PCA, self.best_model, self.best_score, self.training_time, self.comments]

        csv_logs_dir_final = os.path.join(
            self.parent_dir, self.logs_dir, "Final")
        os.makedirs(csv_logs_dir_final, exist_ok=True)
        csv_logs_file = os.path.join(
            csv_logs_dir_final, self.csv_logs_file+".csv")

        with open(csv_logs_file, 'a') as logs_file:
            writer = csv.writer(logs_file, lineterminator='\n')
            if os.stat(csv_logs_file).st_size == 0:
                writer.writerow(logs_header)
            writer.writerow(logs_data)
        if self.db_integration_mongodb:
            try:
                save_logs_in_mongo(self.project_name, dict(
                    zip(logs_header, logs_data)))
            except:
                print("!!! Could not record Logs in MongoDB, Please check the connection string and premissions one again")
            finally:
                pass
        if self.db_integration_mysql:
            try:
                save_logs_in_mysql(
                    data=logs_data, columns=logs_header, project_name=self.project_name)
            except:
                print("!!! Could not record Logs in MySQL, Please check the credentials and premissions one again")
            finally:
                pass
        print("\n")
        print("*****" * 13)
        print(f'Final CSV logs have been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {csv_logs_file}\n")
        print(f"\n************* Kudos, Experiment compeleted successfully! ************\n")

    def remove_outliers(self, data):
        csv_logs_dir_outliers = os.path.join(
            self.parent_dir, self.logs_dir, "Outliers")
        os.makedirs(csv_logs_dir_outliers, exist_ok=True)
        csv_logs_file = os.path.join(
            csv_logs_dir_outliers, get_unique_filename("outliers", ext="csv"))
        if not isinstance(self.num_features, list):
            self.num_features = [self.num_features]
        data, outliers, outlier_detected_in = remove_outliers_z(
            data, self.num_features)
        if outliers is not None and len(outliers) > 0:
            outliers.to_csv(csv_logs_file, index=False)
            print(
                f" ==> Outliers have been detected and removed in : {outlier_detected_in}")
            print(
                "\n   == File containing outliers has been saved at the following location :")
            print(f"\n   ==> {csv_logs_file}")
        return data
