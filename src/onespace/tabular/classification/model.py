"""
Author : Hasnain Mehmood
Contact : hasnainmehmood3435@gmail.com 
"""
import os

import uuid
import datetime
import csv

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)
from .eda import EDA
from .common import (drop_zero_std, get_data_and_features, get_targets,
                     keep_or_drop_id, sort_by, get_unique_filename, remove_outliers, remove_outliers_z)
from .logs import save_preprocessed_data
from .model_training import TrainingFlow
from .param_selection import get_imputers, get_scaler
from .plot import Plots


class Experiment:
    """ This class shall be used to train a model for a classification problem 
        and save the complete logs and data associated with it
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
        self.HANDLE_IMBALANCE = config.handle_imbalance
        self.PCA = config.pca
        self.RFE = config.feature_selection
        self.RM_OUTLIERS = config.remove_outliers

        self.data_dir = os.path.join(".",config.data_dir, config.csv_file_name)
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
        #data[self.TARGET] = data[self.TARGET].astype(str)
        data = drop_zero_std(data)
        data , self.cat_features, self.num_features = get_data_and_features(data, self.AUTO)

        if (self.cat_features is not None) and (len(self.cat_features) > 0):
            self.id_col, self.cat_features = keep_or_drop_id(data[self.cat_features], self.AUTO)
        else:
            self.id_col = []

        if self.TARGET in self.cat_features:
            self.cat_features.remove(self.TARGET)
        elif self.TARGET in self.num_features:
            self.num_features.remove(self.TARGET)

        if self.TARGET not in self.cat_features + self.num_features:
            data_eda = data[self.cat_features + self.num_features+ [self.TARGET]]
        else:
            data_eda = data[self.cat_features + self.num_features]        
        if self.EDA:
            plots = Plots(data_eda, self.TARGET, self.parent_dir, self.plots_dir)
            plots.EDA()
        
        if self.RM_OUTLIERS:
            data = self.remove_outliers(data)

        self.scaler = get_scaler(self.SCALER)
        self.num_imputer, self.cat_imputer = get_imputers(self.IMPUTER)
        self.sort_by = sort_by(self.METRICS)
        targets = data.pop(self.TARGET)

        self.num_features = data.select_dtypes(include = [np.number]).columns.tolist()
        targets, target_names = get_targets(targets)
        
        self.X_train, self.X_val, self.y_train,  self.y_val = train_test_split(data, targets, test_size=self.VALIDATION_SPLIT, stratify =targets, random_state = 42)
        
        if self.HANDLE_IMBALANCE:
            self.X_train = self.prep_for_smote(self.X_train)
            self.X_val = self.prep_for_smote(self.X_val)
            oversample = SMOTE()
            self.X_train, self.y_train = oversample.fit_resample(self.X_train, self.y_train)
            self.cat_features = self.X_train.select_dtypes(include = ['category', 'object']).columns.tolist()
            self.num_features = self.X_train.select_dtypes(include = np.number).columns.tolist()
  
        training_flow = TrainingFlow(self.id_col, self.num_features, self.cat_features, target_names, self.METRICS,
                                    self.num_imputer,self.cat_imputer, self.scaler, self.X_train, self.X_val, self.y_train, self.y_val, self.HANDLE_IMBALANCE,
                                    self.parent_dir, self.plots_dir, self.pipelines_dir, self.artifacts_dir, self.AUTO, self.PCA, self.POLY, self.RFE, self.sort_by)
        self.prep_pipeline = training_flow.prep_pipeline()
        self.preprocessing()
        training_flow.compare_base_classifiers()
        training_flow.base_model_report()
        training_flow.get_best_model()
        self.best_model, self.best_score , self.training_time = training_flow.hyper_param_tuninig_run()
        self.record_logs()

    def preprocessing(self):
        #try:
        transformed_data = self.prep_pipeline.fit_transform(self.X_train)
        
            #cat_fetures_enc = self.prep_pipeline.named_steps['col_transformer'].transformers_[2][1]\
            #    .named_steps['onehotenc'].get_feature_names_out(self.cat_features)
            #all_features = self.num_features + cat_fetures_enc.tolist()
        all_features = self.prep_pipeline.get_feature_names_out()
        
        if all_features is None :
            all_features = self.prep_pipeline[:-1].get_feature_names_out()
        try:
            self.features_out = [feature.split("__")[1] for feature in all_features]
            transformed_data = pd.DataFrame(transformed_data , columns=self.features_out)
            print(f"\n ==> Features being used for training : {self.features_out}")
        except IndexError:
            # print("\n == PCA added in pipeline")
            # self.features_out = [feature.split("__")[1] for feature in all_features]
            # transformed_data = pd.DataFrame(transformed_data , columns=self.features_out)
            transformed_data = pd.DataFrame(transformed_data, columns=all_features)
        # except AttributeError:
        #     transformed_data = self.prep_pipeline.fit_transform(self.X_train)
        #     all_features = self.cat_features + self.num_features
            
        save_preprocessed_data(transformed_data, self.parent_dir, self.logs_dir)
        

    def prep_for_smote(self, df):
        encoded_data = pd.get_dummies(df[self.cat_features])
        df.drop(columns=self.cat_features+self.id_col, inplace=True)
        df = pd.concat([df, encoded_data], axis= 1)
        X_train_cols = df.columns.tolist()
        df = self.num_imputer.fit_transform(df)
        df = pd.DataFrame(df, columns= X_train_cols)
        return df
    def record_logs(self):
        logs_header = ['Experiment ID','Exeriment Name', 'Executed By', 'Local Date Time','UTC Date Time','Target Column',
         'Metrics', 'Validation Split', 'Auto Pilot', 'EDA', 'SCALER', 'IMPUTER', 'Remove Outliers','Polynomial Features','Handle Imbalance',
         'Recursive Feature Elimination', 'PCA', 'Best Model','Best Score', 'Training Time', 'Comments']

        logs_data = [uuid.uuid4(),self.experiment_name, self.executed_by, datetime.datetime.now(), datetime.datetime.utcnow(),
         self.TARGET, self.METRICS, self.VALIDATION_SPLIT, self.AUTO, self.EDA,
         self.SCALER, self.IMPUTER, self.RM_OUTLIERS, self.POLY, self.HANDLE_IMBALANCE, self.RFE,
         self.PCA, self.best_model, self.best_score, self.training_time, self.comments]
        
        csv_logs_dir_final = os.path.join(self.parent_dir, self.logs_dir, "Final")
        os.makedirs(csv_logs_dir_final, exist_ok=True)
        csv_logs_file = os.path.join(csv_logs_dir_final, self.csv_logs_file+".csv")
        
        with open(csv_logs_file, 'a') as logs_file:
            writer = csv.writer(logs_file, lineterminator='\n')
            if os.stat(csv_logs_file).st_size == 0:
                writer.writerow(logs_header)
            writer.writerow(logs_data)
        print("\n")
        print("*****" * 13)
        print(f'Final CSV logs has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {csv_logs_file}\n")
        print(f"\n************* Kudos, Experiment compeleted successfully! ************\n")
    
    def remove_outliers(self, data):
        csv_logs_dir_outliers = os.path.join(self.parent_dir, self.logs_dir, "Outliers")
        os.makedirs(csv_logs_dir_outliers, exist_ok=True)
        csv_logs_file = os.path.join(csv_logs_dir_outliers, get_unique_filename("outliers", ext="csv"))
        if not isinstance(self.num_features, list):
            self.num_features = [self.num_features]
        data, outliers, outlier_detected_in = remove_outliers_z(data, self.num_features)
        if outliers is not None and len(outliers) > 0:
            outliers.to_csv(csv_logs_file, index=False)
            print(f" ==> Outliers have been detected and removed in : {outlier_detected_in}")
            print("\n   == File containing outliers has been saved at the following location :")
            print(f"\n   ==> {csv_logs_file}")
        return data 
