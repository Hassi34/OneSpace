"""
Author : Hasanain Mehmood
Contact : hasanain@aicaliber.com
"""
import json
import os
import pickle
import timeit

import lightgbm as lgbm
import pandas as pd
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from .common import get_unique_filename
from .param_selection import get_heper_param_grid
from .plot import Eval


class TrainingFlow:
    def __init__(self, id_col, num_features, cat_features, target_names, metrics, num_imputer, cat_imputer, scaler, X_train, X_val,
                 y_train, y_val, smote, parent_dir, plots_dir, pipelines_dir, artifacts_dir, auto=True, pca=False, poly=False, rfe = False, sort_by='acc_val'):
        self.id_col = id_col
        self.num_features = num_features
        self.cat_features = cat_features
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.target_names = target_names
        self.metrics = metrics
        self.scaler = scaler
        self.poly = poly
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.smote = smote
        self.parent_dir = parent_dir
        self.plots_dir = plots_dir 
        self.artifacts_dir = artifacts_dir
        self.pipelines_dir = os.path.join(parent_dir, artifacts_dir, get_unique_filename(pipelines_dir))
        self.auto = auto
        self.pca = pca
        self.rfe = rfe
        self.sort_by = sort_by
        self.models = {
            'KNeighborsClassifier': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000),
            'LGBMClassifier': lgbm.LGBMClassifier(),
            'XGBClassifier': XGBClassifier(),
            'SGDClassifier': SGDClassifier(max_iter=10000000),
            'GaussianNB': GaussianNB(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC(probability=True),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier()
        }

    def prep_pipeline(self):
        # Create a temporary folder to store the transformers of the pipeline
        memory = Memory(location='cachedir')
        if self.smote: 
            self.id_col = []
        if self.poly:
            numeric_transformer = Pipeline(steps=[
                ('num_imputer', self.num_imputer),
                ('scaler', self.scaler)
            ])
            categorical_transformer = Pipeline(steps=[
                ('cat_imputer', self.cat_imputer),
                ('onehotenc', OneHotEncoder(handle_unknown="ignore"))
            ])
            col_transformer = ColumnTransformer(transformers=[
                ('drop_columns', 'drop', self.id_col),
                ('num_processing', numeric_transformer, self.num_features),
                ('cat_processing', categorical_transformer, self.cat_features)
            ], remainder='drop')
            
            self.prep_pipeline = Pipeline(steps=[
                ('col_transformer', col_transformer),
                ('poly', PolynomialFeatures(degree=2, include_bias=True))
            ], memory=memory)
        else:
            if self.pca:
                pca = PCA(n_components=0.95)
                numeric_transformer = Pipeline(steps=[
                    ('num_imputer', self.num_imputer),
                    ('scaler', self.scaler)
                ])
                categorical_transformer = Pipeline(steps=[
                    ('cat_imputer', self.cat_imputer),
                    ('onehotenc', OneHotEncoder(handle_unknown="ignore"))
                ])
                col_transformer = ColumnTransformer(transformers=[
                    ('drop_columns', 'drop', self.id_col),
                    ('num_processing', numeric_transformer, self.num_features),
                    ('cat_processing', categorical_transformer, self.cat_features)
                ], remainder='drop')
                
                self.prep_pipeline = Pipeline(steps=[
                    ('col_transformer', col_transformer),
                    ('pca', pca)
                ], memory=memory)
            else:
                numeric_transformer = Pipeline(steps=[
                    ('num_imputer', self.num_imputer),
                    ('scaler', self.scaler)
                ])
                categorical_transformer = Pipeline(steps=[
                    ('cat_imputer', self.cat_imputer),
                    ('onehotenc', OneHotEncoder(handle_unknown="ignore"))
                ])
                col_transformer = ColumnTransformer(transformers=[
                    ('drop_columns', 'drop', self.id_col),
                    ('num_processing', numeric_transformer, self.num_features),
                    ('cat_processing', categorical_transformer, self.cat_features)
                ], remainder='drop')
                self.prep_pipeline = Pipeline(steps=[
                    ('col_transformer', col_transformer)
                ], memory=memory)
        return self.prep_pipeline

    def compare_classifiers(self, models = None):
        '''
        This method takes the dictionary of pre-defined models along 
        with the respective dataframes to evaluate the model
        sort_by options are [acc_val,f1_val,recall_val]
        '''
        if models is None:
            models = self.models
        metrics_dict = {'acc_train': [],
                        'recall_train': [],
                        'precision_train': [],
                        'f1_train': [],
                        'acc_val': [],
                        'recall_val': [],
                        'precision_val': [],
                        'f1_val': []}
        n_features_to_select = int(self.X_train.shape[1] * 0.7)
        self.rfe_ = RFE(estimator=DecisionTreeClassifier(), n_features_to_select= n_features_to_select)
        for i in tqdm(models, desc="Training"):
            if self.rfe:
                pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('classifier', models[i])
                ])
            else:
                pipe = Pipeline([
                    ('preprocessing', self.prep_pipeline),
                    ('classifier', models[i])
                ])
            pipe.fit(self.X_train, self.y_train)
            y_predicted_train = pipe.predict(self.X_train)
            y_predicted = pipe.predict(self.X_val)

            metrics_dict['acc_train'].append(
                round(accuracy_score(self.y_train, y_predicted_train), 4))
            metrics_dict['acc_val'].append(
                round(accuracy_score(self.y_val, y_predicted), 4))

            metrics_dict['f1_train'].append(
                round(f1_score(self.y_train, y_predicted_train, average='weighted'), 4))
            metrics_dict['f1_val'].append(
                round(f1_score(self.y_val, y_predicted, average='weighted'), 4))

            metrics_dict['recall_train'].append(
                round(recall_score(self.y_train, y_predicted_train, average='weighted'), 4))
            metrics_dict['recall_val'].append(
                round(recall_score(self.y_val, y_predicted, average='weighted'), 4))

            metrics_dict['precision_train'].append(round(precision_score(
                self.y_train, y_predicted_train, average='weighted'), 4))
            metrics_dict['precision_val'].append(round(precision_score(
                self.y_train, y_predicted_train, average='weighted'), 4))

        self.model_training_results = pd.DataFrame(
            metrics_dict, index=models.keys()).sort_values(by=self.sort_by, ascending=False)
    def base_model_report(self):
        self.best_base_model = self.model_training_results.index[0]
        self.best_base_pipe = Pipeline([
            ('preprocessing', self.prep_pipeline),
            ('classifier', self.models[self.best_base_model])
        ])
        self.best_base_pipe.fit(self.X_train, self.y_train)
        y_predicted = self.best_base_pipe.predict(self.X_val)
        print("\n")
        print("*****" * 13)
        print(f'Classificatin Report With "{self.best_base_model}" as Base Model')
        print("*****" * 13)
        print(classification_report(y_predicted,
              self.y_val, target_names=self.target_names))

    def hyper_param_tuninig_run(self):
        best_models = self.best_model
        self.eval = Eval(self.y_val, self.target_names, self.parent_dir, self.plots_dir)
        self.all_pipeline_paths = []
        self.model_names = []
        self.tuned_models = {}
        start_time = timeit.default_timer()
        if isinstance(best_models, list):
            self.best_score = []
            for best_model in tqdm(best_models, desc="Hyperparameter Tuninig"):
                print(f'\nTuning hyperparameters for "{best_model}"...')
                self.best_model = best_model
                self.hyper_param_tuninig()
                self.save_best_pipeline()
                self.all_pipeline_paths.append(self.pipe_path_to_save)
                self.model_names.append(self.best_model)
                self.tuned_models[self.best_model] = self.best_classifier
                self.best_score.append(round(self.hyper_param_pipe.best_score_,4))
                self.display_mertics()
                self.eval.confusion_matrix(self.y_predicted, self.best_model)
                self.eval.feature_importance(self.best_pipeline_after_training, self.X_val)
            self.append_stacking_voting()
        else:
            self.best_score = []
            print(f'Tuning hyperparameters for "{self.best_model}"...')
            self.hyper_param_tuninig()
            self.save_best_pipeline()
            self.all_pipeline_paths.append(self.pipe_path_to_save)
            self.model_names.append(self.best_model)
            self.tuned_models[self.best_model] = self.best_classifier
            self.best_score.append(round(self.hyper_param_pipe.best_score_,4))
            self.display_mertics()
            self.eval.confusion_matrix(self.y_predicted, self.best_model)
            self.eval.feature_importance(self.best_pipeline_after_training, self.X_val)
        end_time = timeit.default_timer()
        training_time = round((end_time - start_time)/60.0, 3)
        self.compare_classifiers(models = self.tuned_models)
        self.compare_after_tuning()
        if isinstance(best_models, list):
            self.train_save_stacking_voting()

        self.eval.auc_roc(self.X_val, self.all_pipeline_paths, self.model_names)
        return best_models, self.best_score, training_time

    def hyper_param_tuninig(self):
        if self.rfe:
            self.best_model_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('classifier', self.models[self.best_model])
            ])
        else:
            self.best_model_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('classifier', self.models[self.best_model])
            ])
        param_grid, scoring = get_heper_param_grid(
            self.best_model, self.metrics)
        self.hyper_param_pipe = GridSearchCV(estimator=self.best_model_pipe, param_grid=param_grid,
                            cv=3, scoring=scoring,  verbose=1, n_jobs=-1)
        self.hyper_param_pipe.fit(self.X_train, self.y_train)
        self.best_pipeline_after_training = self.hyper_param_pipe.best_estimator_
        self.best_classifier = self.hyper_param_pipe.best_estimator_['classifier']
        self.y_predicted = self.best_pipeline_after_training.predict(self.X_val)
        
    def display_mertics(self):
        print("\n")
        print("*****" * 13)
        print(f'Classificatin Report With "{self.best_model}" after tuning ')
        print("*****" * 13)
        print(classification_report(self.y_predicted,
            self.y_val, target_names=self.target_names))
        print("\n")
        print("*****" * 13)
        print(f'Confusion Matrix for "{self.best_model}"')
        print("*****" * 13)
        print(confusion_matrix(self.y_predicted, self.y_val))
        print("\n")
        print("*****" * 13)
        print(f'Best params for "{self.best_model}"')
        print("*****" * 13)
        print(f"{json.dumps(self.hyper_param_pipe.best_params_, indent=2, default=str)}\n")
        print("*****" * 13)
        print(f'Pipeline with best estimator')
        print("*****" * 13)
        print(self.best_pipeline_after_training)
        print(f'\n ==> Best score with "{self.best_model}" is "{round(self.hyper_param_pipe.best_score_,4)}"\n')
        print("*****" * 13)
        print(f'Best Estimator pipeline for "{self.best_model}" has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {self.pipe_path_to_save}\n")
    def get_best_model(self):
        self.best_model = self.model_training_results.index[0]
        if not self.auto:
            usr_rsp = input(
                f'\n ==> "{self.best_model}" is selected as the best model for hyperparameter tuning, type "yes" if you agree with the selection otherwise "no" :')
            print('\n')
            if usr_rsp.title() == 'No':
                usr_rsp = input(
                    "   == Please enter a number(integer) for top N Models to run hyperparameter tuning for or pass an exact model name(lookup in the above result table) to tune a single specific model :")
                if len(usr_rsp) <= 2:
                    top_n = int(usr_rsp)
                    self.best_model = self.model_training_results.iloc[:top_n].index.tolist()
                    print(f"==> You selected these models for hyperparameter tuning : {self.best_model}")
                else:
                    self.best_model = usr_rsp
                    print(f"==> Selected Model to run the  hyperparameter tuning on : {self.best_model}")
                print('\n')

    def save_best_pipeline(self):
        os.makedirs(self.pipelines_dir, exist_ok=True)
        self.pipe_path_to_save = os.path.join(self.pipelines_dir,get_unique_filename(self.best_model, is_model_name=True))
        with open(self.pipe_path_to_save, 'wb') as file:
            pickle.dump(self.best_pipeline_after_training, file) 

    def compare_before_tuning(self):
        print("\n")
        print("*****" * 13)
        print(f'Training result with base models')
        print("*****" * 13)
        print(self.model_training_results)
        print('\n')
    def compare_after_tuning(self):
        print("\n")
        print("*****" * 13)
        print(f'Training result with tuned models')
        print("*****" * 13)
        print(self.model_training_results)
        print('\n')
    def append_stacking_voting(self):
        clf_list = [(k, v) for k, v in self.tuned_models.items()]
        weights = [len(clf_list)+1-n for n in range(1,len(clf_list)+1)]
        self.votingClf = VotingClassifier(clf_list, weights= weights, voting='soft')
        self.stackingClf = StackingClassifier(clf_list, stack_method = "predict_proba", final_estimator = SVC(probability=True))
        self.tuned_models["VotingClf"] =  self.votingClf
        self.tuned_models["StackingClf"] = self.stackingClf
    
    def train_save_stacking_voting(self):
        if self.rfe:
            self.votingClf_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('classifier', self.votingClf)
            ])
        else:
            self.votingClf_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('classifier', self.votingClf)
            ])
        if self.rfe:
            self.stackingClf_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('classifier', self.stackingClf)
            ])
        else:
            self.stackingClf_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('classifier', self.stackingClf)
            ])

        self.votingClf_pipe.fit(self.X_train, self.y_train)
        self.stackingClf_pipe.fit(self.X_train, self.y_train)
        self.eval.confusion_matrix(y_predicted = self.stackingClf_pipe.predict(self.X_val),
                                     model = "StackingClf")
        self.eval.confusion_matrix(y_predicted = self.votingClf_pipe.predict(self.X_val),
                                     model = "VotingClf")
        self.path_to_voting = os.path.join(self.pipelines_dir,get_unique_filename("VotingClf", is_model_name=True))
        with open(self.path_to_voting, 'wb') as file:
            pickle.dump(self.votingClf_pipe, file)
        print("*****" * 13)
        print('Best Estimator pipeline for Voting Classifier has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {self.path_to_voting}\n")
        
        self.path_to_stacking = os.path.join(self.pipelines_dir,get_unique_filename("StackingClf", is_model_name=True))
        with open(self.path_to_stacking, 'wb') as file:
            pickle.dump(self.stackingClf_pipe, file)
        print("*****" * 13)
        print('Best Estimator pipeline for Stacking Classifier has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {self.path_to_stacking}\n")

        self.all_pipeline_paths.append(self.path_to_voting)
        self.model_names.append("VotingClf")
        self.all_pipeline_paths.append(self.path_to_stacking)
        self.model_names.append("StackingClf")
