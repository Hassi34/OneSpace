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
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.feature_selection import RFE
from sklearn.linear_model import (ElasticNet, Lasso, LassoLars,
                                  PassiveAggressiveRegressor, Ridge,
                                  SGDRegressor)
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

from .common import get_unique_filename
from .logs import (save_regressor_comparision_after_training,
                   save_regressor_comparision_before_training)
from .param_selection import get_heper_param_grid
from .plot import Eval


class TrainingFlow:
    def __init__(self, id_col, num_features, cat_features, metrics, num_imputer, cat_imputer, scaler, X_train, X_val,
                 y_train, y_val, parent_dir, plots_dir, logs_dir, pipelines_dir, artifacts_dir, auto=True, pca=False, poly=False, rfe=False, sort_by='acc_val'):
        self.id_col = id_col
        self.num_features = num_features
        self.cat_features = cat_features
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.metrics = metrics
        self.scaler = scaler
        self.poly = poly
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.parent_dir = parent_dir
        self.plots_dir = plots_dir
        self.logs_dir = logs_dir
        self.artifacts_dir = artifacts_dir
        self.pipelines_dir = os.path.join(
            parent_dir, artifacts_dir, get_unique_filename(pipelines_dir))
        self.auto = auto
        self.pca = pca
        self.rfe = rfe
        self.sort_by = sort_by
        self.models = {
            'KNeighborsRegressor': KNeighborsRegressor(),
            'ElasticNet': ElasticNet(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            # 'MLPRegressor': MLPRegressor(max_iter=1000000),
            'LGBMRegressor': lgbm.LGBMRegressor(),
            'XGBRegressor': XGBRegressor(),
            'SGDRegressor': SGDRegressor(max_iter=10000000),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'LassoLars': LassoLars(),
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'SVR': SVR(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'RandomForestRegressor': RandomForestRegressor()
        }

    def prep_pipeline(self):
        # Create a temporary folder to store the transformers of the pipeline
        memory = Memory(location='cachedir')
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

    def compare_regressors(self, models=None):
        '''
        This method takes the dictionary of pre-defined models along 
        with the respective dataframes to evaluate the model
        sort by a defined metrics
        '''
        if models is None:
            models = self.models
        metrics_dict = {'r2_score_train': [],
                        'mean_absolute_error_train': [],
                        'mean_squared_error_train': [],
                        'mean_absolute_percentage_error_train': [],
                        'median_absolute_error_train': [],
                        'explained_variance_score_train': [],
                        'r2_score_val': [],
                        'mean_absolute_error_val': [],
                        'mean_squared_error_val': [],
                        'mean_absolute_percentage_error_val': [],
                        'median_absolute_error_val': [],
                        'explained_variance_score_val': []}
        n_features_to_select = int(self.X_train.shape[1] * 0.7)
        self.rfe_ = RFE(estimator=DecisionTreeRegressor(),
                        n_features_to_select=n_features_to_select)
        for i in tqdm(models, desc="Training"):
            if self.rfe:
                pipe = Pipeline([
                    ('preprocessing', self.prep_pipeline),
                    ('RFE', self.rfe_),
                    ('regressor', models[i])
                ])
            else:
                pipe = Pipeline([
                    ('preprocessing', self.prep_pipeline),
                    ('regressor', models[i])
                ])
            pipe.fit(self.X_train, self.y_train)
            y_predicted_train = pipe.predict(self.X_train)
            y_predicted = pipe.predict(self.X_val)

            metrics_dict['r2_score_train'].append(
                round(r2_score(self.y_train, y_predicted_train), 3))
            metrics_dict['r2_score_val'].append(
                round(r2_score(self.y_val, y_predicted), 3))

            metrics_dict['mean_absolute_error_train'].append(
                round(mean_absolute_error(self.y_train, y_predicted_train), 3))
            metrics_dict['mean_absolute_error_val'].append(
                round(mean_absolute_error(self.y_val, y_predicted), 3))

            metrics_dict['mean_squared_error_train'].append(
                round(mean_squared_error(self.y_train, y_predicted_train), 3))
            metrics_dict['mean_squared_error_val'].append(
                round(mean_squared_error(self.y_val, y_predicted), 3))

            metrics_dict['mean_absolute_percentage_error_train'].append(round(mean_absolute_percentage_error(
                self.y_train, y_predicted_train), 3))
            metrics_dict['mean_absolute_percentage_error_val'].append(round(mean_absolute_percentage_error(
                self.y_val, y_predicted), 3))

            metrics_dict['median_absolute_error_train'].append(round(median_absolute_error(
                self.y_train, y_predicted_train), 3))
            metrics_dict['median_absolute_error_val'].append(round(median_absolute_error(
                self.y_val, y_predicted), 3))

            metrics_dict['explained_variance_score_train'].append(round(explained_variance_score(
                self.y_train, y_predicted_train), 3))
            metrics_dict['explained_variance_score_val'].append(round(explained_variance_score(
                self.y_val, y_predicted), 3))
        if self.metrics in ['mean_absolute_error','mean_squared_error', 'mean_absolute_percentage_error','median_absolute_error', 'explained_variance_score']:
            ascending = True
        else:
            ascending = False
        self.model_training_results = pd.DataFrame(
            metrics_dict, index=models.keys()).sort_values(by=self.sort_by, ascending=ascending)

    def base_model_report(self):
        self.best_base_model = self.model_training_results.index[0]
        self.best_base_pipe = Pipeline([
            ('preprocessing', self.prep_pipeline),
            ('regressor', self.models[self.best_base_model])
        ])
        self.best_base_pipe.fit(self.X_train, self.y_train)

    def hyper_param_tuninig_run(self):
        best_models = self.best_model
        self.eval = Eval(self.y_val, self.parent_dir, self.plots_dir)
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
                self.tuned_models[self.best_model] = self.best_regressor
                self.best_score.append(
                    round(self.hyper_param_pipe.best_score_, 4))
                self.display_mertics()
                self.eval.feature_importance(
                    self.best_pipeline_after_training, self.X_val)
                self.eval.residplot(y_predicted=self.y_predicted)
            self.append_stacking_voting()
        else:
            self.best_score = []
            print(
                f'\nRunning hyperparameter tuning for "{self.best_model}"...')
            self.hyper_param_tuninig()
            self.save_best_pipeline()
            self.all_pipeline_paths.append(self.pipe_path_to_save)
            self.model_names.append(self.best_model)
            self.tuned_models[self.best_model] = self.best_regressor
            self.best_score.append(round(self.hyper_param_pipe.best_score_, 4))
            self.display_mertics()
            self.eval.feature_importance(
                self.best_pipeline_after_training, self.X_val)
            self.eval.residplot(y_predicted=self.y_predicted)
        end_time = timeit.default_timer()
        training_time = round((end_time - start_time)/60.0, 3)
        self.compare_regressors(models=self.tuned_models)
        if isinstance(best_models, list):
            self.train_save_stacking_voting()
        return best_models, self.best_score, training_time

    def hyper_param_tuninig(self):
        if self.rfe:
            self.best_model_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('regressor', self.models[self.best_model])
            ])
        else:
            self.best_model_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('regressor', self.models[self.best_model])
            ])
        param_grid = get_heper_param_grid(self.best_model)
        self.hyper_param_pipe = GridSearchCV(estimator=self.best_model_pipe, param_grid=param_grid,
                                             cv=3, scoring='r2',  verbose=1, n_jobs=-1)
        self.hyper_param_pipe.fit(self.X_train, self.y_train)
        self.best_pipeline_after_training = self.hyper_param_pipe.best_estimator_
        self.best_regressor = self.hyper_param_pipe.best_estimator_[
            'regressor']
        self.y_predicted = self.best_pipeline_after_training.predict(
            self.X_val)

    def display_mertics(self):
        print("\n")
        print("*****" * 13)
        print(f'Best params for "{self.best_model}"')
        print("*****" * 13)
        print(
            f"{json.dumps(self.hyper_param_pipe.best_params_, indent=2, default=str)}\n")
        print("*****" * 13)
        print(f'Pipeline with best estimator')
        print("*****" * 13)
        print(self.best_pipeline_after_training)
        print(
            f'\n ==> Best score with "{self.best_model}" is "{round(self.hyper_param_pipe.best_score_,4)}"\n')
        print("*****" * 13)
        print(
            f'Best Estimator pipeline for "{self.best_model}" has been saved at the following location')
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
                    self.best_model = self.model_training_results.iloc[:top_n].index.tolist(
                    )
                    print(
                        f"==> You selected these models for hyperparameter tuning : {self.best_model}")
                else:
                    self.best_model = usr_rsp
                    print(
                        f"==> Selected Model to run the  hyperparameter tuning on : {self.best_model}")
                print('\n')

    def save_best_pipeline(self):
        os.makedirs(self.pipelines_dir, exist_ok=True)
        self.pipe_path_to_save = os.path.join(
            self.pipelines_dir, get_unique_filename(self.best_model, is_model_name=True))
        with open(self.pipe_path_to_save, 'wb') as file:
            pickle.dump(self.best_pipeline_after_training, file)

    def compare_before_tuning(self):
        print("\n")
        print("*****" * 13)
        print('Model Performance with Base Models')
        print("*****" * 13)
        print(self.model_training_results)
        print('\n')
        save_regressor_comparision_before_training(
            self.model_training_results, self.parent_dir, self.logs_dir)

    def compare_after_tuning(self):
        print("\n")
        print("*****" * 13)
        print('Model Performance with Tuned Models')
        print("*****" * 13)
        print(self.model_training_results)
        print('\n')
        save_regressor_comparision_after_training(
            self.model_training_results, self.parent_dir, self.logs_dir)

    def append_stacking_voting(self):
        reg_list = [(k, v) for k, v in self.tuned_models.items()]
        weights = [len(reg_list)+1-n for n in range(1,len(reg_list)+1)]
        self.votingRegressor = VotingRegressor(reg_list, weights=weights)
        self.stackingRegressor = StackingRegressor(
            reg_list, final_estimator=SVR())
        self.tuned_models["VotingRegressor"] = self.votingRegressor
        self.tuned_models["StackingRegressor"] = self.stackingRegressor

    def train_save_stacking_voting(self):
        if self.rfe:
            self.VotingRegressor_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('regressor', self.votingRegressor)
            ])
        else:
            self.VotingRegressor_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('regressor', self.votingRegressor)
            ])
        if self.rfe:
            self.StackingRegressor_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('RFE', self.rfe_),
                ('regressor', self.stackingRegressor)
            ])
        else:
            self.StackingRegressor_pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('regressor', self.stackingRegressor)
            ])

        self.VotingRegressor_pipe.fit(self.X_train, self.y_train)
        self.StackingRegressor_pipe.fit(self.X_train, self.y_train)

        pipe_path_to_save = os.path.join(
            self.pipelines_dir, get_unique_filename("VotingRegressor", is_model_name=True))
        with open(pipe_path_to_save, 'wb') as file:
            pickle.dump(self.VotingRegressor_pipe, file)
        print("*****" * 13)
        print('Best Estimator pipeline for Voting Regressor has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {pipe_path_to_save}\n")

        pipe_path_to_save = os.path.join(self.pipelines_dir, get_unique_filename(
            "StackingRegressor", is_model_name=True))
        with open(pipe_path_to_save, 'wb') as file:
            pickle.dump(self.StackingRegressor_pipe, file)
        print("*****" * 13)
        print('Best Estimator pipeline for Stacking Regressor has been saved at the following location')
        print("*****" * 13)
        print(f"\n ==> {pipe_path_to_save}\n")
