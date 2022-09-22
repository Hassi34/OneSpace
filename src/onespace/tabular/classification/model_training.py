import json
from email.policy import default

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from joblib import Memory
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from .param_selection import get_heper_param_grid


class TrainingFlow:
    def __init__(self, id_col, num_features, cat_features, target_names, metrics, imputer, scaler, X_train, X_val,
                 y_train, y_val, auto=True, pca=False, poly=False, sort_by='acc_val'):
        self.id_col = id_col
        self.num_features = num_features
        self.cat_features = cat_features
        self.imputer = imputer
        self.target_names = target_names
        self.metrics = metrics
        self.scaler = scaler
        self.poly = poly
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.auto = auto
        self.pca = pca
        self.sort_by = sort_by
        self.models = {
            'knn': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(),
            'AdaBoost': AdaBoostClassifier(),
            'DecisionTree': DecisionTreeClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000),
            'lightGBM': lgbm.LGBMClassifier(),
            'xgb': XGBClassifier(),
            'SGDClassifier': SGDClassifier(max_iter=10000000),
            'GaussianNB': GaussianNB(),
            # 'MultinomialNB': MultinomialNB(),
            'GradientBoostingClf': GradientBoostingClassifier(),
            'SVC': SVC(),
            'ExtraTreesClf': ExtraTreesClassifier(),
            'RandForestClf': RandomForestClassifier()
        }

    def prep_pipeline(self):
        # Create a temporary folder to store the transformers of the pipeline
        memory = Memory(location='cachedir')
        if self.poly:
            numeric_transformer = Pipeline(steps=[
                ('imputer', self.imputer),
                ('scaler', self.scaler)
            ])
            categorical_transformer = Pipeline(steps=[
                ('onehotenc', OneHotEncoder(handle_unknown="ignore"))
            ])
            col_transformer = ColumnTransformer(transformers=[
                ('drop_columns', 'drop', self.id_col),
                ('cat_processing', categorical_transformer, self.cat_features),
                ('num_processing', numeric_transformer, self.num_features)
            ], remainder='drop')
            self.prep_pipeline = Pipeline(steps=[
                ('col_transformer', col_transformer),
                ('poly', PolynomialFeatures(degree=2, include_bias=True))
            ], memory=memory)
        else:
            if self.pca:
                pca = PCA(n_components=0.95)
                numeric_transformer = Pipeline(steps=[
                    ('imputer', self.imputer),
                    ('scaler', self.scaler)
                ])
                categorical_transformer = Pipeline(steps=[
                    ('onehotenc', OneHotEncoder(handle_unknown="ignore"))
                ])
                col_transformer = ColumnTransformer(transformers=[
                    ('drop_columns', 'drop', self.id_col),
                    ('cat_processing', categorical_transformer, self.cat_features),
                    ('num_processing', numeric_transformer, self.num_features)
                ], remainder='drop')
                self.prep_pipeline = Pipeline(steps=[
                    ('col_transformer', col_transformer),
                    ('pca', pca)
                ], memory=memory)
            else:
                numeric_transformer = Pipeline(steps=[
                    ('imputer', self.imputer),
                    ('scaler', self.scaler)
                ])
                categorical_transformer = Pipeline(steps=[
                    ('onehotenc', OneHotEncoder(handle_unknown="ignore"))
                ])
                col_transformer = ColumnTransformer(transformers=[
                    ('drop_columns', 'drop', self.id_col),
                    ('cat_processing', categorical_transformer, self.cat_features),
                    ('num_processing', numeric_transformer, self.num_features)
                ], remainder='drop')
                self.prep_pipeline = Pipeline(steps=[
                    ('col_transformer', col_transformer)
                ], memory=memory)
        return self.prep_pipeline

    def compare_base_classifiers(self):
        '''
        This method takes the dictionary of pre-defined models along 
        with the respective dataframes to evaluate the model
        sort_by options are [acc_val,f1_val,recall_val]
        '''
        metrics_dict = {'acc_train': [],
                        'recall_train': [],
                        'precision_train': [],
                        'f1_train': [],
                        'acc_val': [],
                        'recall_val': [],
                        'precision_val': [],
                        'f1_val': []}

        for i in tqdm(self.models):
            pipe = Pipeline([
                ('preprocessing', self.prep_pipeline),
                ('classifier', self.models[i])
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
            metrics_dict, index=self.models.keys()).sort_values(by=self.sort_by, ascending=False)
        print(self.model_training_results)
        print('\n')

    def base_model_report(self):
        self.best_base_model = self.model_training_results[self.sort_by].idxmax(
        )
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
        print(self.best_model)
        if isinstance(self.best_model, list):
            best_models = self.best_model
            for best_model in tqdm(best_models):
                self.best_model = best_model
                self.hyper_param_tuninig()
        else:
            self.hyper_param_tuninig()

    def hyper_param_tuninig(self):
        self.best_model_pipe = Pipeline([
            ('preprocessing', self.prep_pipeline),
            ('classifier', self.models[self.best_model])
        ])
        param_grid, scoring = get_heper_param_grid(
            self.best_model, self.metrics)
        pipe = GridSearchCV(estimator=self.best_model_pipe, param_grid=param_grid,
                            cv=3, scoring=scoring,  verbose=1, n_jobs=-1)
        pipe.fit(self.X_train, self.y_train)
        y_predicted = pipe.predict(self.X_val)
        print("\n")
        print("*****" * 13)
        print(f'Classificatin Report With "{self.best_model}" after tuning ')
        print("*****" * 13)
        print(classification_report(y_predicted,
              self.y_val, target_names=self.target_names))
        print("\n")
        print("*****" * 13)
        print(f'Confusion Matrix for "{self.best_model}"')
        print("*****" * 13)
        print(confusion_matrix(y_predicted, self.y_val))
        print("\n")
        print("*****" * 13)
        print(f'Best params for "{self.best_model}"')
        print("*****" * 13)
        print(json.dumps(pipe.best_params_, indent=2, default=str))
        print("\n")
        print("*****" * 13)
        print(f'Pipeline with best estimator')
        print("*****" * 13)
        print(pipe.best_estimator_)
        print(f' ==> Best score with "{self.best_model}" is "{round(pipe.best_score_,4)}"')

    def get_best_model(self):
        if self.auto:
            self.best_model = self.model_training_results[self.sort_by].idxmax()
        else:
            self.best_model = self.model_training_results[self.sort_by].idxmax()
            usr_rsp = input(
                f'\n ==> "{self.best_model}" is selected as the best model for hyperparameter tuning, type "yes" if you agree with the selection otherwise "no" :')
            print('\n')
            if usr_rsp.title() == 'No':
                usr_rsp = input(
                    "   == Please enter a number(integer) for top N Models to run hyperparameter tuning for or pass an exact model name(lookup in the above result table) to tune a single specific model :")
                if len(usr_rsp) == 1:
                    top_n = int(usr_rsp)
                    self.best_model = self.model_training_results.iloc[:top_n].index.tolist()
                    print(self.best_model)
                else:
                    self.best_model = usr_rsp
                print('\n')

    # def final_training(self, complete_data):
