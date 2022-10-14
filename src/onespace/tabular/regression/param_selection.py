import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, RobustScaler,
                                   StandardScaler)


def get_scaler(scaler):
    if scaler == 'StandardScaler':
        return StandardScaler()
    elif scaler == 'MinMaxScaler':
        return MinMaxScaler()
    elif scaler == 'MaxAbsScaler':
        return MaxAbsScaler()
    elif scaler == 'RobustScaler':
        return RobustScaler()
    else:
        raise ValueError(f'Invalid scaler: "{scaler}"')


def get_imputers(imputer):
    cat_imputer = SimpleImputer(
        strategy='most_frequent', missing_values=np.nan)
    if imputer == "SimpleImputer":
        num_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    elif imputer == "KNNImputer":
        num_imputer = KNNImputer(n_neighbors=3, missing_values=np.nan)
    else:
        raise ValueError(f'Invalid imputer: "{imputer}"')
    return num_imputer, cat_imputer


def get_heper_param_grid(model_name):
    if model_name == "KNeighborsRegressor":
        param_grid = {'regressor__n_neighbors': np.arange(2, 14, 2),
                      'regressor__weights': ['uniform', 'distance'],
                      'regressor__metric': ['minkowski', 'euclidean', 'manhattan'],
                      'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'regressor__leaf_size': np.arange(5, 50, 5)
                      }

    elif model_name == "MLPRegressor":
        param_grid = {'regressor__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                      'regressor__activation': ['tanh', 'logistic', 'relu'],
                      'regressor__solver': ['sgd', 'lbfgs', 'adam'],
                      'regressor__alpha':  np.linspace(0.0001, 0.9, 5),
                      'regressor__learning_rate': ['constant', 'invscaling', 'adaptive']
                      }

    elif model_name == "RandomForestRegressor":
        param_grid = {'regressor__max_depth': [10, 30, 60, 70, None],
                      'regressor__bootstrap': [True, False],
                      'regressor__min_samples_split': np.arange(2, 8, 2),
                      'regressor__min_samples_leaf': np.arange(1, 5, 2),
                      'regressor__n_estimators':  np.arange(10, 200, 20),
                      'regressor__max_features': ['sqrt', 'log2', None]
                      }
    elif model_name == "SVR":
        param_grid = {'regressor__C': [0.1, 1, 10, 50],
                      'regressor__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                      'regressor__gamma': ['scale', 'auto']
                      }
    elif model_name == "AdaBoostRegressor":
        param_grid = {'regressor__n_estimators': np.arange(5, 100, 5),
                      'regressor__learning_rate':  np.linspace(0.0001, 0.9, 5),
                      'regressor__loss': ['linear', 'square', 'exponential']
                      }
    elif model_name == "DecisionTreeRegressor":
        param_grid = {'regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                      'regressor__max_depth': np.arange(5, 40, 5),
                      'regressor__min_samples_split': np.arange(2, 6, 2),
                      'regressor__min_samples_leaf': np.arange(1, 10, 2),
                      'regressor__max_features': [None, 'sqrt', 'log2']
                      }

    elif model_name == "GradientBoostingRegressor":
        param_grid = {'regressor__n_estimators': np.arange(5, 200, 10),
                      'regressor__learning_rate': np.linspace(0.0001, 0.9, 3),
                      'regressor__subsample': np.linspace(0.4, 1.0, 3),
                      'regressor__max_depth': np.arange(5, 20, 5),
                      'regressor__max_features': [None, 'sqrt', 'log2']
                      }
    elif model_name == "SGDRegressor":
        param_grid = {'regressor__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
                      'regressor__alpha': [0.0001, 0.001, 0.01, 1, 10, 100, 1000],
                      'regressor__penalty':  ['l1', 'l2', 'elasticnet'],
                      'regressor__eta0': np.linspace(0.0001, 1.0, 5),
                      'regressor__loss': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
                      }
    elif model_name == "ExtraTreesRegressor":
        param_grid = {'regressor__n_estimators': np.arange(10, 150, 20),
                      'regressor__criterion': ['squared_error', 'absolute_error'],
                      'regressor__min_samples_leaf': np.arange(1, 10, 5),
                      'regressor__min_samples_split': np.arange(2, 10, 5),
                      'regressor__max_features': [None, 'sqrt', 'log2']
                      }
    elif model_name == "XGBRegressor":
        param_grid = {'regressor__max_depth': np.arange(3, 18, 5),
                      'regressor__gamma': np.arange(0, 15, 3),
                      'regressor__eta': np.linspace(0.0001, 1.0, 5),
                      'regressor__colsample_bytree': np.linspace(0.5, 1, 3),
                      'regressor__n_estimators': np.arange(10, 150, 20)
                      }
    elif model_name == "LGBMRegressor":
        param_grid = {'regressor__n_estimators': np.arange(5, 150, 20),
                      'regressor__colsample_bytree': np.arange(0.3, 0.8, 0.3),
                      'regressor__max_depth': np.arange(5, 25, 5),
                      'regressor__num_leaves': np.arange(10, 100, 10),
                      'regressor__min_split_gain': np.arange(0.0, 0.5, 0.1),
                      'regressor__subsample': np.arange(0.5, 1.01, 0.1)
                      }

    elif model_name == "ElasticNet":
        param_grid = {'regressor__warm_start': [True, False],
                      'regressor__positive': [True, False],
                      'regressor__selection': ['cyclic', 'random']
                      }

    elif model_name == "Ridge":
        param_grid = {'regressor__alpha': np.arange(0.5, 5, 1),
                      'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                      'regressor__positive': [True, False]
                      }
    elif model_name == "Lasso":
        param_grid = {'regressor__warm_start': [True, False],
                      'regressor__positive': [True, False],
                      'regressor__selection': ['cyclic', 'random']
                      }
    elif model_name == "LassoLars":
        param_grid = {'regressor__fit_intercept': [True, False],
                      'regressor__positive': [True, False]
                      }

    elif model_name == "PassiveAggressiveRegressor":
        param_grid = {'regressor__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                      'regressor__warm_start': [True, False],
                      'regressor__average': [True, False],
                      }

    return param_grid
