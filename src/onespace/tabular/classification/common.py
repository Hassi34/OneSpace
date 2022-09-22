import sys
import time

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_unique_filename(filename, is_model_name=False, ext=None):
    if is_model_name:
        time_stamp = time.strftime("on_%Y%m%d_at_%H%M%S_.pkl")
    elif ext is not None:
        time_stamp = time.strftime("on_%Y%m%d_at_%H%M%S_."+ext)
    else:
        time_stamp = time.strftime("on_%Y%m%d_at_%H%M%S")
    unique_filename = f"{filename}_{time_stamp}"
    return unique_filename


def get_targets(targets):
    if targets.dtype == "object":
        target_names = targets.unique().tolist()
        encoder = LabelEncoder()
        targets = encoder.fit_transform(targets)
    else:
        targets = targets.astype(int)
        target_names = None
    return targets, target_names


def drop_zero_std(df):
    num_cols = df.select_dtypes(exclude='object').columns.tolist()
    data_described = df.describe()
    for col in num_cols:
        if data_described[col]['std'] == 0.0:
            df.drop(col, axis=1, inplace=True)
    return df


def get_data_and_features(df, auto):
    cat_features = df.select_dtypes(include=['object']).columns.tolist()
    num_features = df.select_dtypes(exclude=['object']).columns.tolist()
    cat_data = df[cat_features].astype('category')
    num_data = df[num_features].astype(float)
    df = pd.concat([cat_data, num_data], axis=1)
    print(f'==> Numerical Columns : {num_features}')
    print(f'==> Categorical Columns : {cat_features}')
    if auto == False:
        usr_rsp = input(
            '   == Please type "yes" if you agree with the above selection otherwise, type "no" : ')
        if usr_rsp.title() == "Yes":
            print(f'    == Continuing with default selection...')
        elif usr_rsp.title() == "No":
            num_features = input(
                "Enter a list of numerical features, hit enter if there are none : ")
            cat_features = input(
                "Enter a list of categorical features, hit enter if there are none : ")
            print(cat_features)
            print(type(cat_features))
            num_features = num_features[1:-1].split(',')
            cat_features = cat_features[1:-1].split(',')
            if len(cat_features[0]) > 0:
                cat_features = [(element.replace("'", "").replace(
                    '"', '')).strip() for element in cat_features]
            else:
                cat_features = []
            if len(num_features[0]) > 0:
                num_features = [(element.replace("'", "").replace(
                    '"', '')).strip() for element in num_features]
            else:
                num_features = []
            print(f'    == Numerical Features Selected : {num_features}')
            print(f'    == Categorical Features Selected : {cat_features}')
            
            cat_data = df[cat_features].astype('category')
            num_data = df[num_features].astype(float)
            df = pd.concat([cat_data, num_data], axis=1)
        else:
            raise ValueError(" Not a valid respose")
    print("\n")
    print("*****" * 13)
    print(f'Training Samples')
    print("*****" * 13)
    print(df.head())
    return df, cat_features, num_features


def keep_or_drop_id(cat_data, auto=True):
    try:
        unique = cat_data.nunique()/len(cat_data)
        id_col = unique[unique > 0.15].index.to_list()
        if len(id_col) > 0:
            if auto:
                print(f"==> Dropping the id columns {id_col}")
                cat_data = cat_data.drop(id_col, axis=1)
                return (id_col, cat_data.columns.tolist())
            else:
                print('==> Dropping the following id columns, please enter "yes" if you agree otherwise to keep the columns, type "no"')
                print(f"    ==ID columns to drop : {id_col}")
                usr_rsp = input("   Enter your response :")
                if usr_rsp.title() == "No":
                    print(f'    == We will NOT drop {id_col}')
                    return ([], cat_data.columns.tolist())
                elif usr_rsp.title() == "Yes":
                    cat_data = cat_data.drop(id_col, axis=1)
                    return (id_col, cat_data.columns.tolist())
                else:
                    print('!!! Not a valid response !!!')
                    sys.exit()
        else:
            print("==> No id column to drop")
            if cat_data.empty:
                return ([], [])
            else:
                return ([], cat_data)
    except TypeError:
        return ([], [])


def sort_by(metrics):
    if metrics == "accuracy":
        return "acc_val"
    elif metrics == "f1_score":
        return "f1_val"
    elif metrics == "recall":
        return "recall_val"
    elif metrics == "precision":
        return "precision_val"
    else:
        raise ValueError('"{}" is not a valid metrics'.format(metrics))
