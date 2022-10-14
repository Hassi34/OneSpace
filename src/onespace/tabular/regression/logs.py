import os

from .common import get_unique_filename


def save_preprocessed_data(data, parent_dir, logs_dir):
    csv_file_name = get_unique_filename("preprocessed", ext="csv")
    preprocessed_data_dir = os.path.join(parent_dir, logs_dir, "Preprocessing")
    path_to_preprocessed_data = os.path.join(
        preprocessed_data_dir, csv_file_name)
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    data.to_csv(path_to_preprocessed_data, index=False)
    print("\n")
    print("*****" * 13)
    print("Preprocessed Data has been saved at the following location :")
    print("*****" * 13)
    print(f"\n ==> {path_to_preprocessed_data}\n")


def save_regressor_comparision_before_training(data, parent_dir, logs_dir):
    csv_file_name = get_unique_filename(
        "modelsComparisionBeforTuning", ext=".csv")
    model_comparision_data_dir = os.path.join(
        parent_dir, logs_dir, "Model Comparision")
    path_to_model_comparision = os.path.join(
        model_comparision_data_dir, csv_file_name)
    os.makedirs(model_comparision_data_dir, exist_ok=True)
    data = data.reset_index().rename(columns={'index': 'Regressors'})
    data.to_csv(path_to_model_comparision, index=False)
    print("\n")
    print("*****" * 13)
    print("Complete model comparision results have been saved at following location\nPlease Check it out before proceeding further...")
    print("*****" * 13)
    print(f"\n ==> {path_to_model_comparision}\n")


def save_regressor_comparision_after_training(data, parent_dir, logs_dir):
    csv_file_name = get_unique_filename(
        "modelsComparisionAfterTuning", ext=".csv")
    model_comparision_data_dir = os.path.join(
        parent_dir, logs_dir, "Model Comparision")
    path_to_model_comparision = os.path.join(
        model_comparision_data_dir, csv_file_name)
    os.makedirs(model_comparision_data_dir, exist_ok=True)
    data = data.reset_index().rename(columns={'index': 'Regressors'})
    data.to_csv(path_to_model_comparision, index=False)
    print("\n")
    print("*****" * 13)
    print("Model comparision results after tuning have been saved at following location")
    print("*****" * 13)
    print(f"\n ==> {path_to_model_comparision}\n")
