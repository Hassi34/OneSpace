import os

from .common import get_unique_filename


def save_preprocessed_data(data, parent_dir, logs_dir):
    csv_file_name = get_unique_filename("preprocessed", ext = "csv")
    preprocessed_data_dir = os.path.join(parent_dir, logs_dir, "Preprocessing")
    path_to_preprocessed_data = os.path.join(preprocessed_data_dir, csv_file_name)
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    data.to_csv(path_to_preprocessed_data, index=False)
    print("\n")
    print("*****" * 13)
    print("Preprocessed Data has been saved at the following location :")
    print("*****" * 13)
    print(f"\n ==> {path_to_preprocessed_data}\n")
